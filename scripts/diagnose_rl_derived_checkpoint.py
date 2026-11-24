"""Read-only frozen diagnostics from sealed uniform ERL full-state checkpoints."""
import argparse, hashlib, json, re, time, importlib.metadata, platform
from pathlib import Path
import jax
import numpy as np
import orbax.checkpoint as ocp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from evorl.types import State
from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import ERLUtilitySourceControlsWorkflow
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import erl_replace_td3_actor_params
from evorl.algorithms.erl.erl_td3.utility_source_controls import tree_fingerprint

def read_path(raw, path):
    for flat, quoted, index, attr in re.findall(r"\[<flat index (\d+)>\]|\['([^']+)'\]|\[(\d+)\]|\.([A-Za-z_][A-Za-z_0-9]*)", path):
        raw = raw[int(flat or index)] if flat or index else raw[quoted or attr]
    return raw

def verify_restore(path):
    metadata = json.loads((path / "runtime/metadata").read_text())
    seal = path.parent.parent / "checkpoint_checksums" / (path.name + ".json")
    expected = json.loads(seal.read_text())
    actual = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in path.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("checkpoint file seal mismatch")
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as reader:
        raw = reader.restore(str(path / "state"))
    digest = hashlib.sha256()
    for spec in metadata["leaf_specs"]:
        value = np.asarray(jax.device_get(read_path(raw, spec["path"])))
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError("restored leaf specification mismatch")
        digest.update(spec["path"].encode())
        digest.update(str((value.dtype, value.shape)).encode())
        digest.update(value.tobytes())
    if digest.hexdigest() != metadata["state_sha256"]:
        raise ValueError("full historical state hash mismatch")
    return raw, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    raw, metadata = verify_restore(args.checkpoint)
    historical = OmegaConf.load(args.historical_config)
    manifest = json.loads((args.historical_config.parent / "manifest.json").read_text())
    config_sha = hashlib.sha256(args.historical_config.read_bytes()).hexdigest()
    if config_sha != manifest["resolved_config_sha256"]:
        raise ValueError("historical configuration checksum mismatch")
    if historical.arm != "uniform" or float(historical.alpha) != 0:
        raise ValueError("only historical uniform baseline checkpoints are supported")
    contract = OmegaConf.to_container(historical, resolve=True)
    for key in ("output_dir", "arm_manifest_path", "checkpoint", "recorders", "project", "tags"):
        contract.pop(key, None)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != metadata["config_contract"]:
        raise ValueError("historical checkpoint/config contract mismatch")
    environment = metadata["environment"]
    if any(importlib.metadata.version(name) != value for name, value in environment["packages"].items()):
        raise ValueError("checkpoint package versions differ")
    if (platform.python_version() != environment["python"] or
        jax.default_backend() != environment["backend"] or
        bool(jax.config.jax_enable_x64) != environment["x64"] or
        str(jax.config.jax_default_prng_impl) != environment["prng"]):
        raise ValueError("checkpoint runtime contract mismatch")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name="config", overrides=["env=brax/swimmer", "agent=erl/erl-utility-source-controls"])
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(args.historical_config), {
        "diagnostic_enabled": True, "derived_enabled": True,
        "reference_target_bias_enabled": False, "diagnostic_interval": 1,
        "derived_small_strength": 0.005, "derived_large_strength": 0.05,
        "artifact_root": str(args.output / "artifacts"),
        "workflow_cls": "evorl.algorithms.erl.erl_td3.erl_utility_source_controls.ERLUtilitySourceControlsWorkflow",
        "recorders": [], "output_dir": str(args.output),
    })
    OmegaConf.save(config, args.output / "resolved_config.yaml")
    cls = type("FrozenHistoricalDiagnostic", (ERLUtilitySourceControlsWorkflow,), {})
    workflow = cls.build_from_config(config, enable_jit=True)
    def template(root):
        key, agent_key, rb_key = jax.random.split(root, 3)
        agent, opt, ec = workflow._setup_agent_and_optimizer(agent_key)
        return State(agent_state=agent, ec_opt_state=ec, key=key,
                     metrics=workflow._setup_workflow_metrics(), opt_state=opt,
                     replay_buffer_state=workflow._setup_replaybuffer(rb_key))
    abstract = jax.eval_shape(template, jax.random.PRNGKey(int(config.seed)))
    pairs, tree = jax.tree_util.tree_flatten_with_path(abstract)
    values = []
    for path, shape in pairs:
        value = read_path(raw, jax.tree_util.keystr(path))
        if value.shape != shape.shape or value.dtype != shape.dtype:
            raise ValueError("historical native component mismatch: " + jax.tree_util.keystr(path))
        values.append(value)
    state = jax.tree_util.tree_unflatten(tree, values)
    if int(state.metrics.iterations) != int(args.checkpoint.name):
        raise ValueError("historical generation mismatch")
    before = tree_fingerprint(state)
    # Diagnose a hypothetical next rollout at g+1 using only the saved Replay at g.
    # Select fixed first four original population members; no EA or TD3 tell/update.
    context = workflow._make_pre_exposure_context(state)
    if not context.controls_eligible:
        raise ValueError(context.controls_reason)
    workflow._diagnostic_context, workflow._diagnostic_active = context, True
    workflow.config.pop_size = 4
    pop, _ = workflow.ec_optimizer.ask(state.ec_opt_state)
    pop = jax.tree.map(lambda x: x[:4], pop)
    _, ea_key, rl_key, _ = jax.random.split(state.key, 4)
    ea_metrics, _, ephemeral_replay = workflow._ec_rollout(
        erl_replace_td3_actor_params(state.agent_state, pop),
        state.replay_buffer_state, ea_key)
    rl_metrics, _, _ = workflow._rl_rollout(state.agent_state, ephemeral_replay, rl_key)
    if tree_fingerprint(state) != before:
        raise ValueError("historical baseline state mutated")
    workflow._source_control_writer.append_checkpoint(
        workflow._latest_source_control_diagnostic,
        checkpoint_provenance=workflow._latest_source_control_provenance)
    workflow.close()
    evidence = dict(
        checkpoint=str(args.checkpoint), historical_generation=int(state.metrics.iterations),
        diagnostic_generation=context.generation, historical_state_sha256=metadata["state_sha256"],
        file_seal_verified=True, full_state_hash_verified=True, native_shapes_verified=True,
        baseline_unchanged=True, baseline_fingerprint=before,
        reference_timing="Replay captured at historical g, before hypothetical g+1 exposure",
        original_ea_selection="first four saved population members",
        reference_fingerprints=list(context.snapshot.reference_fingerprints),
        historical_config_sha256=config_sha,
        rl_fitness=float(jax.device_get(rl_metrics.episode_returns.mean())),
        elapsed_seconds=time.monotonic()-started,
        note="All 4 EA + 1 RL + 3 EC-control rollouts are offline diagnostic interactions; expected 8000 steps",
    )
    # Count real lengths: 4 EA, 1 RL and 3 counterfactual policies.
    evidence["extra_environment_steps"] = int(jax.device_get(ea_metrics.episode_lengths.sum())) + int(jax.device_get(rl_metrics.episode_lengths.sum())) + context.derived_timesteps
    (args.output / "restore_evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)

if __name__ == "__main__":
    main()
