"""Matched R sampler with uniform probability per resident EA transition."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.source_replay import build_sampling_cdf
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import (
    PilotCheckpointStore, mechanism_digest, restore_template,
)
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
from scripts.random_source_priority import RandomSourcePriorityWorkflow
from scripts.run_random_source_priority import ROOT as R_ROOT
from scripts.run_random_source_priority import RObserver, rows_for
from scripts.run_s_utility_extension import create_bootstrap_checkpoint, find_resume_checkpoint, validate_extension_config

ROOT = Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/uniform-overlay-cp100-v1")
SEEDS = (3083, 3079)
FINAL_GENERATION = 100
FINAL_EPISODES = 3290


def uniform_ea_distribution(distribution):
    """Uniform over EA transitions, hence source mass proportional to residency."""
    unit = jnp.reciprocal(jnp.maximum(distribution.n_ea, 1).astype(jnp.float32))
    probability = jnp.where(distribution.transition_probability > 0, unit, 0.0)
    baseline = distribution.baseline_source_probability
    ess = jnp.sum(baseline ** 2)
    ess = jnp.where(distribution.has_resident_ea, 1 / jnp.maximum(ess, 1e-30), 0.0)
    return distribution._replace(
        weighted_source_probability=baseline,
        source_probability=baseline,
        transition_probability=probability,
        transition_cdf=build_sampling_cdf(probability),
        effective_source_count=ess,
        ess_fraction=ess / jnp.maximum(distribution.resident_source_count, 1),
        max_source_probability=jnp.max(baseline),
        uniform_floor_per_transition=jnp.where(distribution.has_resident_ea, unit, 0.0),
        uniform_component_total_mass=distribution.has_resident_ea.astype(jnp.float32),
    )


class UniformOverlay(RandomSourcePriorityWorkflow):
    def _weighted_rl_update(self, agent_state, opt_state, replay_buffer_state, key,
                            num_updates, utility_replay_state):
        # Keep R's prepared flag and intervention key. The inherited learner
        # still calls the corrected weighted sampler and replaces EA slots.
        uniform_state = utility_replay_state._replace(
            distribution=uniform_ea_distribution(utility_replay_state.distribution),
        )
        return super()._weighted_rl_update(
            agent_state, opt_state, replay_buffer_state, key, num_updates, uniform_state,
        )


def verify_sampler(workflow, state, full_trace):
    replay, _, updates, utility = workflow.learner_input
    distribution = utility.distribution
    if not bool(utility.distribution_prepared):
        raise ValueError("EA overlay is inactive")
    np.testing.assert_array_equal(distribution.source_probability, distribution.baseline_source_probability)
    if not full_trace:
        return {}
    origin = np.asarray(replay.origin_type)
    resident_ea = (np.arange(len(origin)) < int(replay.buffer_size)) & (origin == 0)
    n_ea = int(resident_ea.sum())
    probability = np.asarray(distribution.transition_probability)
    np.testing.assert_array_equal(probability > 0, resident_ea)
    ea_probabilities = probability[resident_ea]
    np.testing.assert_array_equal(ea_probabilities, ea_probabilities[0])
    np.testing.assert_allclose(ea_probabilities[0], 1 / n_ea, rtol=np.finfo(np.float32).eps, atol=0)
    cdf = np.asarray(distribution.transition_cdf)
    assert cdf.dtype == np.float64 and cdf[-1] == 1
    increments = np.diff(cdf, prepend=0.0)
    cdf_tv = float(np.abs(increments - resident_ea / n_ea).sum() / 2)
    assert cdf_tv < 1e-9
    baseline, final = map(np.asarray, workflow.learner_trace[:2])
    assert baseline.shape[0] == updates and np.all((0 <= final) & (final < int(replay.buffer_size)))
    rl = origin[baseline] == 1
    np.testing.assert_array_equal(final[rl], baseline[rl])
    np.testing.assert_array_equal(origin[final], origin[baseline])
    changed_ea = int(np.count_nonzero((baseline != final) & ~rl))
    assert changed_ea > 0
    assert not np.array_equal(utility.intervention_key, state.utility_replay_state.intervention_key)
    return dict(cdf_tv=cdf_tv, changed_ea=changed_ea, changed_rl=0, n_ea=n_ea,
                critic_batches=int(np.prod(final.shape[:-1])),
                final_intervention_key=np.asarray(state.utility_replay_state.intervention_key).tolist())


class OverlayObserver(RObserver):
    workflow_type = UniformOverlay

    def __init__(self, plan, seed):
        super().__init__(plan, seed)
        self.identity.update(algorithm="U-Overlay", distribution="uniform_per_resident_ea_transition")

    def fork_state(self, workflow, run_id):
        parent_config = OmegaConf.load(self.parent.parents[3] / "resolved_config.yaml")
        comparable = copy.deepcopy(workflow.config)
        comparable.total_episodes = parent_config.total_episodes
        validate_extension_config(parent_config, comparable)
        assert int(workflow.config.total_episodes) == FINAL_EPISODES
        with PilotCheckpointStore(self.parent.parent, parent_config, read_only=True) as store:
            state, runtime = store.restore(10, restore_template(workflow))
        if initialization_fingerprint(state) != self.expected or runtime["evaluations"] or runtime["budget"]["steps"]:
            raise ValueError("frozen A CP10 mismatch")
        runtime = copy.deepcopy(runtime)
        runtime.update(run_id=run_id, target_generation=FINAL_GENERATION)
        create_bootstrap_checkpoint(self.segment, workflow.config, state, runtime)
        return state, runtime

    def record(self, workflow, state, checkpoint, runtime):
        generation = int(state.metrics.iterations)
        audit = verify_sampler(workflow, state, generation in (20, FINAL_GENERATION))
        workflow.priority_assignment.update(
            sampling_distribution="uniform_per_resident_ea_transition", overlay_audit=audit,
        )
        super().record(workflow, state, checkpoint, runtime)
        if generation % 10 == 0:
            print(f"CP{generation} verified", flush=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare():
    ROOT.mkdir(exist_ok=False)
    r_plan = json.loads((R_ROOT / "preregistration.json").read_text())
    for seed in SEEDS:
        protocol = ROOT / f"seed_{seed}" / "protocol"
        protocol.mkdir(parents=True)
        configs = []
        for arm in ("uniform", "fitness", "td", "utility"):
            name = f"arm_{arm}.yaml"
            config = OmegaConf.load(R_ROOT / f"seed_{seed}" / "protocol" / name)
            config.total_episodes = FINAL_EPISODES
            config.output_dir = str(protocol.parent / "runs" / arm)
            config.arm_manifest_path = str(protocol / "arm_manifest.json")
            OmegaConf.save(config, protocol / name, resolve=True)
            configs.append(config)
        prepare_arm_manifest(configs, protocol / "arm_manifest.json")
    write_json(ROOT / "preregistration.json", dict(
        algorithm="U-Overlay", seeds=r_plan["seeds"], seed_order=SEEDS,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        mechanism_digest=mechanism_digest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary="Return AUC CP20-100 over 650000-3290000 training environment steps",
        final_generation=FINAL_GENERATION, threshold="first scheduled mean Return >= 350 within CP100",
        evaluation="CP20-100 every 10 generations, 128 episodes each",
        training_timing="synchronized post-CP10 workflow steps, including compilation but excluding evaluation and checkpoint I/O",
        mechanism="retain R scheduling, bookkeeping, EA overlay and intervention RNG, replace sampling probability with uniform EA transition mass",
        utility_calls=0,
    ))


def smoke():
    import evorl.algorithms.erl.erl_td3.erl_learner_utility as utility_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Utility/source minibatch invoked")

    utility_module.virtual_critic_utility = forbidden
    utility_module.build_source_batches = forbidden
    config = OmegaConf.load(ROOT / "seed_3083/protocol/arm_uniform.yaml")
    config.env.max_episode_steps = 8
    config.replay_buffer_capacity = 4096
    config.batch_size = 8
    config.source_batch_size = 4
    config.agent_network.actor_hidden_layer_sizes = [8, 8]
    config.agent_network.critic_hidden_layer_sizes = [8, 8]
    base = type("SmokeUniform", (AuditedD0,), {}).build_from_config(config, enable_jit=True)
    initial = base.init(jax.random.PRNGKey(3083))
    config.arm = config.score_kind = "utility"
    config.alpha = .5
    config.diagnostic_enabled = True
    workflow = type("SmokeOverlay", (UniformOverlay,), {}).build_from_config(config, enable_jit=True)
    _, state = workflow.step(initial)
    audit = verify_sampler(workflow, state, True)
    base.close()
    workflow.close()
    write_json(ROOT / "smoke.json", dict(status="PASS", actual_brax=True, utility_calls=0, **audit))
    print("GPU SMOKE PASS", audit, flush=True)


def train(seed):
    assert (ROOT / "smoke.json").exists()
    plan = json.loads((ROOT / "preregistration.json").read_text())
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if plan["runner_sha256"] != runner_sha256:
        revision = json.loads((ROOT / "verification-revision.json").read_text())
        assert revision["original_runner_sha256"] == plan["runner_sha256"]
        assert revision["revised_runner_sha256"] == runner_sha256
    assert plan["mechanism_digest"] == mechanism_digest()
    folder = ROOT / f"seed_{seed}"
    observer = OverlayObserver(plan, seed)
    config = OmegaConf.load(folder / "protocol/arm_utility.yaml")
    output = folder / "runs/U"
    resume = find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:
        raise ValueError("No sealed checkpoint for existing run")
    started = time.perf_counter()
    state, report = run_training(config, output, checkpoint_interval=25, keep=5,
                                 ablation=observer, fork=resume is None, resume_checkpoint=resume)
    rows = rows_for(output)
    assert [row["generation"] for row in rows] == list(range(11, 101))
    assert sum(row["new_priorities"] for row in rows) == 288
    assert all(row["ea_conditional_tv"] == 0 for row in rows)
    rrows = rows_for(R_ROOT / f"seed_{seed}/runs/R")
    assert [r["selected_individuals"] for r in rows] == [r["selected_individuals"] for r in rrows[:90]]
    assert report["final"]["timesteps"] == 3290000 and report["final"]["generation"] == 100
    assert [e["generation"] for e in report["evaluations"]] == list(range(20, 101, 10))
    write_json(folder / "summary-U.json", dict(
        algorithm="U-Overlay", seed=seed, performance=report, utility_calls=0,
        process_wall_seconds=time.perf_counter() - started,
        training_step_seconds=sum(row["step_seconds"] for row in rows),
        audits={str(r["generation"]): r["overlay_audit"] for r in rows if r["overlay_audit"]},
        final_intervention_key=np.asarray(state.utility_replay_state.intervention_key).tolist(),
    ))
    print("COMPLETE", seed, report["auc"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3083)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        assert jax.default_backend() == "gpu"
        smoke() if args.stage == "smoke" else train(args.seed)


if __name__ == "__main__":
    main()
