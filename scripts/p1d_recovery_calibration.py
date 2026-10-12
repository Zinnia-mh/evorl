"""Pilot-only P1-D common-pre-step recovery harness.

This module is inert on import. A real run is an R2 action, not an R1 test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np

REQUIRED_STATE = (
    "agent_state", "opt_state", "ec_opt_state", "key",
    "replay_buffer_state", "metrics",
)
OUTPUT_PARENT = Path(
    "/disk1/zmh_projects/evorl_utility_source_controls_outputs/"
    "reference_target_bias_pilot/calibration"
)


class RecoveryHold(RuntimeError):
    """A named fail-closed correctness gate."""

    def __init__(self, reason: str, field: str | None = None,
                 details: dict | None = None):
        self.reason = reason
        self.field = field
        self.details = details or {}
        super().__init__(f"{reason}: {field}" if field else reason)


def _host_array(value: Any) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    if array.dtype.kind not in "biufc":
        raise RecoveryHold("unsupported_state_leaf", str(array.dtype))
    return array


def _hash_tree(tree: Any, *, occupied: int | None = None) -> str:
    leaves, structure = jtu.tree_flatten(tree)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        if occupied is not None:
            if not hasattr(leaf, "shape") or len(leaf.shape) < 1 or leaf.shape[0] < occupied:
                raise RecoveryHold("invalid_replay_structure")
            array = _host_array(leaf[:occupied])
        else:
            array = _host_array(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def occupied_replay_fingerprint(replay_state: Any) -> str:
    """Hash real occupied physical slots, never the jnp.empty_like tail.

    ReplayBuffer.add maintains occupied slots [0,size) until full, then all
    physical slots remain occupied through ring overwrite. This is a state
    equality check, not an insertion/version identity.
    """
    for name in ("data", "current_index", "buffer_size"):
        if not hasattr(replay_state, name):
            raise RecoveryHold("incomplete_replay_state", name)
    leaves = jtu.tree_leaves(replay_state.data)
    if not leaves:
        raise RecoveryHold("invalid_replay_structure", "no data leaves")
    if any(not hasattr(leaf, "shape") or len(leaf.shape) < 1 for leaf in leaves):
        raise RecoveryHold("invalid_replay_structure", "scalar data leaf")
    capacities = {int(leaf.shape[0]) for leaf in leaves}
    if len(capacities) != 1:
        raise RecoveryHold("invalid_replay_structure", "capacity")
    capacity = capacities.pop()
    size = int(_host_array(replay_state.buffer_size))
    pointer = int(_host_array(replay_state.current_index))
    if capacity <= 0 or not 0 <= size <= capacity or not 0 <= pointer < capacity:
        raise RecoveryHold("invalid_replay_metadata")
    if size < capacity and pointer != size:
        raise RecoveryHold("replay_pointer", f"size={size},pointer={pointer}")
    digest = hashlib.sha256()
    digest.update(f"size={size};pointer={pointer};capacity={capacity}".encode())
    digest.update(_hash_tree(replay_state.data, occupied=size).encode())
    return digest.hexdigest()


def training_state_fingerprints(state: Any) -> dict[str, str]:
    if not hasattr(state, "keys"):
        raise RecoveryHold("incomplete_state", "not a State mapping")
    extras = set(state.keys()) - set(REQUIRED_STATE)
    if extras:
        raise RecoveryHold("unexpected_state_fields", ",".join(sorted(extras)))
    if not all(field in state and hasattr(state, field) for field in REQUIRED_STATE):
        missing = [field for field in REQUIRED_STATE
                   if field not in state or not hasattr(state, field)]
        raise RecoveryHold("incomplete_state", ",".join(missing))
    result = {}
    for field in REQUIRED_STATE:
        if field == "replay_buffer_state":
            result[field] = occupied_replay_fingerprint(getattr(state, field))
        else:
            result[field] = _hash_tree(getattr(state, field))
    return result


def compare_training_states(a: Any, b: Any, *,
                            reason: str = "common_pre_step_state_mismatch"
                            ) -> dict[str, str]:
    left, right = training_state_fingerprints(a), training_state_fingerprints(b)
    for field in REQUIRED_STATE:
        if left[field] != right[field]:
            raise RecoveryHold(reason, field, {"left": left, "right": right})
    return left


def validate_pair_configs(off: Any, on: Any) -> None:
    from omegaconf import OmegaConf
    def data(value):
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise RecoveryHold("incomplete_config")
        value = dict(value)
        for name in ("diagnostic_enabled", "reference_target_bias_enabled",
                     "pilot_artifact_root"):
            value.pop(name, None)
        return value
    if data(off) != data(on):
        raise RecoveryHold("config_mismatch")
    def get(value, key):
        return value.get(key) if isinstance(value, dict) else getattr(value, key, None)
    if get(off, "diagnostic_enabled") is not False or get(off, "reference_target_bias_enabled") is not False:
        raise RecoveryHold("config_mismatch", "OFF flags")
    if get(on, "diagnostic_enabled") is not True or get(on, "reference_target_bias_enabled") is not True:
        raise RecoveryHold("config_mismatch", "ON flags")


def audit_workflow_contexts(off: Any, on: Any) -> dict:
    if off is on:
        raise RecoveryHold("workflow_not_isolated")
    components = (
        "env", "agent", "ec_collector", "rl_collector", "evaluator",
        "replay_buffer", "ec_optimizer", "optimizer", "config",
    )
    checked = []
    for name in components:
        left, right = getattr(off, name, None), getattr(on, name, None)
        if left is not None and right is not None:
            if left is right:
                raise RecoveryHold("workflow_context_shared", name)
            checked.append(name)
    return {"distinct_components": checked,
            "off_attributes": sorted(off.__dict__),
            "on_attributes": sorted(on.__dict__)}


def _counts(workflow: Any) -> dict[str, int]:
    observed = getattr(workflow, "_recovery_counts", None)
    if observed is not None:
        return dict(observed)
    return {
        "collectors": int(getattr(workflow, "collectors", -1)),
        "replay_adds": int(getattr(workflow, "replay_adds", -1)),
    }


def _step(workflow: Any, pre: Any) -> tuple[Any, dict[str, int], float]:
    start = time.perf_counter()
    _, post = workflow.step(pre)
    for leaf in jtu.tree_leaves(post):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()
    return post, _counts(workflow), time.perf_counter() - start


def _environment_counts(pre: Any, post: Any) -> dict[str, int]:
    try:
        total = int(_host_array(post.metrics.sampled_timesteps)) - int(
            _host_array(pre.metrics.sampled_timesteps)
        )
        rl = int(_host_array(post.metrics.rl_sampled_timesteps)) - int(
            _host_array(pre.metrics.rl_sampled_timesteps)
        )
        episodes = int(_host_array(post.metrics.sampled_episodes)) - int(
            _host_array(pre.metrics.sampled_episodes)
        )
    except AttributeError as error:
        raise RecoveryHold("incomplete_state", "environment counters") from error
    ea = total - rl
    if min(total, rl, ea, episodes) < 0:
        raise RecoveryHold("invalid_environment_counters")
    return {"total_steps": total, "ea_steps": ea, "rl_steps": rl,
            "episodes": episodes}


def run_pair(off: Any, on: Any, pre: Any, *,
             on_pre: Any | None = None, off_repeat: Any | None = None) -> dict:
    """Run exactly one step per arm from the exact shared State PyTree.

    off_repeat is solely for synthetic R1 determinism tests; the real R2
    entrypoint does not run a third environment step.
    """
    if off is on:
        raise RecoveryHold("workflow_not_isolated")
    if int(_host_array(pre.metrics.iterations)) != 10:
        raise RecoveryHold("wrong_pre_step_generation")
    same_pre = compare_training_states(pre, pre if on_pre is None else on_pre)
    if on_pre is not None and on_pre is not pre:
        raise RecoveryHold("common_pre_step_object_mismatch")
    original = training_state_fingerprints(pre)
    off_post, off_counts, off_time = _step(off, pre)
    off_post_hashes = training_state_fingerprints(off_post)
    off_environment = _environment_counts(pre, off_post)
    if training_state_fingerprints(pre) != original:
        raise RecoveryHold("pre_step_state_mutation", "OFF")
    if off_repeat is not None:
        repeat_post, _, _ = _step(off_repeat, pre)
        compare_training_states(
            off_post, repeat_post, reason="baseline_replay_nondeterminism"
        )
        del repeat_post
        if training_state_fingerprints(pre) != original:
            raise RecoveryHold("pre_step_state_mutation", "OFF repeat")
    del off_post
    import gc
    gc.collect()
    compare_training_states(pre, pre if on_pre is None else on_pre)
    on_post, on_counts, on_time = _step(on, pre if on_pre is None else on_pre)
    on_environment = _environment_counts(pre, on_post)
    if training_state_fingerprints(pre) != original:
        raise RecoveryHold("pre_step_state_mutation", "ON")
    for name in ("collectors", "ec_collectors", "rl_collectors", "replay_adds"):
        if name not in off_counts and name not in on_counts:
            if name in ("ec_collectors", "rl_collectors"):
                continue  # synthetic minimal workflow; real arms report both
            raise RecoveryHold("missing_call_counts", name)
        if name not in off_counts or name not in on_counts:
            raise RecoveryHold("missing_call_counts", name)
        if off_counts[name] < 0 or on_counts[name] < 0:
            raise RecoveryHold("missing_call_counts", name)
        if off_counts[name] != on_counts[name]:
            raise RecoveryHold(
                "replay_add_call_mismatch" if name == "replay_adds"
                else "collector_call_mismatch", name
            )
    if off_environment != on_environment:
        raise RecoveryHold("environment_step_mismatch", details={
            "off": off_environment, "on": on_environment,
        })
    post_hashes = training_state_fingerprints(on_post)
    for field in REQUIRED_STATE:
        if off_post_hashes[field] != post_hashes[field]:
            raise RecoveryHold(
                "on_off_baseline_mismatch", field,
                {"off": off_post_hashes, "on": post_hashes},
            )
    del on_post
    return {
        "status": "PASS", "generation": 11,
        "common_pre_step_state": True,
        "pre": {"off": same_pre, "on": same_pre},
        "post": {"off": off_post_hashes, "on": post_hashes},
        "baseline_post_step_equal": True,
        "call_counts": {"off": off_counts, "on": on_counts},
        "replay_add_count_basis": "one add per inspected EC/RL rollout host boundary",
        "environment_counts": {"off": off_environment, "on": on_environment},
        "step_seconds": {"off": off_time, "on": on_time},
    }


def _instrument_rollouts(workflow: Any) -> None:
    counts = {"collectors": 0, "replay_adds": 0,
              "ec_collectors": 0, "rl_collectors": 0}
    for method, key in (("_ec_rollout", "ec_collectors"),
                        ("_rl_rollout", "rl_collectors")):
        original = getattr(workflow, method)
        def wrapped(*args, _original=original, _key=key, **kwargs):
            result = _original(*args, **kwargs)
            counts["collectors"] += 1
            counts[_key] += 1
            # Both real rollout paths call ReplayBuffer.add exactly once.
            # This count follows those host boundaries and is checked against
            # Replay pointer/size and occupied data in the post-state gate.
            counts["replay_adds"] += 1
            return result
        setattr(workflow, method, wrapped)
    workflow._recovery_counts = counts


def read_pilot_summary(pilot_run_dir: Path) -> dict:
    path = pilot_run_dir / "paired_summary.jsonl"
    if not path.is_file():
        raise RecoveryHold("missing_pilot_summary")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    except (ValueError, OSError) as error:
        raise RecoveryHold("invalid_pilot_summary") from error
    if len(rows) != 1 or rows[0].get("generation") != 11:
        raise RecoveryHold("invalid_pilot_summary")
    row = rows[0]
    if row.get("status") != "PASS":
        raise RecoveryHold("pilot_diagnostic_hold", ",".join(row.get("reasons", [])))
    return row


def _real_entry(run_dir: Path, overrides: list[str]) -> dict:
    """The R2-only CLI. Importing/testing this module does not invoke it."""
    import copy
    import hydra
    from omegaconf import OmegaConf
    from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import (
        ERLUtilitySourceControlsWorkflow,
    )
    if run_dir.exists() or not run_dir.is_relative_to(OUTPUT_PARENT):
        raise RecoveryHold("unsafe_output_directory", str(run_dir))
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        base = hydra.compose(
            config_name="config",
            overrides=["env=brax/swimmer",
                       "agent=erl/erl-utility-source-controls", *overrides],
        )
    if int(base.seed) != 42 or int(base.warmup_iters) != 10:
        raise RecoveryHold("config_mismatch", "seed/warmup")
    if int(base.diagnostic_interval) != 11:
        raise RecoveryHold("config_mismatch", "diagnostic_interval")
    off_config, on_config = copy.deepcopy(base), copy.deepcopy(base)
    for cfg, enabled in ((off_config, False), (on_config, True)):
        cfg.diagnostic_enabled = enabled
        cfg.reference_target_bias_enabled = enabled
        cfg.output_dir = str(run_dir)
        cfg.recorders = []
        cfg.checkpoint.enable = False
    OmegaConf.update(on_config, "pilot_artifact_root",
                     str(run_dir / "pilot"), force_add=True)
    validate_pair_configs(off_config, on_config)
    run_dir.mkdir(parents=True, exist_ok=False)
    workflow_cls = type("P1DRecoveryWorkflow", (ERLUtilitySourceControlsWorkflow,), {})
    off = workflow_cls.build_from_config(off_config, enable_jit=True)
    on = None
    try:
        # Workflow.init runs the sole real ten-iteration warmup prefix.
        pre = off.init(jax.random.PRNGKey(42))
        if int(_host_array(pre.metrics.iterations)) != 10:
            raise RecoveryHold("wrong_pre_step_generation")
        captured_pre = training_state_fingerprints(pre)
        on = workflow_cls.build_from_config(on_config, enable_jit=False)
        if training_state_fingerprints(pre) != captured_pre:
            raise RecoveryHold("pre_step_state_mutation", "ON workflow build")
        validate_pair_configs(off.config, on.config)
        if off._diagnostic_due(pre) or not on._diagnostic_due(pre):
            raise RecoveryHold("diagnostic_due_mismatch")
        internal = audit_workflow_contexts(off, on)
        _instrument_rollouts(off)
        _instrument_rollouts(on)
        audit = run_pair(off, on, pre)
        audit["workflow_internal_state_audit"] = internal
        audit["pilot_artifact_directory"] = str(on._pilot_writer.root)
        try:
            summary = read_pilot_summary(on._pilot_writer.root)
            audit["pilot_status"] = summary["status"]
        except RecoveryHold as error:
            audit["status"] = "HOLD"
            audit["pilot_status"] = "HOLD"
            audit["reason"] = error.reason
            audit["field"] = error.field
        return audit
    finally:
        off.close()
        if on is not None:
            on.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="new, unique directory under the pilot calibration root")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve(strict=False)
    if run_dir.exists() or not run_dir.is_relative_to(OUTPUT_PARENT):
        raise SystemExit("HOLD: unsafe_output_directory")
    result = None
    try:
        result = _real_entry(run_dir, args.overrides)
    except RecoveryHold as error:
        result = {"status": "HOLD", "reason": error.reason,
                  "field": error.field, "details": error.details}
    except Exception as error:
        import traceback
        result = {"status": "HOLD", "reason": "unexpected_exception",
                  "error": repr(error), "traceback": traceback.format_exc()}
    finally:
        if run_dir.exists() and result is not None:
            (run_dir / "state_audit.json").write_text(
                json.dumps(result, indent=2, sort_keys=True)
            )
    return 0 if result and result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
