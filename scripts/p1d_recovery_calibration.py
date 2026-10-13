"""Pilot-only P1-D common-pre-step recovery harness.

This module is inert on import. A real run is an R2 action, not an R1 test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp

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


def state_leaf_differences(left: Any, right: Any) -> list[dict]:
    """Scalar device reductions only; Replay payload is limited to occupied slots."""
    differences = []
    def compare_tree(field: str, a: Any, b: Any, occupied: int | None = None):
        la, sa = jtu.tree_flatten_with_path(a)
        lb, sb = jtu.tree_flatten_with_path(b)
        if sa != sb or len(la) != len(lb):
            differences.append({"path": field, "reason": "tree_structure_mismatch"})
            return
        for (path, av), (_, bv) in zip(la, lb):
            cursor = a
            parts = [field]
            for key in path:
                if isinstance(cursor, dict) and hasattr(key, "key") and isinstance(key.key, int):
                    name = sorted(cursor)[key.key]
                elif hasattr(key, "idx"):
                    name = sorted(cursor)[key.idx] if isinstance(cursor, dict) else key.idx
                elif hasattr(key, "key"):
                    name = key.key
                elif hasattr(key, "name"):
                    name = key.name
                else:
                    name = str(key)
                parts.append(str(name))
                cursor = cursor[name] if isinstance(cursor, dict) else (
                    cursor[name] if isinstance(name, int) else getattr(cursor, name)
                )
            label = ".".join(parts)
            aa, bb = jnp.asarray(av), jnp.asarray(bv)
            if occupied is not None:
                aa, bb = aa[:occupied], bb[:occupied]
            entry = {"path": label, "shape": list(aa.shape),
                     "dtype": str(aa.dtype), "exact_equal": False}
            if aa.shape != bb.shape or aa.dtype != bb.dtype:
                entry["reason"] = "shape_or_dtype_mismatch"
                entry["other_shape"] = list(bb.shape)
                entry["other_dtype"] = str(bb.dtype)
                differences.append(entry)
                continue
            equal = jnp.equal(aa, bb)
            count = int(jax.device_get(jnp.count_nonzero(jnp.logical_not(equal))))
            if count == 0:
                continue
            entry["different_elements"] = count
            if aa.dtype.kind in "fc" and bb.dtype.kind in "fc":
                entry["all_finite"] = bool(jax.device_get(
                    jnp.all(jnp.isfinite(aa)) & jnp.all(jnp.isfinite(bb))))
                entry["max_absolute_difference"] = float(jax.device_get(
                    jnp.max(jnp.abs(aa - bb), initial=0)))
            else:
                entry["all_finite"] = True
                entry["max_absolute_difference"] = None
            differences.append(entry)
    for field in REQUIRED_STATE:
        if field != "replay_buffer_state":
            compare_tree(field, getattr(left, field), getattr(right, field))
            continue
        a, b = left.replay_buffer_state, right.replay_buffer_state
        compare_tree(field + ".buffer_size", a.buffer_size, b.buffer_size)
        compare_tree(field + ".current_index", a.current_index, b.current_index)
        size_a, size_b = int(_host_array(a.buffer_size)), int(_host_array(b.buffer_size))
        if size_a != size_b:
            differences.append({"path": field + ".data",
                                "reason": "occupied_size_mismatch",
                                "left_size": size_a, "right_size": size_b})
        else:
            compare_tree(field + ".data", a.data, b.data, occupied=size_a)
    return differences


def first_divergent_stage(off: dict, on: dict) -> str | None:
    """Order follows ERLOriginWorkflow.step, including add inside rollouts."""
    order = (
        "ea_rollout_input", "ea_replay_add", "ea_rollout_output", "ea_rollout",
        "rl_rollout_input", "rl_replay_add", "rl_rollout_output", "rl_rollout",
        "rl_learner_update_input", "rl_learner_update",
        "ea_evolution_update_input", "ea_evolution_update",
    )
    for name in order:
        if name in off or name in on:
            if off.get(name) != on.get(name):
                return name
    return None


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
    audit = {"pre_state_fingerprints": same_pre}
    if on_pre is not None and on_pre is not pre:
        raise RecoveryHold("common_pre_step_object_mismatch")
    original = training_state_fingerprints(pre)
    off_post, off_counts, off_time = _step(off, pre)
    off_post_hashes = training_state_fingerprints(off_post)
    audit["off_post_fingerprints"] = off_post_hashes
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
    # A new PyTree container keeps device leaves for leaf-level attribution,
    # while releasing the OFF State object before invoking ON.
    off_snapshot = jtu.tree_map(lambda leaf: leaf, off_post)
    del off_post
    import gc
    gc.collect()
    compare_training_states(pre, pre if on_pre is None else on_pre)
    on_post, on_counts, on_time = _step(on, pre if on_pre is None else on_pre)
    on_environment = _environment_counts(pre, on_post)
    audit["on_post_fingerprints"] = training_state_fingerprints(on_post)
    audit["call_counts"] = {"off": off_counts, "on": on_counts}
    audit["environment_counts"] = {"off": off_environment, "on": on_environment}
    audit["step_seconds"] = {"off": off_time, "on": on_time}
    audit["available_stage_fingerprints"] = {
        "off": getattr(off, "_recovery_stage_fingerprints", {}),
        "on": getattr(on, "_recovery_stage_fingerprints", {}),
    }
    audit["first_divergent_stage"] = first_divergent_stage(
        audit["available_stage_fingerprints"]["off"],
        audit["available_stage_fingerprints"]["on"],
    )
    if training_state_fingerprints(pre) != original:
        raise RecoveryHold("pre_step_state_mutation", "ON", {"audit": audit})
    for name in ("collectors", "ec_collectors", "rl_collectors", "replay_adds"):
        if name not in off_counts and name not in on_counts:
            if name in ("ec_collectors", "rl_collectors"):
                continue  # synthetic minimal workflow; real arms report both
            raise RecoveryHold("missing_call_counts", name, {"audit": audit})
        if name not in off_counts or name not in on_counts:
            raise RecoveryHold("missing_call_counts", name, {"audit": audit})
        if off_counts[name] < 0 or on_counts[name] < 0:
            raise RecoveryHold("missing_call_counts", name, {"audit": audit})
        if off_counts[name] != on_counts[name]:
            raise RecoveryHold(
                "replay_add_call_mismatch" if name == "replay_adds"
                else "collector_call_mismatch", name, {"audit": audit}
            )
    if off_environment != on_environment:
        raise RecoveryHold("environment_step_mismatch", details={
            "off": off_environment, "on": on_environment, "audit": audit,
        })
    post_hashes = audit["on_post_fingerprints"]
    mismatched = [field for field in REQUIRED_STATE
                  if off_post_hashes[field] != post_hashes[field]]
    if mismatched:
        audit["all_mismatched_fields"] = mismatched
        audit["first_mismatched_field"] = mismatched[0]
        audit["leaf_differences"] = state_leaf_differences(off_snapshot, on_post)
        raise RecoveryHold("on_off_baseline_mismatch", mismatched[0],
                           {"off": off_post_hashes, "on": post_hashes,
                            "audit": audit})
    del off_snapshot, on_post
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


def instrument_baseline_stages(workflow: Any):
    """Observe eager boundaries of the existing step without extra execution.

    The actual ERLOriginWorkflow.step order is EA rollout/add, RL
    rollout/add, RL learner update, then EA evolution update. No Python
    callback is inserted inside a traced collector or compiled JAX kernel.
    """
    records: dict[str, dict] = {}
    workflow._recovery_stage_fingerprints = records
    replay = workflow.replay_buffer
    original_add = replay.add
    add_calls = 0

    def observed_add(buffer_state, xs, mask=None):
        nonlocal add_calls
        add_calls += 1
        name = ("ea_replay_add" if add_calls == 1 else
                "rl_replay_add" if add_calls == 2 else
                f"unexpected_replay_add_{add_calls}")
        count = (int(_host_array(mask.sum())) if mask is not None else
                 int(jtu.tree_leaves(xs)[0].shape[0]))
        record = {
            "replay_before": occupied_replay_fingerprint(buffer_state),
            "add_input": _hash_tree(xs),
            "mask": _hash_tree(mask) if mask is not None else None,
            "valid_transition_count": count,
        }
        result = original_add(buffer_state, xs, mask)
        record["replay_after"] = occupied_replay_fingerprint(result)
        records[name] = record
        return result
    # ReplayBuffer is a frozen PyTreeNode. Shadow only this instance's method
    # for the one audited step, then remove the shadow in the caller's finally.
    object.__setattr__(replay, "add", observed_add)

    for method, family in (("_ec_rollout", "ea"),
                           ("_rl_rollout", "rl")):
        if not hasattr(workflow, method):
            continue
        original = getattr(workflow, method)
        def observed_rollout(agent_state, replay_state, key,
                             _original=original, _family=family):
            records[_family + "_rollout_input"] = {
                "agent": _hash_tree(agent_state),
                "key": _hash_tree(key),
                "replay": occupied_replay_fingerprint(replay_state),
            }
            metrics, trajectory, result_replay = _original(
                agent_state, replay_state, key
            )
            records[_family + "_rollout_output"] = {
                "metrics": _hash_tree(metrics),
                "transition_batch": _hash_tree(trajectory),
                "replay": occupied_replay_fingerprint(result_replay),
            }
            return metrics, trajectory, result_replay
        setattr(workflow, method, observed_rollout)

    for method, label in (
        ("_rl_update", "rl_learner_update"),
        ("_ec_update", "ea_evolution_update"),
        ("_ec_update_with_rl_injection", "ea_evolution_update"),
    ):
        if not hasattr(workflow, method):
            continue
        original = getattr(workflow, method)
        def observed_update(*args, _original=original, _label=label, **kwargs):
            if _label == "rl_learner_update":
                records[_label + "_input"] = {
                    "agent": _hash_tree(args[0]),
                    "optimizer": _hash_tree(args[1]),
                    "replay": occupied_replay_fingerprint(args[2]),
                    "key": _hash_tree(args[3]),
                    "num_updates": int(_host_array(args[4])) if len(args) > 4 else None,
                }
            else:
                records[_label + "_input"] = {
                    "ec_optimizer": _hash_tree(args[0]),
                    "fitnesses": _hash_tree(args[-1]),
                    "learner_agent": _hash_tree(args[1]) if len(args) > 2 else None,
                }
            result = _original(*args, **kwargs)
            records[_label] = {"output": _hash_tree(result)}
            return result
        setattr(workflow, method, observed_update)

    def restore():
        object.__delattr__(replay, "add")
    return restore


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


def capture_pre_step(workflow: Any, key: Any) -> tuple[Any, float]:
    """Run the sole normal warmup prefix and synchronize before timing."""
    start = time.perf_counter()
    pre = workflow.init(key)
    for leaf in jtu.tree_leaves(pre):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()
    elapsed = time.perf_counter() - start
    if int(_host_array(pre.metrics.iterations)) != 10:
        raise RecoveryHold("wrong_pre_step_generation")
    return pre, elapsed


@contextmanager
def observe_pilot_stages():
    """Time eager diagnostic boundaries, restoring every callable afterwards.

    Durations are host-wall and overlapping/non-additive; first JIT compilation
    and asynchronous device work can contaminate them.
    """
    from evorl.algorithms.erl.erl_td3 import erl_utility_source_controls as workflow_mod
    from evorl.algorithms.erl.erl_td3 import pilot_dual_reference as dual_mod
    from evorl.algorithms.erl.erl_td3 import pilot_gradient_mechanisms as grad_mod
    from evorl.algorithms.erl.erl_td3 import pilot_artifacts as artifact_mod

    targets = (
        (workflow_mod, "sample_disjoint_replay_groups", "replay_sampling"),
        (workflow_mod, "choose_ea_candidates_and_donors", "ea_fresh_construction"),
        (workflow_mod, "sample_fresh_source", "rl_own_construction"),
        (workflow_mod, "sample_rl_heldout", "rl_fresh_construction"),
        (workflow_mod, "check_disjointness", "identity_validation"),
        (workflow_mod, "evaluate_paired_utility", "source_update_and_reference_eval"),
        (workflow_mod.ERLUtilitySourceControlsWorkflow,
         "_run_deferred_pilot_diagnostic", "pilot_diagnostic"),
        (grad_mod.ReferenceGradientCache, "get", "reference_gradient_cache"),
        (dual_mod, "measure_mechanisms", "mechanism_measurement"),
        (artifact_mod.PilotArtifactWriter, "append_checkpoint", "artifact_writing"),
    )
    observed = {"seconds": {}, "counts": {},
                "basis": "eager host-wall; nested, asynchronous and compilation-contaminated"}
    originals = []
    try:
        for target, name, label in targets:
            original = getattr(target, name)
            originals.append((target, name, original))
            def timed(*args, _original=original, _label=label, **kwargs):
                start = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    observed["seconds"][_label] = (
                        observed["seconds"].get(_label, 0.0)
                        + time.perf_counter() - start
                    )
                    observed["counts"][_label] = observed["counts"].get(_label, 0) + 1
            setattr(target, name, timed)
        yield observed
    finally:
        for target, name, original in reversed(originals):
            setattr(target, name, original)


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
    restore_off = restore_on = None
    try:
        # Workflow.init runs the sole real ten-iteration warmup prefix.
        pre, prefix_seconds = capture_pre_step(off, jax.random.PRNGKey(42))
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
        restore_off = instrument_baseline_stages(off)
        restore_on = instrument_baseline_stages(on)
        original_on_step = on.step
        def observed_on_step(state):
            with observe_pilot_stages() as observed:
                try:
                    return original_on_step(state)
                finally:
                    on._recovery_pilot_stages = observed
        on.step = observed_on_step
        try:
            audit = run_pair(off, on, pre)
        except RecoveryHold as error:
            partial = error.details.setdefault("audit", {})
            partial.setdefault("pre_state_fingerprints", captured_pre)
            partial.setdefault("available_stage_fingerprints", {
                "off": getattr(off, "_recovery_stage_fingerprints", {}),
                "on": getattr(on, "_recovery_stage_fingerprints", {}),
            })
            partial.setdefault("call_counts", {
                "off": _counts(off), "on": _counts(on),
            })
            partial.setdefault("replay_add_count_basis",
                               "inferred one add per inspected EC/RL host rollout boundary")
            partial["prefix_seconds"] = prefix_seconds
            partial["pilot_stage_timings"] = getattr(
                on, "_recovery_pilot_stages", None
            )
            partial["workflow_internal_state_audit"] = internal
            partial["pilot_artifact_directory"] = str(on._pilot_writer.root)
            try:
                partial["pilot_status"] = read_pilot_summary(
                    on._pilot_writer.root
                )["status"]
            except RecoveryHold as pilot_error:
                partial["pilot_status"] = "HOLD"
                partial["pilot_hold_reason"] = pilot_error.reason
            raise
        audit["prefix_seconds"] = prefix_seconds
        audit["pilot_stage_timings"] = on._recovery_pilot_stages
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
        if restore_off is not None:
            restore_off()
        if restore_on is not None:
            restore_on()
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
        result.update(error.details.get("audit", {}))
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
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
