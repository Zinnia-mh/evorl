"""R1 contracts using the real State/ReplayBuffer PyTrees and synthetic steps."""
from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from evorl.replay_buffers.replay_buffer import ReplayBuffer, ReplayBufferState
from evorl.types import PyTreeDict, State
from scripts.p1d_recovery_calibration import (
    RecoveryHold, compare_training_states, occupied_replay_fingerprint,
    run_pair, validate_pair_configs,
)


def make_state(*, size=2, capacity=5, tail=0):
    replay = ReplayBuffer(capacity=capacity, sample_batch_size=1)
    data = jnp.array([10, 20] + [tail] * (capacity - 2), dtype=jnp.int32)
    state = State(
        agent_state=PyTreeDict(params=PyTreeDict(
            actor_params={"w": jnp.array([1.0])},
            critic_params={"w": jnp.array([2.0])},
            target_actor_params={"w": jnp.array([3.0])},
            target_critic_params={"w": jnp.array([4.0])},
        )),
        opt_state=PyTreeDict(actor={"m": jnp.array([5.0])},
                             critic={"m": jnp.array([6.0])}),
        ec_opt_state=PyTreeDict(pop={"w": jnp.array([7.0])},
                                optimizer={"key": jax.random.PRNGKey(9)}),
        key=jax.random.PRNGKey(42),
        replay_buffer_state=ReplayBufferState(
            data=PyTreeDict(obs=data), current_index=jnp.array(size % capacity),
            buffer_size=jnp.array(size)),
        metrics=PyTreeDict(iterations=jnp.array(10),
                           sampled_episodes=jnp.array(320),
                           sampled_timesteps=jnp.array(3200),
                           rl_sampled_timesteps=jnp.array(0)),
    )
    return state


def changed(state, path, value):
    if path == "ea":
        return state.replace(ec_opt_state=state.ec_opt_state.replace(
            pop={"w": jnp.array([value])}))
    if path == "critic":
        p = state.agent_state.params.replace(critic_params={"w": jnp.array([value])})
        return state.replace(agent_state=state.agent_state.replace(params=p))
    if path == "optimizer":
        return state.replace(opt_state=state.opt_state.replace(
            critic={"m": jnp.array([value])}))
    if path == "key":
        return state.replace(key=jax.random.PRNGKey(int(value)))
    if path == "replay":
        rb = state.replay_buffer_state
        data = rb.data.replace(obs=rb.data.obs.at[0].set(int(value)))
        return state.replace(replay_buffer_state=rb.replace(data=data))
    raise AssertionError(path)


def test_replay_empty_ignores_all_payload_but_checks_pointer():
    a = make_state(size=0, tail=0)
    b = make_state(size=0, tail=999)
    b = b.replace(replay_buffer_state=b.replay_buffer_state.replace(
        data=PyTreeDict(obs=jnp.array([999] * 5))))
    assert occupied_replay_fingerprint(a.replay_buffer_state) == occupied_replay_fingerprint(b.replay_buffer_state)
    bad = b.replay_buffer_state.replace(current_index=jnp.array(1))
    with pytest.raises(RecoveryHold, match="replay_pointer"):
        occupied_replay_fingerprint(bad)


def test_replay_unfilled_tail_ignored_and_valid_slot_detected():
    a, b = make_state(tail=0), make_state(tail=999)
    assert occupied_replay_fingerprint(a.replay_buffer_state) == occupied_replay_fingerprint(b.replay_buffer_state)
    with pytest.raises(RecoveryHold, match="replay_buffer_state"):
        compare_training_states(a, changed(b, "replay", 99))


def test_replay_full_and_wrapped_compare_every_slot():
    a = make_state(size=5, capacity=5)
    b = a.replace(replay_buffer_state=a.replay_buffer_state.replace(
        data=PyTreeDict(obs=jnp.array([10, 20, 0, 0, 99]))))
    with pytest.raises(RecoveryHold, match="replay_buffer_state"):
        compare_training_states(a, b)
    wrapped = a.replay_buffer_state.replace(current_index=jnp.array(2))
    assert occupied_replay_fingerprint(wrapped) != occupied_replay_fingerprint(a.replay_buffer_state)


def test_masked_add_occupied_prefix_matches_real_buffer_contract():
    buffer = ReplayBuffer(capacity=5, sample_batch_size=1)
    rb = buffer.init(jnp.array(0, dtype=jnp.int32))
    rb = buffer.add(rb, jnp.array([3, 88, 4, 99]), jnp.array([True, False, True, False]))
    assert int(rb.buffer_size) == 2
    same = ReplayBufferState(
        data=jnp.array([3, 4, 123, 124, 125]), current_index=jnp.array(2),
        buffer_size=jnp.array(2))
    assert occupied_replay_fingerprint(rb) == occupied_replay_fingerprint(same)


@pytest.mark.parametrize("field,value", [
    ("ea", 17.0), ("critic", 18.0), ("optimizer", 19.0),
    ("key", 17), ("replay", 77),
])
def test_complete_pytree_mismatch_holds(field, value):
    with pytest.raises(RecoveryHold, match="common_pre_step_state_mismatch"):
        compare_training_states(make_state(), changed(make_state(), field, value))


def test_missing_state_field_fails_closed():
    bad = make_state().copy()
    del bad["opt_state"]
    with pytest.raises(RecoveryHold, match="incomplete_state"):
        compare_training_states(make_state(), bad)


class FakeWorkflow:
    def __init__(self, enabled=False, nondeterministic=False, mutate=False):
        self.enabled = enabled
        self.nondeterministic = nondeterministic
        self.mutate = mutate
        self.calls = 0
        self.collectors = 0
        self.replay_adds = 0

    def step(self, state):
        self.calls += 1
        self.collectors += 2
        self.replay_adds += 2
        if self.mutate:
            state.opt_state.critic = {"m": jnp.array([999.0])}
        delta = self.calls if self.nondeterministic else 1
        return None, state.replace(metrics=state.metrics.replace(
            iterations=state.metrics.iterations + delta))


def test_one_shared_pre_state_two_isolated_arms_and_one_step():
    pre = make_state()
    off, on = FakeWorkflow(), FakeWorkflow(enabled=True)
    audit = run_pair(off, on, pre, off_repeat=FakeWorkflow())
    assert audit["status"] == "PASS"
    assert audit["common_pre_step_state"]
    assert audit["baseline_post_step_equal"]
    assert (off.calls, on.calls) == (1, 1)
    assert audit["call_counts"]["off"]["collectors"] == 2
    assert audit["call_counts"]["on"]["replay_adds"] == 2
    assert int(pre.metrics.iterations) == 10


def test_old_independent_initialization_is_rejected_before_step():
    off, on = FakeWorkflow(), FakeWorkflow(True)
    with pytest.raises(RecoveryHold, match="common_pre_step_state_mismatch"):
        run_pair(off, on, make_state(), on_pre=changed(make_state(), "ea", 25.0))
    assert (off.calls, on.calls) == (0, 0)


def test_first_arm_mutating_shared_input_holds():
    with pytest.raises(RecoveryHold, match="pre_step_state_mutation"):
        run_pair(FakeWorkflow(mutate=True), FakeWorkflow(True), make_state())


def test_off_off_nondeterminism_holds():
    repeat = FakeWorkflow(nondeterministic=True)
    repeat.calls = 1
    with pytest.raises(RecoveryHold, match="baseline_replay_nondeterminism"):
        run_pair(FakeWorkflow(), FakeWorkflow(True), make_state(),
                 off_repeat=repeat)


def test_on_off_training_invariance_holds_on_divergence():
    class Divergent(FakeWorkflow):
        def step(self, state):
            _, out = super().step(state)
            return None, changed(out, "critic", 99.0)
    with pytest.raises(RecoveryHold, match="on_off_baseline_mismatch"):
        run_pair(FakeWorkflow(), Divergent(True), make_state())


def test_collector_and_replay_add_parity_holds():
    class ExtraAdd(FakeWorkflow):
        def step(self, state):
            result = super().step(state)
            self.replay_adds += 1
            return result
    with pytest.raises(RecoveryHold, match="replay_add_call_mismatch"):
        run_pair(FakeWorkflow(), ExtraAdd(True), make_state())


def test_config_only_pilot_fields_may_differ():
    base = {"seed": 42, "batch_size": 256, "diagnostic_enabled": False,
            "reference_target_bias_enabled": False, "pilot_artifact_root": None}
    on = dict(base, diagnostic_enabled=True, reference_target_bias_enabled=True,
              pilot_artifact_root="/tmp/new-run")
    validate_pair_configs(base, on)
    on["batch_size"] = 128
    with pytest.raises(RecoveryHold, match="config_mismatch"):
        validate_pair_configs(base, on)

def test_equal_but_separately_created_on_pre_is_not_claimed_common():
    with pytest.raises(RecoveryHold, match="common_pre_step_object_mismatch"):
        run_pair(FakeWorkflow(), FakeWorkflow(True), make_state(), on_pre=make_state())


def test_pilot_summary_hold_is_not_reported_as_recovery_pass(tmp_path):
    from scripts.p1d_recovery_calibration import read_pilot_summary
    run = tmp_path / "pilot" / "new-id"
    run.mkdir(parents=True)
    (run / "paired_summary.jsonl").write_text(
        '{"generation": 11, "status": "HOLD", "reasons": ["insufficient_fresh"]}\n'
    )
    with pytest.raises(RecoveryHold, match="pilot_diagnostic_hold"):
        read_pilot_summary(run)

def test_ring_overwrite_changes_occupied_fingerprint():
    buffer = ReplayBuffer(capacity=3, sample_batch_size=1)
    rb = buffer.init(jnp.array(0, dtype=jnp.int32))
    rb = buffer.add(rb, jnp.array([1, 2, 3]))
    before = occupied_replay_fingerprint(rb)
    rb = buffer.add(rb, jnp.array([4]))
    assert int(rb.current_index) == 1
    assert int(rb.buffer_size) == 3
    assert occupied_replay_fingerprint(rb) != before


def test_workflow_contexts_must_not_share_mutable_components():
    from scripts.p1d_recovery_calibration import audit_workflow_contexts
    common = object()
    off = SimpleNamespace(env=common, agent=object())
    on = SimpleNamespace(env=common, agent=object())
    with pytest.raises(RecoveryHold, match="workflow_context_shared"):
        audit_workflow_contexts(off, on)

def test_unrecognized_top_level_state_field_fails_closed():
    extra = make_state().replace(hidden_counter=jnp.array(1))
    with pytest.raises(RecoveryHold, match="unexpected_state_fields"):
        compare_training_states(make_state(), extra)

def test_off_post_state_released_before_on_step():
    import weakref
    holder = {}
    class Off(FakeWorkflow):
        def step(self, state):
            result = super().step(state)
            holder["ref"] = weakref.ref(result[1])
            return result
    class On(FakeWorkflow):
        def step(self, state):
            assert holder["ref"]() is None
            return super().step(state)
    assert run_pair(Off(), On(True), make_state())["status"] == "PASS"

def test_ec_and_rl_collector_counts_checked_separately():
    class DifferentMix(FakeWorkflow):
        def __init__(self):
            super().__init__(enabled=True)
            self._recovery_counts = {
                "collectors": 2, "ec_collectors": 0,
                "rl_collectors": 2, "replay_adds": 2,
            }
        def step(self, state):
            self.calls += 1
            return None, state.replace(metrics=state.metrics.replace(
                iterations=state.metrics.iterations + 1))
    class Ordinary(FakeWorkflow):
        def __init__(self):
            super().__init__()
            self._recovery_counts = {
                "collectors": 2, "ec_collectors": 1,
                "rl_collectors": 1, "replay_adds": 2,
            }
    with pytest.raises(RecoveryHold, match="collector_call_mismatch"):
        run_pair(Ordinary(), DifferentMix(), make_state())

def test_environment_step_count_parity_is_explicit():
    class ExtraEnvironmentStep(FakeWorkflow):
        def step(self, state):
            _, post = super().step(state)
            return None, post.replace(metrics=post.metrics.replace(
                sampled_timesteps=post.metrics.sampled_timesteps + 1))
    with pytest.raises(RecoveryHold, match="environment_step_mismatch"):
        run_pair(FakeWorkflow(), ExtraEnvironmentStep(True), make_state())

def test_capture_prefix_runs_init_once_and_requires_generation_11_input():
    from scripts.p1d_recovery_calibration import capture_pre_step
    class Init:
        calls = 0
        def init(self, key):
            self.calls += 1
            return make_state()
    workflow = Init()
    state, seconds = capture_pre_step(workflow, jax.random.PRNGKey(42))
    assert workflow.calls == 1
    assert int(state.metrics.iterations) == 10
    assert seconds >= 0


def test_stage_timer_restores_observed_function(monkeypatch):
    from evorl.algorithms.erl.erl_td3 import erl_utility_source_controls as module
    from scripts.p1d_recovery_calibration import observe_pilot_stages
    original = lambda: 73
    monkeypatch.setattr(module, "sample_fresh_source", original)
    with observe_pilot_stages() as observed:
        assert module.sample_fresh_source() == 73
    assert module.sample_fresh_source is original
    assert observed["counts"]["rl_own_construction"] == 1
    assert observed["seconds"]["rl_own_construction"] >= 0


def test_hold_retains_all_post_mismatches_and_pre_evidence():
    class Divergent(FakeWorkflow):
        def step(self, state):
            _, out = super().step(state)
            out = changed(out, "critic", 99.0)
            out = changed(out, "optimizer", 88.0)
            return None, changed(out, "replay", 77)
    with pytest.raises(RecoveryHold) as caught:
        run_pair(FakeWorkflow(), Divergent(True), make_state())
    audit = caught.value.details["audit"]
    assert audit["pre_state_fingerprints"]
    assert audit["all_mismatched_fields"] == [
        "agent_state", "opt_state", "replay_buffer_state"
    ]
    assert audit["first_mismatched_field"] == "agent_state"
    assert audit["call_counts"]["off"]["replay_adds"] == 2
    assert audit["call_counts"]["on"]["replay_adds"] == 2
    paths = [x["path"] for x in audit["leaf_differences"]]
    assert any("critic_params" in x for x in paths)
    assert any("replay_buffer_state.data" in x for x in paths)


def test_occupied_replay_leaf_evidence_ignores_uninitialized_tail():
    from scripts.p1d_recovery_calibration import state_leaf_differences
    a, b = make_state(tail=0), make_state(tail=999)
    assert state_leaf_differences(a, b) == []
    differences = state_leaf_differences(a, changed(b, "replay", 77))
    assert len(differences) == 1
    assert differences[0]["different_elements"] == 1
    assert differences[0]["shape"] == [2]


def test_first_divergence_uses_real_step_order():
    from scripts.p1d_recovery_calibration import first_divergent_stage
    off = {"ea_rollout": {"replay": "a"}, "rl_rollout": {"replay": "x"},
           "rl_learner_update": {"agent": "one"}}
    on = {"ea_rollout": {"replay": "a"}, "rl_rollout": {"replay": "y"},
          "rl_learner_update": {"agent": "two"}}
    assert first_divergent_stage(off, on) == "rl_rollout"
    on["rl_rollout"] = off["rl_rollout"]
    assert first_divergent_stage(off, on) == "rl_learner_update"


def test_hold_file_preserves_partial_audit(tmp_path, monkeypatch):
    import scripts.p1d_recovery_calibration as harness
    monkeypatch.setattr(harness, "OUTPUT_PARENT", tmp_path)
    def fail(*args):
        raise RecoveryHold("on_off_baseline_mismatch", "agent_state",
                           {"audit": {"pre_state_fingerprints": {"key": "abc"},
                                      "all_mismatched_fields": ["agent_state"]}})
    monkeypatch.setattr(harness, "_real_entry", fail)
    run = tmp_path / "new"
    assert harness.main(["--run-dir", str(run)]) == 2
    row = __import__("json").loads((run / "state_audit.json").read_text())
    assert row["pre_state_fingerprints"]["key"] == "abc"
    assert row["all_mismatched_fields"] == ["agent_state"]


def test_class_level_jit_wrapping_is_visible_to_both_arms(monkeypatch):
    from evorl.algorithms.erl.erl_td3.erl_learner_utility import ERLLearnerUtilityWorkflow as Parent
    from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import ERLUtilitySourceControlsWorkflow as Child
    names = ("_rl_rollout", "_ec_update", "_ec_update_with_rl_injection",
             "evaluate", "_postsetup_replaybuffer")
    originals = {name: Parent.__dict__.get(name) for name in names}
    dynamic = type("R4DynamicWorkflow", (Child,), {})
    try:
        dynamic.enable_jit()
        assert Parent.__dict__["_ec_update"] is not originals["_ec_update"]
        assert dynamic._ec_update is Parent._ec_update
        # A later enable_jit=False build does not undo class-level wrapping.
        assert dynamic._ec_update is Parent._ec_update
    finally:
        for name, old in originals.items():
            if old is None:
                delattr(Parent, name)
            else:
                setattr(Parent, name, old)


def test_host_stage_probes_observe_without_extra_rollout_or_add():
    from scripts.p1d_recovery_calibration import instrument_baseline_stages
    class Replay:
        calls = 0
        def add(self, rb, xs, mask=None):
            self.calls += 1
            return rb
    class Workflow:
        def __init__(self):
            self.replay_buffer = Replay()
            self.rollouts = 0
        def _ec_rollout(self, agent, replay, key):
            self.rollouts += 1
            after = self.replay_buffer.add(replay, jnp.array([1, 2]))
            return jnp.array(1), jnp.array([1, 2]), after
    workflow = Workflow()
    instrument_baseline_stages(workflow)
    state = make_state()
    workflow._ec_rollout(state.agent_state, state.replay_buffer_state, state.key)
    assert workflow.rollouts == 1
    assert workflow.replay_buffer.calls == 1
    evidence = workflow._recovery_stage_fingerprints
    assert "ea_rollout_input" in evidence
    assert "ea_replay_add" in evidence
    assert "ea_rollout_output" in evidence
    assert evidence["ea_replay_add"]["valid_transition_count"] == 2


@pytest.mark.parametrize("stage", [
    "ea_rollout_input", "ea_replay_add", "rl_learner_update",
])
def test_stage_comparator_localizes_synthetic_first_difference(stage):
    from scripts.p1d_recovery_calibration import first_divergent_stage
    order = ("ea_rollout_input", "ea_replay_add", "ea_rollout_output",
             "rl_rollout_input", "rl_replay_add", "rl_rollout_output",
             "rl_learner_update", "ea_evolution_update")
    off = {name: {"fingerprint": "same"} for name in order}
    on = {name: dict(value) for name, value in off.items()}
    on[stage]["fingerprint"] = "changed"
    on["ea_evolution_update"]["fingerprint"] = "also_changed"
    assert first_divergent_stage(off, on) == stage


def test_disabled_diagnostics_do_not_change_synthetic_baseline():
    off, on = FakeWorkflow(enabled=False), FakeWorkflow(enabled=False)
    pre = make_state()
    audit = run_pair(off, on, pre)
    assert audit["baseline_post_step_equal"]
    assert (off.calls, on.calls) == (1, 1)
    assert int(pre.metrics.iterations) == 10


def test_leaf_numeric_delta_reports_same_dtype_float_difference():
    from scripts.p1d_recovery_calibration import state_leaf_differences
    pre = make_state()
    altered = changed(pre, "critic", jnp.float32(9))
    row = next(x for x in state_leaf_differences(pre, altered)
               if "critic_params" in x["path"])
    assert row["different_elements"] == 1
    assert row["all_finite"]
    assert row["max_absolute_difference"] == 7.0


def test_stage_probe_supports_frozen_real_replay_buffer_and_restores():
    from scripts.p1d_recovery_calibration import instrument_baseline_stages
    real = ReplayBuffer(capacity=5, sample_batch_size=1)
    workflow = SimpleNamespace(replay_buffer=real)
    restore = instrument_baseline_stages(workflow)
    assert "add" in real.__dict__
    before = real.init(jnp.array(0, dtype=jnp.int32))
    after = real.add(before, jnp.array([4, 5], dtype=jnp.int32))
    assert int(after.buffer_size) == 2
    assert workflow._recovery_stage_fingerprints["ea_replay_add"]["valid_transition_count"] == 2
    restore()
    assert "add" not in real.__dict__


def test_call_count_hold_preserves_already_computed_post_hashes():
    class ExtraAdd(FakeWorkflow):
        def step(self, state):
            result = super().step(state)
            self.replay_adds += 1
            return result
    with pytest.raises(RecoveryHold, match="replay_add_call_mismatch") as caught:
        run_pair(FakeWorkflow(), ExtraAdd(True), make_state())
    audit = caught.value.details["audit"]
    assert audit["pre_state_fingerprints"]
    assert audit["off_post_fingerprints"]
    assert audit["on_post_fingerprints"]


def test_pre_step_component_record_covers_training_state():
    from scripts.p1d_recovery_calibration import pre_step_component_record
    record = pre_step_component_record(make_state())
    for name in ("actor", "critic", "target_actor", "target_critic",
                 "actor_optimizer", "critic_optimizer", "ea_population",
                 "ec_optimizer", "baseline_key", "replay_valid_slots",
                 "replay_pointer", "replay_size", "training_counters"):
        assert name in record
    assert record["replay_pointer"] == 2
    assert record["replay_size"] == 2


def test_leaf_numeric_delta_reports_mean_absolute_difference():
    from scripts.p1d_recovery_calibration import state_leaf_differences
    pre = make_state()
    changed_state = changed(pre, "critic", jnp.float32(9))
    row = next(x for x in state_leaf_differences(pre, changed_state)
               if "critic_params" in x["path"])
    assert row["mean_absolute_difference"] == 7.0


def test_stage_probe_records_leaf_shape_and_dtype():
    from scripts.p1d_recovery_calibration import instrument_baseline_stages
    real = ReplayBuffer(capacity=5, sample_batch_size=1)
    workflow = SimpleNamespace(replay_buffer=real)
    restore = instrument_baseline_stages(workflow)
    try:
        before = real.init(jnp.array(0, dtype=jnp.int32))
        real.add(before, jnp.array([4, 5], dtype=jnp.int32))
        leaves = workflow._recovery_stage_fingerprints["ea_replay_add"]["add_input_leaves"]
        assert leaves == [{"shape": [2], "dtype": "int32"}]
    finally:
        restore()


def test_stage_leaf_comparison_reports_numeric_magnitude():
    from scripts.p1d_recovery_calibration import stage_numeric_differences
    off = {"ea_rollout_output": {"transition_batch":
           jnp.array([1.0, 2.0], dtype=jnp.float32)}}
    on = {"ea_rollout_output": {"transition_batch":
          jnp.array([1.0, 4.0], dtype=jnp.float32)}}
    rows = stage_numeric_differences(off, on, "ea_rollout_output")
    assert len(rows) == 1
    row = rows[0]
    assert row["path"].startswith("ea_rollout_output.transition_batch")
    assert row["different_elements"] == 1
    assert row["max_absolute_difference"] == 2.0
    assert row["mean_absolute_difference"] == 1.0


def test_stage_replay_comparison_excludes_unoccupied_tail():
    from scripts.p1d_recovery_calibration import stage_numeric_differences
    a = make_state(tail=0).replay_buffer_state
    b = make_state(tail=99).replay_buffer_state
    assert stage_numeric_differences({"ea_replay_add": {"replay_after": a}},
                                     {"ea_replay_add": {"replay_after": b}},
                                     "ea_replay_add") == []
