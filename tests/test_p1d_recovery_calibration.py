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
