from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import optax
import pytest

from evorl.agent import AgentState
from evorl.algorithms.erl.erl_td3 import (
    erl_utility_source_controls as workflow_module,
)
from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import (
    ERLUtilitySourceControlsWorkflow,
    SOURCE_CONTROL_TAG,
)
from evorl.algorithms.erl.erl_td3.erl_learner_utility import (
    ERLLearnerUtilityWorkflow,
)
from evorl.algorithms.erl.erl_td3.learner_utility import derive_diagnostic_key
from evorl.algorithms.erl.erl_td3.utility_source_controls import (
    SourceControlBatch,
    tree_fingerprint,
)
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict


class _MarkerReplay:
    def __init__(self, *, sample_batch_size=2, ready=True):
        self.sample_batch_size = sample_batch_size
        self.ready = ready
        self.sample_keys = []

    def can_sample(self, state):
        del state
        return self.ready

    def sample(self, state, key):
        self.sample_keys.append(jax.random.key_data(key))
        return state.data

    def add_marker(self, state, marker):
        return SimpleNamespace(
            data=SampleBatch(obs=jnp.full((self.sample_batch_size, 1), marker)),
            current_size=state.current_size + self.sample_batch_size,
            current_position=state.current_position + self.sample_batch_size,
        )


class _EACollector:
    def __init__(self):
        self.calls = 0

    def rollout(self, individual, key, num_episodes):
        del key, num_episodes
        self.calls += 1
        marker = individual
        metrics = PyTreeDict(
            episode_returns=jnp.asarray([marker * 10.0]),
            episode_lengths=jnp.asarray([2]),
        )
        trajectory = SampleBatch(
            obs=jnp.full((2, 1, 1), marker),
            actions=jnp.zeros((2, 1, 1)),
            rewards=jnp.full((2, 1), marker),
            dones=jnp.zeros((2, 1), dtype=jnp.bool_),
        )
        return metrics, trajectory


class _EAReplay:
    def __init__(self):
        self.add_calls = 0
        self.added = None

    def add(self, state, trajectory, mask):
        self.add_calls += 1
        self.added = (trajectory, mask)
        return state + 1


class _RLCollector:
    def __init__(self):
        self.calls = 0

    def rollout(self, agent_state, key, num_episodes):
        del key, num_episodes
        self.calls += 1
        marker = agent_state.params.actor_params["w"][0]
        metrics = PyTreeDict(
            episode_returns=jnp.asarray([marker]),
            episode_lengths=jnp.asarray([2]),
        )
        trajectory = SampleBatch(
            obs=jnp.full((2, 1, 1), marker),
            actions=jnp.zeros((2, 1, 1)),
            rewards=jnp.full((2, 1), marker),
            dones=jnp.zeros((2, 1), dtype=jnp.bool_),
        )
        return metrics, trajectory


def _state():
    params = PyTreeDict(
        critic_params={"w": jnp.array([[1.0, 2.0]])},
        target_critic_params={"w": jnp.array([[3.0, 4.0]])},
        actor_params={"w": jnp.array([[5.0]])},
        target_actor_params={"w": jnp.array([[6.0]])},
    )
    return SimpleNamespace(
        agent_state=AgentState(params=params, obs_preprocessor_state=None),
        opt_state=SimpleNamespace(
            critic={"count": jnp.array(7), "mu": {"w": jnp.array([[0.1, 0.2]])}}
        ),
        key=jax.random.key(11),
        metrics=SimpleNamespace(
            iterations=jnp.array(4),
            sampled_episodes=jnp.array(12),
            sampled_timesteps=jnp.array(96),
            rl_sampled_timesteps=jnp.array(32),
        ),
        replay_buffer_state=SimpleNamespace(
            current_size=jnp.array(100), current_position=jnp.array(20)
        ),
    )


def _control_state(marker=1.0):
    state = _state()
    state.replay_buffer_state = SimpleNamespace(
        data=SampleBatch(obs=jnp.full((2, 1), marker)),
        current_size=jnp.array(20),
        current_position=jnp.array(4),
    )
    return state


def _control_workflow(*, references=4, replay_controls=8, ready=True):
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    workflow.replay_buffer = _MarkerReplay(ready=ready)
    workflow.config = SimpleNamespace(
        source_batch_size=2,
        num_reference_batches=references,
        num_random_control_batches=replay_controls,
    )
    return workflow


def _ea_workflow(workflow_cls=ERLUtilitySourceControlsWorkflow, *, active=True):
    workflow = object.__new__(workflow_cls)
    workflow.ec_collector = _EACollector()
    workflow.replay_buffer = _EAReplay()
    workflow.agent_state_vmap_axes = 0
    workflow.config = SimpleNamespace(
        pop_size=2,
        episodes_for_fitness=1,
        source_batch_size=2,
        diagnostic_enabled=True,
        diagnostic_interval=2,
    )
    workflow._diagnostic_enabled = True
    workflow._diagnostic_active = active
    workflow._diagnostic_context = SimpleNamespace(
        generation=5,
        key=jax.random.key(19),
        snapshot=SimpleNamespace(
            reference_fingerprints=("ref-a", "ref-b"),
            replay_control_fingerprints=("random-a", "random-b"),
        ),
    )
    workflow.immediate_calls = 0

    def _old_immediate(raw, replay_state):
        del raw, replay_state
        workflow.immediate_calls += 1
        return ("old-immediate-record",)

    workflow._run_pre_exposure_diagnostic = _old_immediate
    return workflow


def _rl_agent_state(critic_value=1.0):
    params = PyTreeDict(
        critic_params={"w": jnp.full((2, 1), critic_value)},
        target_critic_params={"w": jnp.full((2, 1), 3.0)},
        actor_params={"w": jnp.array([[4.0], [5.0]])},
        target_actor_params={"w": jnp.array([[6.0], [7.0]])},
    )
    return AgentState(params=params, obs_preprocessor_state=None)


def _rl_workflow(*, active=True, frozen_critic=1.0):
    workflow = _ea_workflow(active=active)
    workflow.rl_collector = _RLCollector()
    workflow.config.num_rl_agents = 2
    workflow.config.rollout_episodes = 1
    workflow._diagnostic_context.snapshot.critic_fingerprint = tree_fingerprint(
        {"w": jnp.array([frozen_critic])}
    )
    workflow._diagnostic_context.ea_sources = ("ea-source",)
    workflow.deferred_calls = []

    def _deferred(context, rl_source, replay_after_rl):
        workflow.deferred_calls.append((context, rl_source, replay_after_rl))
        return None

    workflow._run_deferred_source_control_diagnostic = _deferred
    return workflow


def test_snapshot_selects_learner_zero_critic_target_and_optimizer():
    workflow = _control_workflow()
    context = workflow._make_pre_exposure_context(_control_state())
    snapshot = context.snapshot
    chex.assert_trees_all_equal(
        snapshot.learner_agent_state.params.critic_params["w"],
        jnp.array([1.0, 2.0]),
    )
    chex.assert_trees_all_equal(
        snapshot.learner_agent_state.params.target_critic_params["w"],
        jnp.array([3.0, 4.0]),
    )
    chex.assert_trees_all_equal(
        snapshot.critic_opt_state["mu"]["w"], jnp.array([0.1, 0.2])
    )


def test_snapshot_keeps_immutable_state_after_baseline_replacement():
    workflow = _control_workflow()
    state = _control_state()
    context = workflow._make_pre_exposure_context(state)
    state.agent_state = state.agent_state.replace(
        params=state.agent_state.params.replace(
            critic_params={"w": jnp.array([[99.0, 99.0]])}
        )
    )
    chex.assert_trees_all_equal(
        context.snapshot.learner_agent_state.params.critic_params["w"],
        jnp.array([1.0, 2.0]),
    )


def test_diagnostic_keys_do_not_advance_baseline_key():
    state = _control_state()
    before = state.key
    workflow = _control_workflow()
    context = workflow._make_pre_exposure_context(state)
    derived = derive_diagnostic_key(
        context.snapshot.baseline_key, context.snapshot.generation, SOURCE_CONTROL_TAG
    )
    chex.assert_trees_all_equal(state.key, before)
    assert not bool(jnp.all(derived == before))


def test_snapshot_records_generation_counters_and_fingerprints():
    workflow = _control_workflow()
    snapshot = workflow._make_pre_exposure_context(_control_state()).snapshot
    assert snapshot.generation == 5
    assert len(snapshot.baseline_key_fingerprint) == 64
    assert len(snapshot.critic_fingerprint) == 64
    assert len(snapshot.target_critic_fingerprint) == 64
    assert len(snapshot.optimizer_fingerprint) == 64
    assert len(snapshot.replay_metadata_fingerprint) == 64
    assert len(snapshot.counter_fingerprint) == 64


def test_controls_sample_before_ea_collector_and_replay_add():
    workflow = _control_workflow()
    state = _control_state(marker=11.0)
    context = workflow._make_pre_exposure_context(state)
    state.replay_buffer_state = workflow.replay_buffer.add_marker(
        state.replay_buffer_state, marker=99.0
    )
    frozen = (
        context.snapshot.reference_batches
        + context.snapshot.replay_control_batches
    )
    assert len(frozen) == 12
    assert all(bool(jnp.all(batch.obs == 11.0)) for batch in frozen)
    assert bool(jnp.all(state.replay_buffer_state.data.obs == 99.0))


def test_controls_exclude_current_ea_marker():
    workflow = _control_workflow()
    context = workflow._make_pre_exposure_context(_control_state(marker=7.0))
    frozen = (
        context.snapshot.reference_batches
        + context.snapshot.replay_control_batches
    )
    assert frozen
    assert all(not bool(jnp.any(batch.obs == 101.0)) for batch in frozen)
    assert all(bool(jnp.all(batch.obs == 7.0)) for batch in frozen)


def test_reference_and_replay_counts_are_configurable():
    workflow = _control_workflow(references=3, replay_controls=5)
    context = workflow._make_pre_exposure_context(_control_state())
    assert len(context.snapshot.reference_batches) == 3
    assert len(context.snapshot.reference_fingerprints) == 3
    assert len(context.snapshot.replay_control_batches) == 5
    assert len(context.snapshot.replay_control_fingerprints) == 5
    keys = [tuple(map(int, key)) for key in workflow.replay_buffer.sample_keys]
    assert len(keys) == len(set(keys)) == 8
    reference_keys = keys[:3]

    workflow_more_controls = _control_workflow(references=3, replay_controls=8)
    workflow_more_controls._make_pre_exposure_context(_control_state())
    more_keys = [
        tuple(map(int, key)) for key in workflow_more_controls.replay_buffer.sample_keys
    ]
    assert more_keys[:3] == reference_keys


def test_reference_and_replay_counts_default_to_four_and_eight():
    workflow = _control_workflow()
    del workflow.config.num_reference_batches
    del workflow.config.num_random_control_batches
    context = workflow._make_pre_exposure_context(_control_state())
    assert len(context.snapshot.reference_batches) == 4
    assert len(context.snapshot.replay_control_batches) == 8


def test_replay_not_ready_emits_ineligible_checkpoint_without_sampling():
    workflow = _control_workflow(ready=False)
    context = workflow._make_pre_exposure_context(_control_state())
    assert context.controls_eligible is False
    assert context.controls_reason == "replay_not_ready"
    assert context.snapshot.reference_batches == ()
    assert context.snapshot.replay_control_batches == ()
    assert workflow.replay_buffer.sample_keys == []


def test_sampling_leaves_replay_state_treewise_equal():
    workflow = _control_workflow()
    state = _control_state()
    before = state.replay_buffer_state
    workflow._make_pre_exposure_context(state)
    assert len(workflow.replay_buffer.sample_keys) == 12
    chex.assert_trees_all_equal(before, state.replay_buffer_state)


def test_replay_sample_size_must_match_source_batch_size():
    workflow = _control_workflow()
    workflow.replay_buffer.sample_batch_size = 3
    with pytest.raises(ValueError, match="sample_batch_size"):
        workflow._make_pre_exposure_context(_control_state())


def test_ea_hook_retains_pre_mixing_provenance():
    workflow = _ea_workflow()
    workflow._ec_rollout(jnp.array([1.0, 2.0]), 7, jax.random.key(3))
    sources = workflow._diagnostic_context.ea_sources
    assert [source.source_id for source in sources] == [
        "gen_5_ind_0",
        "gen_5_ind_1",
    ]
    assert [source.source_type for source in sources] == ["EA", "EA"]
    assert [source.evolutionary_fitness for source in sources] == [10.0, 20.0]
    assert all(source.valid_transition_count == 2 for source in sources)
    assert bool(jnp.all(sources[0].batch.obs == 1.0))
    assert bool(jnp.all(sources[1].batch.obs == 2.0))


def test_ea_hook_adds_replay_once():
    workflow = _ea_workflow()
    _, _, replay_after = workflow._ec_rollout(
        jnp.array([1.0, 2.0]), 7, jax.random.key(3)
    )
    assert len(workflow._diagnostic_context.ea_sources) == 2
    assert workflow.replay_buffer.add_calls == 1
    assert replay_after == 8


def test_ea_hook_does_not_emit_records_before_rl_rollout():
    workflow = _ea_workflow()
    workflow._evaluate_source_utility = lambda *args: pytest.fail(
        "EA staging must not evaluate utility"
    )
    workflow._ec_rollout(jnp.array([1.0, 2.0]), 7, jax.random.key(3))
    assert not hasattr(workflow, "_latest_diagnostic_records")
    assert not hasattr(workflow._diagnostic_context, "utility_records")


def test_disabled_or_non_due_path_delegates_exactly_to_parent():
    child = _ea_workflow(active=False)
    child._diagnostic_enabled = False
    parent = _ea_workflow(ERLLearnerUtilityWorkflow, active=False)
    child_result = child._ec_rollout(
        jnp.array([1.0, 2.0]), 7, jax.random.key(3)
    )
    parent_result = parent._ec_rollout(
        jnp.array([1.0, 2.0]), 7, jax.random.key(3)
    )
    chex.assert_trees_all_equal(child_result, parent_result)
    assert child.ec_collector.calls == parent.ec_collector.calls == 1
    assert child.replay_buffer.add_calls == parent.replay_buffer.add_calls == 1
    assert not hasattr(child._diagnostic_context, "ea_sources")

    non_due = _ea_workflow(active=False)
    non_due_state = SimpleNamespace(metrics=SimpleNamespace(iterations=jnp.array(0)))
    assert non_due._diagnostic_due(non_due_state) is False
    non_due_result = non_due._ec_rollout(
        jnp.array([1.0, 2.0]), 7, jax.random.key(3)
    )
    chex.assert_trees_all_equal(non_due_result, parent_result)
    assert non_due.ec_collector.calls == 1
    assert non_due.replay_buffer.add_calls == 1
    assert not hasattr(non_due._diagnostic_context, "ea_sources")


def test_ea_hook_calls_collector_exactly_once():
    workflow = _ea_workflow()
    workflow._ec_rollout(jnp.array([1.0, 2.0]), 7, jax.random.key(3))
    assert len(workflow._diagnostic_context.ea_sources) == 2
    assert workflow.ec_collector.calls == 1


def test_ea_source_batch_uses_raw_pre_flatten_trajectory():
    workflow = _ea_workflow()
    before_reference = workflow._diagnostic_context.snapshot.reference_fingerprints
    before_random = workflow._diagnostic_context.snapshot.replay_control_fingerprints
    _, flattened, _ = workflow._ec_rollout(
        jnp.array([1.0, 2.0]), 7, jax.random.key(3)
    )
    sources = workflow._diagnostic_context.ea_sources
    assert sources[0].batch.obs.shape == (2, 1)
    assert flattened.obs.shape == (4, 1)
    assert (
        workflow._diagnostic_context.snapshot.reference_fingerprints
        == before_reference
    )
    assert (
        workflow._diagnostic_context.snapshot.replay_control_fingerprints
        == before_random
    )


def test_rl_own_uses_baseline_transition_identity():
    workflow = _rl_workflow()
    workflow._rl_rollout(_rl_agent_state(), 7, jax.random.key(8))
    source = workflow._diagnostic_context.rl_source
    assert source.source_id == "gen_5_rl_own"
    assert source.valid_transition_count == 4
    assert set(map(float, source.batch.obs[:, 0])) <= {4.0, 5.0}
    assert set(map(float, source.batch.obs[:, 0])) <= set(
        map(float, workflow.replay_buffer.added[0].obs[:, 0])
    )


def test_diagnostic_adds_no_collector_call_or_environment_steps():
    active = _rl_workflow(active=True)
    baseline = _rl_workflow(active=False)
    active_result = active._rl_rollout(_rl_agent_state(), 7, jax.random.key(8))
    baseline_result = baseline._rl_rollout(_rl_agent_state(), 7, jax.random.key(8))
    chex.assert_trees_all_equal(active_result, baseline_result)
    assert active.rl_collector.calls == baseline.rl_collector.calls == 1
    assert (
        int(active_result[0].episode_lengths.sum())
        == int(baseline_result[0].episode_lengths.sum())
        == 4
    )


def test_rl_replay_insertion_occurs_once():
    workflow = _rl_workflow()
    _, _, replay_after = workflow._rl_rollout(
        _rl_agent_state(), 7, jax.random.key(8)
    )
    assert workflow.replay_buffer.add_calls == 1
    assert replay_after == 8


def test_deferred_diagnostic_precedes_real_learner_update():
    workflow = _rl_workflow()
    events = []
    workflow._run_deferred_source_control_diagnostic = (
        lambda context, rl_source, replay: events.append("deferred")
    )
    workflow._rl_rollout(_rl_agent_state(), 7, jax.random.key(8))
    events.append("real_learner_update")
    assert events == ["deferred", "real_learner_update"]


def test_temporal_alignment_exact_when_critic_unchanged():
    workflow = _rl_workflow(frozen_critic=1.0)
    workflow._rl_rollout(_rl_agent_state(1.0), 7, jax.random.key(8))
    assert workflow._diagnostic_context.source_temporal_alignment == "exact"


def test_temporal_alignment_delayed_when_intervening_critic_differs():
    workflow = _rl_workflow(frozen_critic=1.0)
    workflow._rl_rollout(_rl_agent_state(2.0), 7, jax.random.key(8))
    assert workflow._diagnostic_context.source_temporal_alignment == "delayed"


def test_rl_rollout_preserves_frozen_controls_and_baseline_key():
    workflow = _rl_workflow()
    snapshot = workflow._diagnostic_context.snapshot
    before = (
        snapshot.reference_fingerprints,
        snapshot.replay_control_fingerprints,
        workflow._diagnostic_context.key,
        workflow._diagnostic_context.ea_sources,
    )
    workflow._rl_rollout(_rl_agent_state(), 7, jax.random.key(8))
    chex.assert_trees_all_equal(
        before,
        (
            snapshot.reference_fingerprints,
            snapshot.replay_control_fingerprints,
            workflow._diagnostic_context.key,
            workflow._diagnostic_context.ea_sources,
        ),
    )


class _DiagnosticAgent:
    def critic_loss(self, agent_state, batch, key):
        del key
        value = agent_state.params.critic_params["w"]
        return SimpleNamespace(critic_loss=jnp.square(value - batch.obs.mean()))


class _ReplayWithUnreadableCapacity:
    current_size = jnp.array(10)
    current_position = jnp.array(2)

    def __array__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("diagnostic must not read full replay capacity")


def _deferred_workflow_and_context():
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    workflow.agent = _DiagnosticAgent()
    workflow.optimizer = optax.sgd(0.1)
    workflow.config = SimpleNamespace(
        source_batch_size=2,
        source_control_step_scales=(1.0, 0.5, 0.1),
        source_control_numerical_tolerance=1e-6,
    )
    params = PyTreeDict(
        critic_params={"w": jnp.array(1.0)},
        target_critic_params={"w": jnp.array(3.0)},
        actor_params={"w": jnp.array(4.0)},
        target_actor_params={"w": jnp.array(5.0)},
    )
    learner = AgentState(params=params, obs_preprocessor_state=None)
    opt_state = workflow.optimizer.init(params.critic_params)
    batch = lambda value: SampleBatch(obs=jnp.full((2, 1), value))
    snapshot = SimpleNamespace(
        generation=5,
        learner_agent_state=learner,
        critic_opt_state=opt_state,
        baseline_key=jax.random.key(31),
        critic_fingerprint=tree_fingerprint(params.critic_params),
        target_critic_fingerprint=tree_fingerprint(params.target_critic_params),
        optimizer_fingerprint=tree_fingerprint(opt_state),
        reference_batches=tuple(batch(v) for v in (0.5, 1.5, 2.5, 3.5)),
        reference_fingerprints=("r0", "r1", "r2", "r3"),
        replay_control_batches=(batch(2.0), batch(3.0)),
    )
    context = SimpleNamespace(
        generation=5,
        snapshot=snapshot,
        controls_eligible=True,
        controls_reason=None,
        ea_sources=(SourceControlBatch("EA", "ea_0", batch(0.0)),),
        source_temporal_alignment="exact",
    )
    rl_source = SourceControlBatch("RL-own", "rl", batch(1.0))
    replay_state = SimpleNamespace(
        current_size=jnp.array(10), current_position=jnp.array(2)
    )
    return workflow, context, rl_source, replay_state


def test_deferred_diagnostic_builds_complete_frozen_matrix_with_td_stats():
    workflow, context, rl_source, replay_state = _deferred_workflow_and_context()
    diagnostic = workflow._run_deferred_source_control_diagnostic(
        context, rl_source, replay_state
    )
    assert len(diagnostic.records) == 60
    assert len(diagnostic.zero_step_results) == 20
    assert diagnostic.valid is True
    assert all(record.td_error_mean is not None for record in diagnostic.records)
    assert context.checkpoint_diagnostic is diagnostic
    assert workflow._latest_source_control_diagnostic is diagnostic


def test_deferred_diagnostic_does_not_mutate_frozen_or_replay_state():
    workflow, context, rl_source, replay_state = _deferred_workflow_and_context()
    frozen_before = (
        tree_fingerprint(context.snapshot.learner_agent_state),
        tree_fingerprint(context.snapshot.critic_opt_state),
        tree_fingerprint(replay_state),
    )
    workflow._run_deferred_source_control_diagnostic(
        context, rl_source, replay_state
    )
    frozen_after = (
        tree_fingerprint(context.snapshot.learner_agent_state),
        tree_fingerprint(context.snapshot.critic_opt_state),
        tree_fingerprint(replay_state),
    )
    assert frozen_after == frozen_before


def test_deferred_diagnostic_fingerprints_only_replay_metadata():
    workflow, context, rl_source, _ = _deferred_workflow_and_context()
    diagnostic = workflow._run_deferred_source_control_diagnostic(
        context, rl_source, _ReplayWithUnreadableCapacity()
    )
    assert diagnostic.valid is True


def test_td_statistics_do_not_depend_on_step_scale():
    workflow, context, rl_source, replay_state = _deferred_workflow_and_context()
    records = workflow._run_deferred_source_control_diagnostic(
        context, rl_source, replay_state
    ).records
    for source_id in {record.source_id for record in records}:
        by_scale = {
            record.step_scale: (
                record.td_error_mean,
                record.td_error_median,
                record.td_error_std,
            )
            for record in records
            if record.source_id == source_id
        }
        assert len(set(by_scale.values())) == 1


def test_td_statistics_do_not_depend_on_reference_index():
    workflow, context, rl_source, replay_state = _deferred_workflow_and_context()
    records = workflow._run_deferred_source_control_diagnostic(
        context, rl_source, replay_state
    ).records
    for source_id in {record.source_id for record in records}:
        values = {
            (
                record.td_error_mean,
                record.td_error_median,
                record.td_error_std,
            )
            for record in records
            if record.source_id == source_id
        }
        assert len(values) == 1


def test_td_statistics_computed_once_per_source_per_checkpoint(monkeypatch):
    workflow, context, rl_source, replay_state = _deferred_workflow_and_context()
    calls = []
    original = workflow_module.compute_source_td_stats

    def counted(**kwargs):
        calls.append(kwargs["source_batch"])
        return original(**kwargs)

    monkeypatch.setattr(workflow_module, "compute_source_td_stats", counted)
    workflow._run_deferred_source_control_diagnostic(
        context, rl_source, replay_state
    )
    assert len(calls) == 8
