import chex
import jax
import jax.numpy as jnp
from types import SimpleNamespace
import pytest
import optax
from hydra import compose, initialize

from evorl.algorithms.erl.erl_td3.erl_learner_utility import (
    ERLLearnerUtilityWorkflow,
    UTILITY_TAG,
    build_checkpoint_records,
)
from evorl.algorithms.erl.erl_td3.learner_utility import (
    SourceBatch,
    UtilityResult,
    derive_diagnostic_key,
)
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.agent import AgentState
from evorl.algorithms.td3 import make_mlp_td3_agent
from evorl.envs import Box
from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow
from evorl.utils.jax_utils import is_jitted


def test_diagnostic_key_does_not_advance_baseline_key():
    """The workflow's checkpoint key is derived, never consumed or written back."""
    baseline_key = jax.random.key(7)
    before = baseline_key

    derived = derive_diagnostic_key(before, generation=7, tag=UTILITY_TAG)

    assert derived.shape == before.shape
    chex.assert_trees_all_equal(before, baseline_key)
    assert ERLLearnerUtilityWorkflow.name() == "ERL-Learner-Utility"


def test_checkpoint_samples_one_shared_current_reference_batch():
    class Replay:
        def __init__(self):
            self.calls = 0

        def sample(self, state, key):
            del state, key
            self.calls += 1
            return SampleBatch(obs=jnp.array([[1.0]]), rewards=jnp.array([2.0]))

    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.replay_buffer = Replay()
    state = SimpleNamespace(replay_buffer_state=object())

    reference, fingerprint = workflow._sample_current_reference(
        state, jax.random.key(0)
    )

    assert workflow.replay_buffer.calls == 1
    assert reference.rewards.shape == (1,)
    assert fingerprint == workflow._reference_batch_fingerprint(reference)


def test_checkpoint_records_keep_required_fields_and_one_reference_identity():
    sources = (
        SourceBatch("gen_4_ind_0", 0, "batch-0", True, None, 8),
        SourceBatch("gen_4_ind_1", 1, "batch-1", True, None, 8),
    )

    records = build_checkpoint_records(
        sources=sources,
        fitnesses=jnp.array([5.0, 3.0]),
        reference_fingerprint="shared-ref",
        source_batch_size=4,
        reference_batch_size=6,
        generation=4,
        checkpoint=12,
        evaluate=lambda batch: (1.5, UtilityResult(0.25, 2.0, 1.75)),
    )

    assert [record.source_id for record in records] == ["gen_4_ind_0", "gen_4_ind_1"]
    assert {record.reference_fingerprint for record in records} == {"shared-ref"}
    assert [record.fitness for record in records] == [5.0, 3.0]
    assert all(record.td_error == 1.5 for record in records)
    assert all(record.td_error_mean == 1.5 for record in records)
    assert all(record.td_error_median == 1.5 for record in records)
    assert all(record.td_error_std == 0.0 for record in records)
    assert all(record.utility == 0.25 for record in records)
    assert all(record.source_batch_size == 4 for record in records)
    assert all(record.reference_batch_size == 6 for record in records)
    assert all(record.generation == 4 for record in records)
    assert [record.individual_id for record in records] == [0, 1]
    assert all(record.measurement_phase == "pre_exposure" for record in records)
    assert all(record.checkpoint == 12 for record in records)


def test_complete_read_only_hook_preserves_training_state_treewise():
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    state = SimpleNamespace(
        key=jax.random.key(11),
        replay_buffer_state={"cursor": jnp.array(3)},
        agent_state={"critic": jnp.array([1.0])},
        opt_state={"critic": jnp.array([2.0])},
    )
    before = (state.key, state.replay_buffer_state, state.agent_state, state.opt_state)

    workflow._run_read_only_hook(
        state,
        sources=(SourceBatch("gen_1_ind_0", 0, "batch", True, None, 4),),
        fitnesses=jnp.array([1.0]),
        reference_fingerprint="checkpoint-1",
        source_batch_size=4,
        evaluate=lambda batch: (0.5, UtilityResult(0.1, 1.0, 0.9)),
    )

    chex.assert_trees_all_equal(
        before, (state.key, state.replay_buffer_state, state.agent_state, state.opt_state)
    )


def test_pre_exposure_diagnostic_runs_before_current_generation_replay_add():
    marker = 99.0
    raw_trajectory = SampleBatch(
        obs=jnp.array([[[[marker]]]]),
        actions=jnp.zeros((1, 1, 1, 1)),
        rewards=jnp.ones((1, 1, 1)),
        dones=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
    )
    events = []

    class Replay:
        def add(self, replay_state, trajectory, mask):
            del trajectory, mask
            events.append("replay_add")
            return {"existing": jnp.append(replay_state["existing"], marker)}

    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.replay_buffer = Replay()

    def diagnostic(raw, replay_state):
        events.append("diagnostic")
        assert float(raw.obs[0, 0, 0, 0]) == marker
        assert not bool(jnp.any(replay_state["existing"] == marker))
        return ("record",)

    replay_before = {"existing": jnp.array([1.0, 2.0])}
    records, replay_after = workflow._process_pre_exposure_ec_trajectory(
        raw_trajectory, replay_before, diagnostic
    )

    assert events == ["diagnostic", "replay_add"]
    assert records == ("record",)
    assert bool(jnp.any(replay_after["existing"] == marker))


def test_real_ec_rollout_path_captures_raw_population_before_replay_add():
    events = []

    class Collector:
        def rollout(self, individual, key, num_episodes):
            del key, num_episodes
            metrics = PyTreeDict(
                episode_returns=jnp.asarray([individual]),
                episode_lengths=jnp.asarray([1]),
            )
            trajectory = SampleBatch(
                obs=jnp.asarray([[[individual]]]),
                actions=jnp.zeros((1, 1, 1)),
                rewards=jnp.asarray([[individual]]),
                dones=jnp.zeros((1, 1), dtype=jnp.bool_),
            )
            return metrics, trajectory

    class Replay:
        def add(self, replay_state, trajectory, mask):
            del trajectory, mask
            events.append("replay_add")
            return replay_state + 1

    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.ec_collector = Collector()
    workflow.replay_buffer = Replay()
    workflow.agent_state_vmap_axes = 0
    workflow.config = SimpleNamespace(pop_size=2, episodes_for_fitness=1)
    workflow._diagnostic_enabled = True
    workflow._diagnostic_active = True

    def diagnostic(raw, replay_state):
        events.append("diagnostic")
        assert raw.obs.shape[:3] == (2, 1, 1)
        chex.assert_trees_all_equal(raw.obs[:, 0, 0, 0], jnp.array([10.0, 20.0]))
        assert replay_state == 7
        return ("record",)

    workflow._run_pre_exposure_diagnostic = diagnostic
    _, flattened, replay_after = workflow._ec_rollout(
        jnp.array([10.0, 20.0]), 7, jax.random.key(3)
    )

    assert events == ["diagnostic", "replay_add"]
    assert flattened.obs.shape == (2, 1)
    assert replay_after == 8


def test_pre_exposure_checkpoint_samples_one_real_reference_for_all_sources():
    class Replay:
        def __init__(self):
            self.sample_calls = 0

        def can_sample(self, replay_state):
            del replay_state
            return True

        def sample(self, replay_state, key):
            del replay_state, key
            self.sample_calls += 1
            return SampleBatch(
                obs=jnp.array([[3.0]]),
                actions=jnp.zeros((1, 1)),
                rewards=jnp.array([1.0]),
            )

    raw = SampleBatch(
        obs=jnp.array([[[[10.0]], [[11.0]]], [[[20.0]], [[21.0]]]]),
        actions=jnp.zeros((2, 2, 1, 1)),
        rewards=jnp.ones((2, 2, 1)),
        dones=jnp.zeros((2, 2, 1), dtype=jnp.bool_),
    )
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.replay_buffer = Replay()
    workflow.config = SimpleNamespace(source_batch_size=2)
    workflow._diagnostic_context = SimpleNamespace(
        generation=5,
        key=jax.random.key(9),
        fitnesses=jnp.array([7.0, 4.0]),
    )
    workflow._evaluate_source_utility = lambda source, reference, key: (
        0.5,
        UtilityResult(0.2, 1.0, 0.8),
    )

    records = workflow._run_pre_exposure_diagnostic(raw, {"size": 8})

    assert workflow.replay_buffer.sample_calls == 1
    assert len(records) == 2
    assert {record.reference_fingerprint for record in records} == {
        records[0].reference_fingerprint
    }
    assert [record.source_id for record in records] == ["gen_5_ind_0", "gen_5_ind_1"]
    assert all(record.source_batch_size == 2 for record in records)


def test_source_utility_uses_real_td3_loss_and_functional_optax_state():
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = make_mlp_td3_agent(
        action_space,
        critic_hidden_layer_sizes=(8,),
        actor_hidden_layer_sizes=(8,),
        policy_noise=0.0,
    )
    agent_state = agent.init(obs_space, action_space, jax.random.key(1))
    optimizer = optax.adam(1e-3)
    critic_opt_state = optimizer.init(agent_state.params.critic_params)
    batch = SampleBatch(
        obs=jnp.ones((4, 3)),
        actions=jnp.zeros((4, 2)),
        rewards=jnp.array([0.0, 1.0, 2.0, 4.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.full((4, 3), 0.5),
                termination=jnp.zeros(4),
            )
        ),
    )
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.agent = agent
    workflow.optimizer = optimizer
    workflow._diagnostic_context = SimpleNamespace(
        learner_agent_state=agent_state,
        critic_opt_state=critic_opt_state,
    )
    protected = (agent_state, critic_opt_state)

    td_stats, result = workflow._evaluate_source_utility(
        batch, batch, jax.random.key(2)
    )

    assert bool(jnp.isfinite(td_stats.mean))
    assert bool(jnp.isfinite(td_stats.median))
    assert bool(jnp.isfinite(td_stats.std))
    assert float(td_stats.std) > 0
    assert bool(jnp.isfinite(result.utility))
    assert float(result.utility) == pytest.approx(
        float(result.loss_ref_before - result.loss_ref_after)
    )
    chex.assert_trees_all_equal(
        protected,
        (
            workflow._diagnostic_context.learner_agent_state,
            workflow._diagnostic_context.critic_opt_state,
        ),
    )


def _minimal_rollout_workflow(workflow_type, *, enabled):
    class Collector:
        def rollout(self, individual, key, num_episodes):
            del key, num_episodes
            return (
                PyTreeDict(
                    episode_returns=jnp.asarray([individual]),
                    episode_lengths=jnp.asarray([1]),
                ),
                SampleBatch(
                    obs=jnp.asarray([[[individual]]]),
                    actions=jnp.zeros((1, 1, 1)),
                    rewards=jnp.asarray([[individual]]),
                    dones=jnp.zeros((1, 1), dtype=jnp.bool_),
                ),
            )

    class Replay:
        def add(self, replay_state, trajectory, mask):
            del trajectory, mask
            return replay_state + 1

    workflow = object.__new__(workflow_type)
    workflow.ec_collector = Collector()
    workflow.replay_buffer = Replay()
    workflow.agent_state_vmap_axes = 0
    workflow.config = SimpleNamespace(pop_size=2, episodes_for_fitness=1)
    if isinstance(workflow, ERLLearnerUtilityWorkflow):
        workflow._diagnostic_enabled = enabled
        workflow._diagnostic_active = enabled
        workflow._run_pre_exposure_diagnostic = lambda raw, replay: ("record",)
    return workflow


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostic_rollout_training_output_is_exactly_baseline(enabled):
    baseline = _minimal_rollout_workflow(ERLWorkflow, enabled=False)
    diagnostic = _minimal_rollout_workflow(
        ERLLearnerUtilityWorkflow, enabled=enabled
    )
    inputs = (jnp.array([10.0, 20.0]), 7, jax.random.key(3))

    baseline_output = baseline._ec_rollout(*inputs)
    diagnostic_output = diagnostic._ec_rollout(*inputs)

    chex.assert_trees_all_equal(baseline_output, diagnostic_output)
    if enabled:
        assert diagnostic._latest_diagnostic_records == ("record",)


def test_diagnostic_context_is_bound_from_current_real_learner_state():
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = make_mlp_td3_agent(
        action_space,
        critic_hidden_layer_sizes=(8,),
        actor_hidden_layer_sizes=(8,),
    )
    batched_agent_state = jax.vmap(agent.init, in_axes=(None, None, 0))(
        obs_space, action_space, jax.random.split(jax.random.key(1), 2)
    )
    optimizer = optax.adam(1e-3)
    critic_opt_state = optimizer.init(batched_agent_state.params.critic_params)
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.agent_state_vmap_axes = AgentState(params=0, obs_preprocessor_state=None)
    state = PyTreeDict(
        key=jax.random.key(7),
        agent_state=batched_agent_state,
        opt_state=PyTreeDict(critic=critic_opt_state),
        metrics=PyTreeDict(iterations=jnp.array(4)),
    )

    context = workflow._make_pre_exposure_context(state)

    assert context.generation == 5
    chex.assert_trees_all_equal(
        context.learner_agent_state.params.critic_params,
        jax.tree.map(lambda leaf: leaf[0], batched_agent_state.params.critic_params),
    )
    chex.assert_trees_all_equal(context.key, state.key)


def test_step_automatically_scopes_pre_exposure_context(monkeypatch):
    state = SimpleNamespace(
        name="training-state", metrics=SimpleNamespace(iterations=jnp.array(0))
    )
    observed = []

    def parent_step(workflow, received_state):
        observed.append((received_state, workflow._diagnostic_context))
        return "metrics", received_state

    monkeypatch.setattr(ERLWorkflow, "step", parent_step)
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow._diagnostic_enabled = True
    workflow.config = SimpleNamespace(diagnostic_interval=1)
    expected_context = SimpleNamespace(name="context")
    workflow._make_pre_exposure_context = lambda received: expected_context

    result = workflow.step(state)

    assert result == ("metrics", state)
    assert observed == [(state, expected_context)]
    assert not hasattr(workflow, "_diagnostic_context")


def test_insufficient_reference_replay_skips_checkpoint_without_sampling():
    class Replay:
        def __init__(self):
            self.sample_calls = 0

        def can_sample(self, replay_state):
            del replay_state
            return False

        def sample(self, replay_state, key):
            del replay_state, key
            self.sample_calls += 1
            raise AssertionError("sample must not run when replay is insufficient")

    raw = SampleBatch(
        obs=jnp.zeros((2, 2, 1, 1)),
        actions=jnp.zeros((2, 2, 1, 1)),
        rewards=jnp.ones((2, 2, 1)),
        dones=jnp.zeros((2, 2, 1), dtype=jnp.bool_),
    )
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.replay_buffer = Replay()
    workflow.config = SimpleNamespace(source_batch_size=2)
    workflow._diagnostic_context = SimpleNamespace(
        generation=2,
        key=jax.random.key(1),
        fitnesses=jnp.array([3.0, 2.0]),
    )

    records = workflow._run_pre_exposure_diagnostic(raw, object())

    assert workflow.replay_buffer.sample_calls == 0
    assert len(records) == 2
    assert all(record.eligible is False for record in records)
    assert {record.reason for record in records} == {"insufficient_reference_replay"}


def test_diagnostic_checkpoint_cadence_uses_next_generation_number():
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow._diagnostic_enabled = True
    workflow.config = SimpleNamespace(diagnostic_interval=3)

    assert workflow._diagnostic_due(
        PyTreeDict(metrics=PyTreeDict(iterations=jnp.array(2)))
    )
    assert not workflow._diagnostic_due(
        PyTreeDict(metrics=PyTreeDict(iterations=jnp.array(3)))
    )


def test_build_from_config_binds_diagnostic_enabled(monkeypatch):
    built = object.__new__(ERLLearnerUtilityWorkflow)

    def parent_build(cls, config):
        del cls, config
        return built

    monkeypatch.setattr(
        ERLWorkflow, "_build_from_config", classmethod(parent_build)
    )

    result = ERLLearnerUtilityWorkflow._build_from_config(
        SimpleNamespace(diagnostic_enabled=True)
    )

    assert result is built
    assert result._diagnostic_enabled is True


def test_learner_utility_hydra_config_defaults_to_disabled():
    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=["agent=erl/erl-learner-utility", "env=brax/swimmer"],
        )

    assert config.workflow_cls.endswith("ERLLearnerUtilityWorkflow")
    assert config.diagnostic_enabled is False
    assert config.diagnostic_interval == 10
    assert config.source_batch_size == 256
    assert config.reference_setting == "current_replay_pre_exposure"


def test_real_ec_rollout_runs_complete_read_only_utility_chain():
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = make_mlp_td3_agent(
        action_space,
        critic_hidden_layer_sizes=(8,),
        actor_hidden_layer_sizes=(8,),
        policy_noise=0.0,
    )
    learner_state = agent.init(obs_space, action_space, jax.random.key(10))
    optimizer = optax.adam(1e-3)
    critic_opt_state = optimizer.init(learner_state.params.critic_params)
    reference = SampleBatch(
        obs=jnp.zeros((3, 3)),
        actions=jnp.zeros((3, 2)),
        rewards=jnp.array([0.0, 1.0, 2.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.full((3, 3), 0.25),
                termination=jnp.zeros(3),
            )
        ),
    )

    class Collector:
        def rollout(self, individual, key, num_episodes):
            del key, num_episodes
            obs = jnp.full((2, 1, 3), individual)
            return (
                PyTreeDict(
                    episode_returns=jnp.asarray([individual]),
                    episode_lengths=jnp.asarray([2]),
                ),
                SampleBatch(
                    obs=obs,
                    actions=jnp.zeros((2, 1, 2)),
                    rewards=jnp.asarray([[individual], [individual + 1.0]]),
                    dones=jnp.zeros((2, 1), dtype=jnp.bool_),
                    extras=PyTreeDict(
                        env_extras=PyTreeDict(
                            ori_obs=obs + 0.5,
                            termination=jnp.zeros((2, 1)),
                        )
                    ),
                ),
            )

    class Replay:
        def __init__(self):
            self.sample_calls = 0

        def can_sample(self, replay_state):
            del replay_state
            return True

        def sample(self, replay_state, key):
            del replay_state, key
            self.sample_calls += 1
            return reference

        def add(self, replay_state, trajectory, mask):
            del trajectory, mask
            return replay_state + 1

    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.agent = agent
    workflow.optimizer = optimizer
    workflow.ec_collector = Collector()
    workflow.replay_buffer = Replay()
    workflow.agent_state_vmap_axes = 0
    workflow.config = SimpleNamespace(
        pop_size=2, episodes_for_fitness=1, source_batch_size=2
    )
    workflow._diagnostic_enabled = True
    workflow._diagnostic_active = True
    workflow._diagnostic_context = SimpleNamespace(
        generation=6,
        key=jax.random.key(11),
        learner_agent_state=learner_state,
        critic_opt_state=critic_opt_state,
        fitnesses=None,
    )
    protected = (learner_state, critic_opt_state, workflow._diagnostic_context.key)

    _, _, replay_after = workflow._ec_rollout(
        jnp.array([1.0, 3.0]), 5, jax.random.key(12)
    )

    records = workflow._latest_diagnostic_records
    assert replay_after == 6
    assert workflow.replay_buffer.sample_calls == 1
    assert len(records) == 2
    assert all(record.eligible for record in records)
    assert {record.reference_fingerprint for record in records} == {
        records[0].reference_fingerprint
    }
    assert [record.fitness for record in records] == [1.0, 3.0]
    assert all(jnp.isfinite(record.utility) for record in records)
    chex.assert_trees_all_equal(
        protected,
        (
            workflow._diagnostic_context.learner_agent_state,
            workflow._diagnostic_context.critic_opt_state,
            workflow._diagnostic_context.key,
        ),
    )


def test_enabled_but_non_checkpoint_ec_rollout_is_baseline_only():
    workflow = _minimal_rollout_workflow(
        ERLLearnerUtilityWorkflow, enabled=False
    )
    workflow._diagnostic_enabled = True
    workflow._diagnostic_active = False
    del workflow._run_pre_exposure_diagnostic

    output = workflow._ec_rollout(
        jnp.array([10.0, 20.0]), 7, jax.random.key(3)
    )

    assert output[2] == 8
    assert not hasattr(workflow, "_latest_diagnostic_records")


def test_enable_jit_keeps_host_diagnostic_ec_rollout_eager():
    class JittedLearnerUtilityWorkflow(ERLLearnerUtilityWorkflow):
        pass

    JittedLearnerUtilityWorkflow.enable_jit()

    assert not is_jitted(JittedLearnerUtilityWorkflow._ec_rollout)
    assert is_jitted(JittedLearnerUtilityWorkflow._rl_rollout)
    assert is_jitted(JittedLearnerUtilityWorkflow._ec_update)
    assert is_jitted(JittedLearnerUtilityWorkflow.evaluate)
