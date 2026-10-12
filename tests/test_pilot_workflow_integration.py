"""Synthetic P1-B hook integration: no environment rollout is started."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentState
from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import ERLUtilitySourceControlsWorkflow
from evorl.algorithms.td3 import TD3NetworkParams
from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict


class Collector:
    def rollout(self, individual, key, episodes):
        del key, episodes
        value = individual.params.actor_params["w"][0]
        trajectory = SampleBatch(
            obs=jnp.full((8,1,1), value),
            actions=jnp.zeros((8,1,1)),
            rewards=jnp.full((8,1), value),
            dones=jnp.zeros((8,1), dtype=jnp.bool_),
        )
        metrics = PyTreeDict(
            episode_returns=jnp.asarray([value]),
            episode_lengths=jnp.asarray([8]),
        )
        return metrics, trajectory


class Agent:
    def critic_loss(self, state, batch, key):
        del key
        w = state.params.critic_params["w"]
        return SimpleNamespace(critic_loss=jnp.mean((w-batch.obs)**2))


class Writer:
    def __init__(self):
        self.items = []
    def append_checkpoint(self, **kwargs):
        self.items.append(kwargs)


def setup(initial_replay=40):
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    workflow._pilot_enabled = True
    workflow._pilot_run_id = "synthetic-run"
    workflow._diagnostic_active = True
    workflow._pilot_writer = Writer()
    workflow.agent = Agent()
    workflow.optimizer = optax.sgd(0.1)
    workflow.replay_buffer = ReplayBuffer(capacity=64, sample_batch_size=2)
    workflow.ec_collector = Collector()
    workflow.rl_collector = Collector()
    workflow.agent_state_vmap_axes = 0
    workflow.config = SimpleNamespace(
        source_batch_size=2, num_reference_batches=4,
        num_random_control_batches=8, pop_size=5,
        episodes_for_fitness=1, num_rl_agents=1, rollout_episodes=1,
        source_control_step_scales=(1.0,0.5,0.1),
    )
    params = PyTreeDict(
        critic_params={"w": jnp.ones((5,1))},
        target_critic_params={"w": jnp.ones((5,1))},
        actor_params={"w": jnp.arange(5., dtype=jnp.float32)[:,None]},
        target_actor_params={"w": jnp.zeros((5,1))},
    )
    state = SimpleNamespace(
        agent_state=AgentState(params=params, obs_preprocessor_state=None),
        opt_state=SimpleNamespace(critic=workflow.optimizer.init({"w":jnp.ones((5,1))})),
        key=jax.random.key(7),
        metrics=SimpleNamespace(iterations=jnp.asarray(4), sampled_episodes=jnp.asarray(0), sampled_timesteps=jnp.asarray(0), rl_sampled_timesteps=jnp.asarray(0)),
    )
    spec = SampleBatch(obs=jnp.zeros((1,)), actions=jnp.zeros((1,)), rewards=jnp.zeros(()))
    replay = workflow.replay_buffer.init(spec)
    if initial_replay:
        replay = workflow.replay_buffer.add(replay, SampleBatch(
            obs=jnp.arange(initial_replay, dtype=jnp.float32)[:,None],
            actions=jnp.zeros((initial_replay,1)),
            rewards=jnp.zeros((initial_replay,)),
        ))
    state.replay_buffer_state = replay
    workflow._diagnostic_context = workflow._make_pre_exposure_context(state)
    return workflow, state


def test_full_pilot_hooks_use_real_order_and_paired_results():
    workflow, state = setup()
    context = workflow._diagnostic_context
    assert context.controls_eligible
    before_key = state.key
    _, _, after_ea = workflow._ec_rollout(state.agent_state, state.replay_buffer_state, jax.random.key(8))
    rl_params = jax.tree_util.tree_map(lambda x: x[:1], state.agent_state.params)
    rl_agent_state = state.agent_state.replace(params=rl_params)
    _, _, after_rl = workflow._rl_rollout(rl_agent_state, after_ea, jax.random.key(9))
    written = workflow._pilot_writer.items
    assert len(written) == 1
    item = written[0]
    assert item["status"] == "PASS", item["reasons"]
    assert len(item["source_results"]) == 12
    assert all(len(x.records) == 24 for x in item["source_results"].values())
    assert list(item["provenance"]["event_ordinals"].values()) == [1,2,3,4,5]
    assert item["provenance"]["gate_results"] and all(x["eligible"] for x in item["provenance"]["gate_results"])
    assert int(after_rl.buffer_size) > int(state.replay_buffer_state.buffer_size)
    assert jnp.array_equal(state.key, before_key)


def test_pilot_replay_shortage_returns_hold_without_virtual_update():
    workflow, state = setup(initial_replay=20)
    _, _, after_ea = workflow._ec_rollout(state.agent_state, state.replay_buffer_state, jax.random.key(8))
    rl_params = jax.tree_util.tree_map(lambda x: x[:1], state.agent_state.params)
    workflow._rl_rollout(state.agent_state.replace(params=rl_params), after_ea, jax.random.key(9))
    item = workflow._pilot_writer.items[0]
    assert item["status"] == "HOLD"
    assert "insufficient_disjoint_replay_slots" in item["reasons"]
    assert item["source_results"] == {}

def test_ea_shortage_is_hold_not_exception():
    workflow, state = setup()
    class ShortCollector(Collector):
        def rollout(self, individual, key, episodes):
            metric, batch = super().rollout(individual, key, episodes)
            return metric, jax.tree_util.tree_map(lambda x: x[:1], batch)
    workflow.ec_collector = ShortCollector()
    _, _, after_ea = workflow._ec_rollout(state.agent_state, state.replay_buffer_state, jax.random.key(8))
    rl_params = jax.tree_util.tree_map(lambda x: x[:1], state.agent_state.params)
    workflow._rl_rollout(state.agent_state.replace(params=rl_params), after_ea, jax.random.key(9))
    item = workflow._pilot_writer.items[0]
    assert item["status"] == "HOLD"
    assert "insufficient_fresh_samples" in item["reasons"]
    assert item["source_results"] == {}


def test_rl_fresh_shortage_is_hold_without_reusing_own_positions():
    workflow, state = setup()
    workflow.config.source_batch_size = 3
    workflow.replay_buffer = workflow.replay_buffer.replace(sample_batch_size=3)
    workflow._diagnostic_context = workflow._make_pre_exposure_context(state)
    _, _, after_ea = workflow._ec_rollout(state.agent_state, state.replay_buffer_state, jax.random.key(8))
    rl_params = jax.tree_util.tree_map(lambda x: x[:1], state.agent_state.params)
    workflow._rl_rollout(state.agent_state.replace(params=rl_params), after_ea, jax.random.key(9))
    item = workflow._pilot_writer.items[0]
    assert item["status"] == "HOLD"
    assert "insufficient_fresh_samples" in item["reasons"]
    assert item["source_results"] == {}


def test_pilot_requires_exact_reference_and_control_counts():
    import pytest
    workflow, state = setup()
    workflow.config.num_reference_batches = 3
    with pytest.raises(ValueError, match="exactly four"):
        workflow._make_pre_exposure_context(state)

def test_pilot_on_off_keep_same_real_replay_add_and_rollout():
    import chex
    on, state_on = setup()
    off, state_off = setup()
    off._pilot_enabled = False
    off._run_deferred_source_control_diagnostic = lambda *args: None
    ea_on, trajectory_on, replay_ea_on = on._ec_rollout(
        state_on.agent_state, state_on.replay_buffer_state, jax.random.key(8)
    )
    ea_off, trajectory_off, replay_ea_off = off._ec_rollout(
        state_off.agent_state, state_off.replay_buffer_state, jax.random.key(8)
    )
    chex.assert_trees_all_equal(ea_on, ea_off)
    chex.assert_trees_all_equal(trajectory_on, trajectory_off)
    chex.assert_trees_all_equal(replay_ea_on, replay_ea_off)
    rl_on = state_on.agent_state.replace(params=jax.tree_util.tree_map(
        lambda x: x[:1], state_on.agent_state.params
    ))
    rl_off = state_off.agent_state.replace(params=jax.tree_util.tree_map(
        lambda x: x[:1], state_off.agent_state.params
    ))
    rl_metrics_on, rl_trajectory_on, replay_rl_on = on._rl_rollout(
        rl_on, replay_ea_on, jax.random.key(9)
    )
    rl_metrics_off, rl_trajectory_off, replay_rl_off = off._rl_rollout(
        rl_off, replay_ea_off, jax.random.key(9)
    )
    chex.assert_trees_all_equal(rl_metrics_on, rl_metrics_off)
    chex.assert_trees_all_equal(rl_trajectory_on, rl_trajectory_off)
    chex.assert_trees_all_equal(replay_rl_on, replay_rl_off)
    chex.assert_trees_all_equal(state_on.key, state_off.key)
    assert on._pilot_writer.items[0]["status"] == "PASS"
