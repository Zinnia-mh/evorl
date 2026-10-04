from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp

from evorl.agent import AgentState
from evorl.algorithms.erl.erl_td3.erl_utility_source_controls import (
    ERLUtilitySourceControlsWorkflow,
    SOURCE_CONTROL_TAG,
)
from evorl.algorithms.erl.erl_td3.learner_utility import derive_diagnostic_key
from evorl.types import PyTreeDict


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


def test_snapshot_selects_learner_zero_critic_target_and_optimizer():
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    context = workflow._make_pre_exposure_context(_state())
    snapshot = context.snapshot
    chex.assert_trees_all_equal(
        snapshot.learner_agent_state.params.critic_params["w"],
        jnp.array([1.0, 2.0]),
    )
    chex.assert_trees_all_equal(
        snapshot.learner_agent_state.params.target_critic_params["w"],
        jnp.array([3.0, 4.0]),
    )
    chex.assert_trees_all_equal(snapshot.critic_opt_state["mu"]["w"], jnp.array([0.1, 0.2]))


def test_snapshot_keeps_immutable_state_after_baseline_replacement():
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    state = _state()
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
    state = _state()
    before = state.key
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    context = workflow._make_pre_exposure_context(state)
    derived = derive_diagnostic_key(
        context.snapshot.baseline_key, context.snapshot.generation, SOURCE_CONTROL_TAG
    )
    chex.assert_trees_all_equal(state.key, before)
    assert not bool(jnp.all(derived == before))


def test_snapshot_records_generation_counters_and_fingerprints():
    workflow = object.__new__(ERLUtilitySourceControlsWorkflow)
    snapshot = workflow._make_pre_exposure_context(_state()).snapshot
    assert snapshot.generation == 5
    assert len(snapshot.baseline_key_fingerprint) == 64
    assert len(snapshot.critic_fingerprint) == 64
    assert len(snapshot.target_critic_fingerprint) == 64
    assert len(snapshot.optimizer_fingerprint) == 64
    assert len(snapshot.replay_metadata_fingerprint) == 64
    assert len(snapshot.counter_fingerprint) == 64
