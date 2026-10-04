from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from hydra import compose, initialize

from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
    ALLOWED_PILOT_ONLY_PATHS,
    ERLUtilityReplayWorkflow,
    UtilityReplayBuffer,
    alpha_zero_sample,
    make_ea_rollout_provenance,
)
from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow
from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    NO_SCORE_VERSION,
    RL_ORIGIN,
    RL_SCORE_VERSION,
    RL_SOURCE_GENERATION,
    RL_SOURCE_TOKEN,
    make_rl_provenance,
)
from evorl.utils.jax_utils import is_jitted


def _filled_states():
    baseline = ReplayBuffer(capacity=16, min_sample_timesteps=1, sample_batch_size=6)
    pilot = UtilityReplayBuffer(
        capacity=16, min_sample_timesteps=1, sample_batch_size=6
    )
    payload = {
        "obs": jnp.arange(32, dtype=jnp.float32).reshape(16, 2),
        "actions": jnp.arange(16, dtype=jnp.int32),
        "rewards": -jnp.arange(16, dtype=jnp.float32),
    }
    base_state = baseline.init(jtu.tree_map(lambda x: x[0], payload)).replace(
        data=payload,
        current_index=jnp.asarray(0, jnp.int32),
        buffer_size=jnp.asarray(16, jnp.int32),
    )
    pilot_state = pilot.init(jtu.tree_map(lambda x: x[0], payload)).replace(
        data=payload,
        current_index=jnp.asarray(0, jnp.int32),
        buffer_size=jnp.asarray(16, jnp.int32),
        origin_type=jnp.asarray([EA_ORIGIN, RL_ORIGIN] * 8, jnp.int8),
    )
    return baseline, base_state, pilot, pilot_state


def test_alpha_zero_indices_are_bitwise_baseline_indices():
    _, _, pilot, state = _filled_states()
    key = jax.random.key(7)
    result = alpha_zero_sample(state, key, jax.random.key(99), pilot.sample_batch_size)
    expected = jax.random.randint(key, (6,), minval=0, maxval=state.buffer_size)
    chex.assert_trees_all_equal(result.baseline_indices, expected)
    chex.assert_trees_all_equal(result.final_indices, expected)


def test_alpha_zero_batches_are_treewise_bitwise_equal():
    baseline, base_state, pilot, pilot_state = _filled_states()
    key = jax.random.key(11)
    expected = baseline.sample(base_state, key)
    actual = pilot.sample(pilot_state, key)
    chex.assert_trees_all_equal(actual, expected)


def test_alpha_zero_baseline_prng_progression_is_equal():
    _, _, pilot, state = _filled_states()
    incoming = jax.random.key(13)
    baseline_next, baseline_draw = jax.random.split(incoming)
    pilot_next, pilot_draw = jax.random.split(incoming)
    alpha_zero_sample(state, pilot_draw, jax.random.key(101), pilot.sample_batch_size)
    chex.assert_trees_all_equal(pilot_draw, baseline_draw)
    chex.assert_trees_all_equal(pilot_next, baseline_next)


def test_alpha_zero_does_not_advance_intervention_key():
    _, _, pilot, state = _filled_states()
    intervention_key = jax.random.key(17)
    result = alpha_zero_sample(
        state, jax.random.key(19), intervention_key, pilot.sample_batch_size
    )
    chex.assert_trees_all_equal(result.intervention_key, intervention_key)


def test_alpha_zero_uses_exactly_one_baseline_draw(monkeypatch):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module

    _, _, pilot, state = _filled_states()
    real_draw = module.draw_baseline_indices
    calls = []

    def counted_draw(*args, **kwargs):
        calls.append((args, kwargs))
        return real_draw(*args, **kwargs)

    monkeypatch.setattr(module, "draw_baseline_indices", counted_draw)
    result = alpha_zero_sample(
        state, jax.random.key(23), jax.random.key(29), pilot.sample_batch_size
    )
    assert len(calls) == 1
    chex.assert_trees_all_equal(result.baseline_draw_count, jnp.asarray(1, jnp.uint32))


def test_alpha_zero_pilot_metadata_does_not_affect_training_tree():
    _, _, pilot, state = _filled_states()
    changed = state.replace(
        origin_type=state.origin_type[::-1],
        source_token=jnp.arange(16, dtype=jnp.int32) + 1000,
        source_generation=jnp.arange(16, dtype=jnp.int32) + 2000,
        score_version=jnp.arange(16, dtype=jnp.int32) + 3000,
    )
    key = jax.random.key(31)
    chex.assert_trees_all_equal(pilot.sample(state, key), pilot.sample(changed, key))
    assert ALLOWED_PILOT_ONLY_PATHS == (
        "replay_buffer_state.origin_type",
        "replay_buffer_state.source_token",
        "replay_buffer_state.source_generation",
        "replay_buffer_state.score_version",
        "workflow.intervention_key",
        "train_metrics.time_cost_per_iter",
    )


def test_alpha_zero_tiny_workflow_is_exactly_baseline_equivalent():
    overrides = [
        "env=brax/swimmer",
        "env.max_episode_steps=1",
        "num_envs=1",
        "pop_size=4",
        "num_elites=1",
        "num_rl_agents=1",
        "episodes_for_fitness=1",
        "rollout_episodes=1",
        "batch_size=4",
        "replay_buffer_capacity=64",
        "warmup_iters=0",
        "actor_update_interval=1",
        "rl_updates_frac=0.25",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
        "num_eval_envs=1",
        "eval_episodes=1",
    ]
    with initialize(config_path="../configs", version_base=None):
        base_cfg = compose(
            config_name="config",
            overrides=["agent=erl/erl-ori", *overrides],
        )
        pilot_cfg = compose(
            config_name="config",
            overrides=["agent=erl/erl-utility-replay", *overrides],
        )
    baseline = ERLWorkflow.build_from_config(base_cfg, enable_jit=False)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=False)
    key = jax.random.PRNGKey(2026)
    baseline_state = baseline.setup(key)
    pilot_state = pilot.setup(key)
    initial_intervention_key = pilot.intervention_key

    for _ in range(1):
        baseline_metrics, baseline_state = baseline.step(baseline_state)
        pilot_metrics, pilot_state = pilot.step(pilot_state)

        chex.assert_trees_all_equal(baseline_state.key, pilot_state.key)
        chex.assert_trees_all_equal(
            baseline_state.agent_state, pilot_state.agent_state
        )
        chex.assert_trees_all_equal(baseline_state.opt_state, pilot_state.opt_state)
        chex.assert_trees_all_equal(
            baseline_state.ec_opt_state, pilot_state.ec_opt_state
        )
        chex.assert_trees_all_equal(baseline_state.metrics, pilot_state.metrics)
        chex.assert_trees_all_equal(
            baseline_state.replay_buffer_state.data,
            pilot_state.replay_buffer_state.data,
        )
        chex.assert_trees_all_equal(
            baseline_state.replay_buffer_state.current_index,
            pilot_state.replay_buffer_state.current_index,
        )
        chex.assert_trees_all_equal(
            baseline_state.replay_buffer_state.buffer_size,
            pilot_state.replay_buffer_state.buffer_size,
        )
        chex.assert_trees_all_equal(
            baseline_metrics.replace(time_cost_per_iter=0.0),
            pilot_metrics.replace(time_cost_per_iter=0.0),
        )
        chex.assert_trees_all_equal(pilot.intervention_key, initial_intervention_key)

    origins = pilot_state.replay_buffer_state.origin_type[
        : pilot_state.replay_buffer_state.buffer_size
    ]
    assert bool(jnp.any(origins == EA_ORIGIN))
    assert bool(jnp.any(origins == RL_ORIGIN))


def test_alpha_zero_rejects_nonzero_alpha_at_build_gate(monkeypatch):
    built = object.__new__(ERLUtilityReplayWorkflow)
    built.replay_buffer = SimpleNamespace(
        capacity=8, min_sample_timesteps=2, sample_batch_size=2
    )

    def parent_build(cls, config):
        del cls, config
        return built

    monkeypatch.setattr(ERLWorkflow, "_build_from_config", classmethod(parent_build))
    try:
        ERLUtilityReplayWorkflow._build_from_config(SimpleNamespace(alpha=0.5))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Phase 3 must reject non-zero alpha")


def test_workflow_ea_provenance_records_actual_generation():
    provenance = make_ea_rollout_provenance(
        generation=7, population_size=4, time_size=2, episode_batch=1
    )
    chex.assert_trees_all_equal(
        provenance.source_generation, jnp.full((8,), 7, dtype=jnp.int32)
    )


def test_same_individual_across_generations_has_distinct_source_token():
    gen_10 = make_ea_rollout_provenance(
        generation=10, population_size=4, time_size=1, episode_batch=1
    )
    gen_11 = make_ea_rollout_provenance(
        generation=11, population_size=4, time_size=1, episode_batch=1
    )
    assert int(gen_10.source_token[3]) != int(gen_11.source_token[3])


def test_source_token_matches_generation_population_individual_encoding():
    provenance = make_ea_rollout_provenance(
        generation=10, population_size=4, time_size=2, episode_batch=1
    )
    chex.assert_trees_all_equal(
        provenance.source_token,
        jnp.asarray([40, 41, 42, 43, 40, 41, 42, 43], dtype=jnp.int32),
    )


def test_unscored_ea_transition_uses_no_score_version_sentinel():
    provenance = make_ea_rollout_provenance(
        generation=3, population_size=2, time_size=2, episode_batch=1
    )
    chex.assert_trees_all_equal(
        provenance.score_version,
        jnp.full((4,), NO_SCORE_VERSION, dtype=jnp.int32),
    )


def test_rl_provenance_sentinels_remain_unchanged():
    provenance = make_rl_provenance(batch_size=3)
    chex.assert_trees_all_equal(
        provenance.source_token, jnp.full((3,), RL_SOURCE_TOKEN, jnp.int32)
    )
    chex.assert_trees_all_equal(
        provenance.source_generation,
        jnp.full((3,), RL_SOURCE_GENERATION, jnp.int32),
    )
    chex.assert_trees_all_equal(
        provenance.score_version, jnp.full((3,), RL_SCORE_VERSION, jnp.int32)
    )


def test_td3_tau_and_replay_temperature_are_distinct_config_fields():
    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "agent=erl/erl-utility-replay",
                "env=brax/swimmer",
                "replay_temperature=0.37",
            ],
        )
    assert config.tau == 0.005
    assert config.replay_temperature == 0.37


def test_alpha_zero_tiny_jit_workflow_is_exactly_baseline_equivalent():
    overrides = [
        "env=brax/swimmer",
        "env.max_episode_steps=1",
        "num_envs=1",
        "pop_size=4",
        "num_elites=1",
        "num_rl_agents=1",
        "episodes_for_fitness=1",
        "rollout_episodes=1",
        "batch_size=4",
        "replay_buffer_capacity=64",
        "warmup_iters=0",
        "actor_update_interval=1",
        "rl_updates_frac=0.25",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
        "num_eval_envs=1",
        "eval_episodes=1",
    ]
    with initialize(config_path="../configs", version_base=None):
        base_cfg = compose(
            config_name="config", overrides=["agent=erl/erl-ori", *overrides]
        )
        pilot_cfg = compose(
            config_name="config",
            overrides=["agent=erl/erl-utility-replay", *overrides],
        )
    baseline = ERLWorkflow.build_from_config(base_cfg, enable_jit=True)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=True)
    assert is_jitted(baseline._ec_rollout)
    assert is_jitted(pilot._baseline_ec_rollout)
    assert is_jitted(pilot._baseline_rl_rollout)
    assert not is_jitted(pilot._ec_rollout)
    assert not is_jitted(pilot._rl_rollout)

    key = jax.random.PRNGKey(3031)
    baseline_state = baseline.setup(key)
    pilot_state = pilot.setup(key)
    intervention_key = pilot.intervention_key
    baseline_metrics, baseline_state = baseline.step(baseline_state)
    pilot_metrics, pilot_state = pilot.step(pilot_state)

    chex.assert_trees_all_equal(baseline_state.key, pilot_state.key)
    chex.assert_trees_all_equal(baseline_state.agent_state, pilot_state.agent_state)
    chex.assert_trees_all_equal(baseline_state.opt_state, pilot_state.opt_state)
    chex.assert_trees_all_equal(baseline_state.ec_opt_state, pilot_state.ec_opt_state)
    chex.assert_trees_all_equal(baseline_state.metrics, pilot_state.metrics)
    chex.assert_trees_all_equal(
        baseline_state.replay_buffer_state.data,
        pilot_state.replay_buffer_state.data,
    )
    chex.assert_trees_all_equal(
        baseline_state.replay_buffer_state.current_index,
        pilot_state.replay_buffer_state.current_index,
    )
    chex.assert_trees_all_equal(
        baseline_state.replay_buffer_state.buffer_size,
        pilot_state.replay_buffer_state.buffer_size,
    )
    chex.assert_trees_all_equal(
        baseline_metrics.replace(time_cost_per_iter=0.0),
        pilot_metrics.replace(time_cost_per_iter=0.0),
    )
    chex.assert_trees_all_equal(pilot.intervention_key, intervention_key)

    _, pilot_state = pilot.step(pilot_state)
    resident = jnp.arange(pilot_state.replay_buffer_state.buffer_size)
    ea_resident = (
        pilot_state.replay_buffer_state.origin_type[resident] == EA_ORIGIN
    )
    ea_generations = pilot_state.replay_buffer_state.source_generation[resident][
        ea_resident
    ]
    assert set(map(int, jnp.unique(ea_generations).tolist())) == {1, 2}
