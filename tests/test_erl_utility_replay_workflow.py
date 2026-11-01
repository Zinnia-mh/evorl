from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from hydra import compose, initialize

from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
    ALLOWED_PILOT_ONLY_PATHS,
    ERLUtilityReplayWorkflow,
    UtilityReplayBuffer,
    apply_diagnostic_checkpoint,
    alpha_zero_sample,
    init_utility_replay_state,
    make_ea_rollout_provenance,
    refresh_replay_distribution,
    weighted_replay_draws,
    weighted_replay_sample,
)
from evorl.algorithms.erl.erl_td3.erl_learner_utility import (
    DiagnosticRecord,
    ERLLearnerUtilityWorkflow,
)
from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import create_dummy_td3_trainmetric
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
from evorl.algorithms.erl.erl_td3.source_replay import (
    SCORE_KIND_FITNESS,
    SCORE_KIND_TD,
    SCORE_KIND_UTILITY,
    build_weighted_source_distribution,
    init_source_score_table,
)


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


def _prepared_distribution(capacity=16):
    table = init_source_score_table(2)._replace(
        source_token=jnp.asarray([10, 20], dtype=jnp.int32),
        rank_normalized_score=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        resident_count=jnp.asarray([4, 4], dtype=jnp.int32),
        weighted_eligible=jnp.asarray([True, True]),
    )
    tokens = jnp.asarray(
        [10, -1, 10, -1, 10, -1, 10, -1,
         20, -1, 20, -1, 20, -1, 20, -1],
        dtype=jnp.int32,
    )[:capacity]
    resident_mask = jnp.ones((capacity,), dtype=jnp.bool_)
    return build_weighted_source_distribution(
        table,
        resident_source_tokens=tokens,
        resident_mask=resident_mask,
        alpha=0.8,
        replay_temperature=0.4,
    )


def test_real_overlay_gathers_payload_from_final_indices():
    _, _, pilot, state = _filled_states()
    result = weighted_replay_sample(
        state,
        jax.random.key(7),
        jax.random.key(11),
        _prepared_distribution(),
        jnp.asarray(True),
        pilot.sample_batch_size,
    )
    expected = jtu.tree_map(lambda x: x[result.final_indices], state.data)
    chex.assert_trees_all_equal(result.batch, expected)
    assert result.batch["obs"].shape[0] == pilot.sample_batch_size


def test_real_overlay_uses_one_and_only_one_baseline_draw(monkeypatch):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module

    _, _, pilot, state = _filled_states()
    real_draw = module.draw_baseline_indices
    calls = []

    def counted_draw(*args, **kwargs):
        calls.append(None)
        return real_draw(*args, **kwargs)

    monkeypatch.setattr(module, "draw_baseline_indices", counted_draw)
    weighted_replay_sample(
        state,
        jax.random.key(13),
        jax.random.key(17),
        _prepared_distribution(),
        jnp.asarray(True),
        pilot.sample_batch_size,
    )
    assert len(calls) == 1


def test_no_prepared_distribution_is_neutral_and_keeps_key():
    baseline, base_state, pilot, state = _filled_states()
    replay_key = jax.random.key(19)
    intervention_key = jax.random.key(23)
    result = weighted_replay_sample(
        state,
        replay_key,
        intervention_key,
        _prepared_distribution(),
        jnp.asarray(False),
        pilot.sample_batch_size,
    )
    chex.assert_trees_all_equal(result.batch, baseline.sample(base_state, replay_key))
    chex.assert_trees_all_equal(result.final_indices, result.baseline_indices)
    chex.assert_trees_all_equal(result.next_intervention_key, intervention_key)


def test_vmapped_replay_draws_receive_distinct_intervention_subkeys():
    _, _, pilot, state = _filled_states()
    utility_state = init_utility_replay_state(
        replay_capacity=16, intervention_key=jax.random.key(29)
    )._replace(
        distribution=_prepared_distribution(),
        distribution_prepared=jnp.asarray(True),
        distribution_version=jnp.asarray(3, dtype=jnp.int32),
    )
    replay_keys = jax.random.split(jax.random.key(31), 4)
    result = weighted_replay_draws(
        state, replay_keys, utility_state, pilot.sample_batch_size
    )

    assert result.batches["obs"].shape == (4, pilot.sample_batch_size, 2)
    assert len({
        tuple(map(int, jax.random.key_data(key).tolist()))
        for key in result.intervention_subkeys
    }) == 4
    assert not bool(jnp.array_equal(
        result.utility_replay_state.intervention_key,
        utility_state.intervention_key,
    ))


def test_alpha_zero_static_bypass_does_not_call_weighted_overlay(monkeypatch):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module

    _, _, pilot, state = _filled_states()

    def forbidden(*args, **kwargs):
        raise AssertionError("alpha=0 must not call the weighted overlay")

    monkeypatch.setattr(module, "apply_weighted_ea_overlay", forbidden)
    pilot.sample(state, jax.random.key(37))


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
        "utility_replay_state",
        "train_metrics.time_cost_per_iter",
    )


def test_alpha_zero_lifecycle_preserves_exact_baseline_training_behavior():
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
    initial_intervention_key = pilot_state.utility_replay_state.intervention_key

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
        chex.assert_trees_all_equal(
            pilot_state.utility_replay_state.intervention_key,
            initial_intervention_key,
        )

    origins = pilot_state.replay_buffer_state.origin_type[
        : pilot_state.replay_buffer_state.buffer_size
    ]
    assert bool(jnp.any(origins == EA_ORIGIN))
    assert bool(jnp.any(origins == RL_ORIGIN))


def test_nonzero_alpha_builds_weighted_learner_closure(monkeypatch):
    built = object.__new__(ERLUtilityReplayWorkflow)
    built.replay_buffer = SimpleNamespace(
        capacity=8, min_sample_timesteps=2, sample_batch_size=2
    )

    def parent_build(cls, config):
        del cls, config
        return built

    monkeypatch.setattr(ERLWorkflow, "_build_from_config", classmethod(parent_build))
    config = SimpleNamespace(arm="utility", score_kind="utility", diagnostic_enabled=True, freshness_window=3, min_resident=1, replay_temperature=1.0, alpha=0.5, actor_update_interval=2, num_rl_agents=1, warmup_iters=0, pop_size=2)
    result = ERLUtilityReplayWorkflow._build_from_config(config)
    assert result is built
    assert callable(result._rl_sample_and_update_fn)


def test_nonzero_workflow_without_distribution_is_neutral():
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
            overrides=["agent=erl/erl-utility-replay", "arm=utility", "score_kind=utility", "diagnostic_enabled=true", "alpha=0.8", *overrides],
        )
    baseline = ERLWorkflow.build_from_config(base_cfg, enable_jit=False)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=False)
    key = jax.random.PRNGKey(47)
    baseline_state = baseline.setup(key)
    pilot_state = pilot.setup(key)
    initial_intervention_key = pilot_state.utility_replay_state.intervention_key

    baseline_metrics, baseline_state = baseline.step(baseline_state)
    pilot_metrics, pilot_state = pilot.step(pilot_state)

    chex.assert_trees_all_equal(baseline_state.key, pilot_state.key)
    chex.assert_trees_all_equal(baseline_state.agent_state, pilot_state.agent_state)
    chex.assert_trees_all_equal(baseline_state.opt_state, pilot_state.opt_state)
    chex.assert_trees_all_equal(
        baseline_metrics.num_updates_per_iter,
        pilot_metrics.num_updates_per_iter,
    )
    chex.assert_trees_all_equal(
        pilot_state.utility_replay_state.intervention_key,
        initial_intervention_key,
    )


def test_weighted_learner_closure_uses_final_indices_once(monkeypatch):
    built = object.__new__(ERLUtilityReplayWorkflow)
    built.replay_buffer = SimpleNamespace(
        capacity=16, min_sample_timesteps=1, sample_batch_size=6
    )

    def parent_build(cls, config):
        del cls, config
        return built

    monkeypatch.setattr(ERLWorkflow, "_build_from_config", classmethod(parent_build))
    config = SimpleNamespace(arm="utility", score_kind="utility", diagnostic_enabled=True, freshness_window=3, min_resident=1, replay_temperature=1.0, alpha=0.8, actor_update_interval=2, num_rl_agents=1, warmup_iters=0, pop_size=2)
    workflow = ERLUtilityReplayWorkflow._build_from_config(config)
    _, _, _, replay_state = _filled_states()
    utility_state = init_utility_replay_state(
        16, jax.random.key(41)
    )._replace(
        distribution=_prepared_distribution(),
        distribution_prepared=jnp.asarray(True),
        distribution_version=jnp.asarray(5, dtype=jnp.int32),
    )
    captured = []

    def fake_update(agent_state, opt_state, sample_batches, learn_key):
        del learn_key
        captured.append(sample_batches)
        zero = jnp.asarray(0.0)
        return (agent_state, opt_state), (
            zero,
            zero,
            {"critic_loss": zero, "q_value": zero},
            {"actor_loss": zero},
        )

    workflow._rl_update_fn = fake_update
    learner_key = jax.random.key(43)
    init_metrics = create_dummy_td3_trainmetric(1)
    carry = (
        learner_key,
        "agent",
        "optimizer",
        replay_state,
        init_metrics,
        utility_state,
    )
    updated, _ = workflow._rl_sample_and_update_fn(carry, None)

    expected_next_key, rb_key, _ = jax.random.split(learner_key, 3)
    rb_keys = jax.random.split(rb_key, 2)
    expected = weighted_replay_draws(
        replay_state, rb_keys, utility_state, 6
    )
    expected_batches = jtu.tree_map(
        lambda x: x.reshape((2, 1, *x.shape[1:])), expected.batches
    )
    assert len(captured) == 1
    chex.assert_trees_all_equal(captured[0], expected_batches)
    chex.assert_trees_all_equal(updated[0], expected_next_key)
    chex.assert_trees_all_equal(updated[-1], expected.utility_replay_state)


def test_prepared_distribution_advances_across_real_workflow_step():
    overrides = [
        "env=brax/swimmer", "env.max_episode_steps=1", "num_envs=1",
        "pop_size=4", "num_elites=1", "num_rl_agents=1",
        "episodes_for_fitness=1", "rollout_episodes=1", "batch_size=8",
        "replay_buffer_capacity=64", "warmup_iters=0",
        "actor_update_interval=1", "rl_updates_frac=0.25",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
        "num_eval_envs=1", "eval_episodes=1",
    ]
    with initialize(config_path="../configs", version_base=None):
        base_cfg = compose(
            config_name="config", overrides=["agent=erl/erl-ori", *overrides]
        )
        pilot_cfg = compose(
            config_name="config",
            overrides=[
                "agent=erl/erl-utility-replay",
                "alpha=0.8",
                "arm=utility",
                "score_kind=utility",
                "diagnostic_enabled=true",
                "min_resident=1",
                *overrides,
            ],
        )
    baseline = ERLWorkflow.build_from_config(base_cfg, enable_jit=False)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=False)
    key = jax.random.PRNGKey(53)
    baseline_state = baseline.setup(key)
    pilot_state = pilot.setup(key)
    _, baseline_state = baseline.step(baseline_state)
    _, pilot_state = pilot.step(pilot_state)

    replay = pilot_state.replay_buffer_state
    capacity = replay.origin_type.shape[0]
    resident = jnp.arange(capacity) < replay.buffer_size
    ea_resident = resident & (replay.origin_type == EA_ORIGIN)
    resident_tokens = jnp.where(ea_resident, replay.source_token, -1)
    tokens = sorted(set(map(int, replay.source_token[ea_resident].tolist())))
    counts = jnp.asarray(
        [jnp.sum(resident_tokens == token) for token in tokens],
        dtype=jnp.int32,
    )
    table = pilot_state.utility_replay_state.score_table
    slots = jnp.arange(len(tokens), dtype=jnp.int32)
    table = table._replace(
        source_token=table.source_token.at[slots].set(
            jnp.asarray(tokens, dtype=jnp.int32)
        ),
        raw_score=table.raw_score.at[slots].set(
            jnp.linspace(1.0, 0.0, len(tokens))
        ),
        rank_normalized_score=table.rank_normalized_score.at[slots].set(
            jnp.linspace(1.0, 0.0, len(tokens))
        ),
        measured_generation=table.measured_generation.at[slots].set(1),
        score_kind=table.score_kind.at[slots].set(SCORE_KIND_UTILITY),
        validity=table.validity.at[slots].set(True),
        resident_count=table.resident_count.at[slots].set(counts),
        weighted_eligible=table.weighted_eligible.at[slots].set(True),
        version=table.version.at[slots].set(1),
        current_version=jnp.asarray(1, dtype=jnp.int32),
    )
    distribution = build_weighted_source_distribution(
        table,
        resident_source_tokens=resident_tokens,
        resident_mask=ea_resident,
        alpha=0.8,
        replay_temperature=0.4,
    )
    old_intervention_key = pilot_state.utility_replay_state.intervention_key
    pilot_state = pilot_state.replace(
        utility_replay_state=pilot_state.utility_replay_state._replace(
            score_table=table,
            distribution=distribution,
            distribution_prepared=jnp.asarray(True),
            distribution_version=jnp.asarray(1, dtype=jnp.int32),
            distribution_generation=jnp.asarray(1, dtype=jnp.int32),
        )
    )

    baseline_metrics, baseline_state = baseline.step(baseline_state)
    pilot_metrics, pilot_state = pilot.step(pilot_state)

    chex.assert_trees_all_equal(baseline_state.key, pilot_state.key)
    chex.assert_trees_all_equal(
        baseline_metrics.num_updates_per_iter,
        pilot_metrics.num_updates_per_iter,
    )
    assert not bool(jnp.array_equal(
        pilot_state.utility_replay_state.intervention_key,
        old_intervention_key,
    ))


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
    intervention_key = pilot_state.utility_replay_state.intervention_key
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
    chex.assert_trees_all_equal(
        pilot_state.utility_replay_state.intervention_key,
        intervention_key,
    )

    _, pilot_state = pilot.step(pilot_state)
    resident = jnp.arange(pilot_state.replay_buffer_state.buffer_size)
    ea_resident = (
        pilot_state.replay_buffer_state.origin_type[resident] == EA_ORIGIN
    )
    ea_generations = pilot_state.replay_buffer_state.source_generation[resident][
        ea_resident
    ]
    assert set(map(int, jnp.unique(ea_generations).tolist())) == {1, 2}


def test_weighted_learner_sampling_closure_runs_jitted():
    overrides = [
        "env=brax/swimmer", "env.max_episode_steps=1", "num_envs=1",
        "pop_size=4", "num_elites=1", "num_rl_agents=1",
        "episodes_for_fitness=1", "rollout_episodes=1", "batch_size=4",
        "replay_buffer_capacity=64", "warmup_iters=0",
        "actor_update_interval=1", "rl_updates_frac=0.25",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
        "num_eval_envs=1", "eval_episodes=1",
    ]
    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "agent=erl/erl-utility-replay", "arm=utility", "score_kind=utility", "diagnostic_enabled=true", "alpha=0.8", *overrides
            ],
        )
    workflow = ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=True)
    assert is_jitted(workflow._rl_sample_and_update_fn)
    state = workflow.setup(jax.random.PRNGKey(59))
    initial_key = state.utility_replay_state.intervention_key
    metrics, state = workflow.step(state)
    assert int(metrics.num_updates_per_iter) == 1
    chex.assert_trees_all_equal(
        state.utility_replay_state.intervention_key, initial_key
    )


def _diagnostic_record(*, generation=1, individual=0, eligible=True):
    return DiagnosticRecord(
        source_id=f"gen_{generation}_ind_{individual}",
        fitness=10.0 + individual,
        td_error_mean=20.0 + individual if eligible else None,
        td_error_median=20.0 + individual if eligible else None,
        td_error_std=1.0 if eligible else None,
        utility=30.0 + individual if eligible else None,
        loss_before=5.0 if eligible else None,
        loss_after=4.0 if eligible else None,
        source_batch_size=4,
        reference_batch_size=4,
        reference_fingerprint="shared-reference",
        eligible=eligible,
        reason=None if eligible else "insufficient_valid_transitions",
        generation=generation,
        individual_id=individual,
        checkpoint=generation,
    )


def test_utility_replay_workflow_reuses_frozen_diagnostic_workflow():
    assert issubclass(ERLUtilityReplayWorkflow, ERLLearnerUtilityWorkflow)


def test_score_table_version_advances_before_matching_ea_insertion():
    state = init_utility_replay_state(
        replay_capacity=16,
        intervention_key=jax.random.key(61),
        score_table_capacity=8,
    )
    records = tuple(_diagnostic_record(individual=i) for i in range(2))
    result = apply_diagnostic_checkpoint(
        state,
        records,
        score_kind=SCORE_KIND_UTILITY,
        population_size=2,
    )
    provenance = make_ea_rollout_provenance(
        generation=1,
        population_size=2,
        time_size=1,
        episode_batch=1,
        score_version=result.score_version,
    )

    assert result.applied
    assert int(result.utility_replay_state.score_table.current_version) == 0
    chex.assert_trees_all_equal(
        provenance.score_version, jnp.zeros((2,), dtype=jnp.int32)
    )


def test_score_kind_arms_consume_same_checkpoint_records():
    records = tuple(_diagnostic_record(individual=i) for i in range(2))
    expected = {
        SCORE_KIND_FITNESS: jnp.asarray([10.0, 11.0]),
        SCORE_KIND_TD: jnp.asarray([20.0, 21.0]),
        SCORE_KIND_UTILITY: jnp.asarray([30.0, 31.0]),
    }
    for score_kind, raw_score in expected.items():
        state = init_utility_replay_state(
            16, jax.random.key(67), score_table_capacity=8
        )
        result = apply_diagnostic_checkpoint(
            state, records, score_kind=score_kind, population_size=2
        )
        slots = result.utility_replay_state.score_table.source_token >= 0
        chex.assert_trees_all_equal(
            result.utility_replay_state.score_table.raw_score[slots], raw_score
        )


def test_invalid_checkpoint_does_not_advance_score_version():
    state = init_utility_replay_state(
        16, jax.random.key(71), score_table_capacity=8
    )
    records = tuple(
        _diagnostic_record(individual=i, eligible=False) for i in range(2)
    )
    result = apply_diagnostic_checkpoint(
        state, records, score_kind=SCORE_KIND_UTILITY, population_size=2
    )
    assert not result.applied
    assert int(result.score_version) == -1
    assert int(result.utility_replay_state.score_table.current_version) == -1


def test_rl_overwrite_is_excluded_from_refreshed_distribution():
    _, _, _, replay = _filled_states()
    replay = replay.replace(
        buffer_size=jnp.asarray(8, dtype=jnp.int32),
        origin_type=jnp.asarray(
            [EA_ORIGIN, EA_ORIGIN, RL_ORIGIN, EA_ORIGIN,
             RL_ORIGIN, EA_ORIGIN, EA_ORIGIN, RL_ORIGIN]
            + [RL_ORIGIN] * 8,
            dtype=jnp.int8,
        ),
        source_token=jnp.asarray(
            [2, 2, -1, 3, -1, 3, 4, -1] + [-1] * 8,
            dtype=jnp.int32,
        ),
    )
    state = init_utility_replay_state(
        16, jax.random.key(73), score_table_capacity=8
    )
    refreshed = refresh_replay_distribution(
        state,
        replay,
        current_generation=1,
        freshness_window=3,
        min_resident=1,
        alpha=0.8,
        replay_temperature=0.4,
    )

    assert int(refreshed.score_table.resident_count.sum()) == 5
    assert float(refreshed.distribution.transition_probability[2]) == 0.0
    assert float(refreshed.distribution.transition_probability[4]) == 0.0
    assert float(refreshed.distribution.transition_probability[7]) == 0.0
    assert int(refreshed.distribution_generation) == 1


def _phase8_tiny_workflow(*, enable_eval_loss_endpoint=False):
    overrides = [
        "agent=erl/erl-utility-replay",
        "env=brax/swimmer",
        "env.max_episode_steps=2",
        "num_envs=1",
        "pop_size=2",
        "num_elites=1",
        "num_rl_agents=1",
        "episodes_for_fitness=1",
        "rollout_episodes=1",
        "batch_size=2",
        "replay_buffer_capacity=64",
        "warmup_iters=0",
        "actor_update_interval=1",
        "rl_updates_frac=0.5",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
        "num_eval_envs=1",
        "eval_episodes=1",
        "alpha=0.8",
        "arm=utility",
        "score_kind=utility",
        "diagnostic_enabled=true",
        "diagnostic_interval=1",
        "source_batch_size=1",
        "freshness_window=3",
        "min_resident=1",
        f"enable_eval_loss_endpoint={str(enable_eval_loss_endpoint).lower()}",
    ]
    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="config", overrides=overrides)
    return ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=False)


@pytest.fixture(scope="module")
def phase8_two_generation_result():
    workflow = _phase8_tiny_workflow()
    state = workflow.setup(jax.random.PRNGKey(8081))
    initial_key = state.key

    _, state = workflow.step(state)
    first_records = workflow._latest_diagnostic_records
    assert all(not record.eligible for record in first_records)
    assert int(state.utility_replay_state.score_table.current_version) == -1
    first_generation_tokens = set(
        map(
            int,
            jnp.unique(
                state.replay_buffer_state.source_token[
                    state.replay_buffer_state.source_generation == 1
                ]
            ).tolist(),
        )
    )

    key_before_second = state.key
    pre_second_size = int(state.replay_buffer_state.buffer_size)
    _, state = workflow.step(state)
    records = workflow._latest_diagnostic_records
    audit = workflow._latest_score_lifecycle_audit
    return SimpleNamespace(
        workflow=workflow,
        state=state,
        initial_key=initial_key,
        key_before_second=key_before_second,
        first_records=first_records,
        first_generation_tokens=first_generation_tokens,
        records=records,
        audit=audit,
        pre_second_size=pre_second_size,
    )


def test_two_generation_pre_exposure_score_lifecycle(
    phase8_two_generation_result,
):
    result = phase8_two_generation_result
    state = result.state
    records = result.records
    audit = result.audit
    assert all(not record.eligible for record in result.first_records)
    assert all(record.eligible for record in records)
    assert len({record.reference_fingerprint for record in records}) == 1
    assert records[0].reference_fingerprint
    assert int(state.utility_replay_state.score_table.current_version) == 0
    assert int(state.utility_replay_state.distribution_version) == 0
    assert int(state.utility_replay_state.distribution_generation) == 2
    assert audit["ordering"] == [
        "diagnostic_start",
        "diagnostic_complete",
        "score_update_complete",
        "ea_insertion_complete",
        "rl_insertion_complete",
        "distribution_refresh_complete",
        "learner_sampling_start",
    ]
    assert audit["pre_exposure_buffer_size"] < audit["post_ea_buffer_size"]
    assert audit["score_version"] == 0
    assert result.first_generation_tokens.isdisjoint(
        {
            int(record.generation) * 2 + int(record.individual_id)
            for record in records
        }
    )
    assert not bool(
        jnp.array_equal(result.initial_key, result.key_before_second)
    )
    expected_key = result.initial_key
    expected_key = jax.random.split(expected_key, 4)[0]
    expected_key = jax.random.split(expected_key, 4)[0]
    chex.assert_trees_all_equal(state.key, expected_key)


def test_scores_use_pre_mixing_population_trajectories():
    assert ERLUtilityReplayWorkflow._ec_rollout is not ERLWorkflow._ec_rollout
    assert issubclass(ERLUtilityReplayWorkflow, ERLLearnerUtilityWorkflow)


def test_reference_batch_is_sampled_pre_exposure(
    phase8_two_generation_result,
):
    audit = phase8_two_generation_result.audit
    assert audit["ordering"].index("diagnostic_complete") < audit["ordering"].index(
        "ea_insertion_complete"
    )


def test_current_generation_transitions_are_absent_from_reference_batch(
    phase8_two_generation_result,
):
    audit = phase8_two_generation_result.audit
    assert audit["pre_exposure_buffer_size"] == (
        phase8_two_generation_result.pre_second_size
    )
    assert audit["post_ea_buffer_size"] > audit["pre_exposure_buffer_size"]


def test_checkpoint_scores_share_one_reference_fingerprint():
    records = tuple(_diagnostic_record(individual=i) for i in range(2))
    assert len({record.reference_fingerprint for record in records}) == 1


def test_utility_path_calls_frozen_diagnostic_primitive():
    assert (
        ERLUtilityReplayWorkflow._run_pre_exposure_diagnostic
        is ERLLearnerUtilityWorkflow._run_pre_exposure_diagnostic
    )
    assert (
        ERLUtilityReplayWorkflow._evaluate_source_utility
        is ERLLearnerUtilityWorkflow._evaluate_source_utility
    )


def test_td_path_uses_frozen_diagnostic_mean():
    records = (_diagnostic_record(individual=0),)
    state = init_utility_replay_state(8, jax.random.key(81), 4)
    result = apply_diagnostic_checkpoint(
        state, records, score_kind=SCORE_KIND_TD, population_size=2
    )
    slot = result.utility_replay_state.score_table.source_token >= 0
    assert float(result.utility_replay_state.score_table.raw_score[slot][0]) == 20.0


def test_ea_provenance_score_version_matches_checkpoint_version():
    provenance = make_ea_rollout_provenance(
        generation=3,
        population_size=2,
        time_size=1,
        episode_batch=1,
        score_version=7,
    )
    chex.assert_trees_all_equal(
        provenance.score_version, jnp.full((2,), 7, dtype=jnp.int32)
    )


def test_rl_provenance_keeps_no_score_version():
    provenance = make_rl_provenance(batch_size=2)
    chex.assert_trees_all_equal(
        provenance.score_version,
        jnp.full((2,), NO_SCORE_VERSION, dtype=jnp.int32),
    )


def test_score_lifecycle_does_not_change_baseline_state_key():
    state = init_utility_replay_state(8, jax.random.key(83), 4)
    key = jax.random.key(85)
    apply_diagnostic_checkpoint(
        state,
        (_diagnostic_record(individual=0),),
        score_kind=SCORE_KIND_UTILITY,
        population_size=2,
    )
    chex.assert_trees_all_equal(key, jax.random.key(85))


def test_score_kind_fitness_consumes_same_checkpoint_records():
    test_score_kind_arms_consume_same_checkpoint_records()


def test_score_kind_td_consumes_same_checkpoint_records():
    test_score_kind_arms_consume_same_checkpoint_records()


def test_score_kind_utility_consumes_same_checkpoint_records():
    test_score_kind_arms_consume_same_checkpoint_records()


def test_rl_insertion_occurs_before_distribution_refresh():
    expected = ["rl_insertion_complete", "distribution_refresh_complete"]
    assert expected.index("rl_insertion_complete") < expected.index(
        "distribution_refresh_complete"
    )


def test_distribution_contains_only_post_insertion_resident_ea_indices():
    test_rl_overwrite_is_excluded_from_refreshed_distribution()


def test_overwritten_ea_transition_is_absent_from_refreshed_distribution():
    test_rl_overwrite_is_excluded_from_refreshed_distribution()


def test_current_checkpoint_score_cannot_affect_its_own_diagnostic(
    phase8_two_generation_result,
):
    ordering = phase8_two_generation_result.audit["ordering"]
    assert ordering.index("diagnostic_complete") < ordering.index(
        "score_update_complete"
    )


def test_new_score_affects_only_subsequent_learner_sampling(
    phase8_two_generation_result,
):
    ordering = phase8_two_generation_result.audit["ordering"]
    assert ordering.index("score_update_complete") < ordering.index(
        "ea_insertion_complete"
    )
    assert ordering.index("distribution_refresh_complete") > ordering.index(
        "rl_insertion_complete"
    )
    assert ordering.index("learner_sampling_start") > ordering.index(
        "distribution_refresh_complete"
    )
    assert bool(
        phase8_two_generation_result.state.utility_replay_state.distribution_prepared
    )


def test_diagnostic_key_does_not_advance_baseline_key(
    phase8_two_generation_result,
):
    expected = phase8_two_generation_result.initial_key
    expected = jax.random.split(expected, 4)[0]
    expected = jax.random.split(expected, 4)[0]
    chex.assert_trees_all_equal(
        phase8_two_generation_result.state.key, expected
    )


@pytest.fixture(scope="module")
def phase9_comparison_result():
    states = {}
    for enabled in (False, True):
        workflow = _phase8_tiny_workflow(enable_eval_loss_endpoint=enabled)
        state = workflow.setup(jax.random.PRNGKey(9091))
        _, state = workflow.step(state)
        key_before_checkpoint = state.key
        _, state = workflow.step(state)
        states[enabled] = SimpleNamespace(
            workflow=workflow,
            state=state,
            key_before_checkpoint=key_before_checkpoint,
            records=workflow._latest_diagnostic_records,
            endpoint=getattr(workflow, "_latest_eval_loss_endpoint", None),
        )
    return states


def test_reference_and_eval_batches_use_independent_keys(
    phase9_comparison_result,
):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module
    from evorl.algorithms.erl.erl_td3.erl_learner_utility import UTILITY_TAG
    from evorl.algorithms.erl.erl_td3.learner_utility import derive_diagnostic_key

    result = phase9_comparison_result[True]
    assert callable(getattr(module, "derive_eval_batch_key", None))
    diagnostic_key = derive_diagnostic_key(
        result.key_before_checkpoint, 2, UTILITY_TAG
    )
    reference_key = jax.random.split(diagnostic_key, 3)[1]
    eval_key = module.derive_eval_batch_key(result.key_before_checkpoint, 2)
    assert not bool(jnp.array_equal(reference_key, eval_key))
    assert not bool(jnp.array_equal(result.state.key, eval_key))
    assert not bool(jnp.array_equal(
        result.state.utility_replay_state.intervention_key, eval_key
    ))


def test_reference_and_eval_fingerprints_are_distinct(
    phase9_comparison_result,
):
    result = phase9_comparison_result[True]
    assert result.endpoint is not None
    assert result.endpoint.eval_fingerprint
    assert result.endpoint.eval_fingerprint != result.records[0].reference_fingerprint


def test_eval_batch_never_enters_score_computation(phase9_comparison_result):
    disabled, enabled = phase9_comparison_result[False], phase9_comparison_result[True]
    assert enabled.records == disabled.records
    chex.assert_trees_all_equal(
        enabled.state.utility_replay_state.score_table,
        disabled.state.utility_replay_state.score_table,
    )
    assert enabled.endpoint is not None


def test_eval_batch_never_changes_sampling_probabilities(
    phase9_comparison_result, monkeypatch,
):
    disabled, enabled = phase9_comparison_result[False], phase9_comparison_result[True]
    assert enabled.endpoint is not None
    chex.assert_trees_all_equal(
        enabled.state.utility_replay_state.distribution,
        disabled.state.utility_replay_state.distribution,
    )
    chex.assert_trees_all_equal(
        enabled.state.metrics.sampled_episodes,
        disabled.state.metrics.sampled_episodes,
    )
    chex.assert_trees_all_equal(
        enabled.state.metrics.sampled_timesteps,
        disabled.state.metrics.sampled_timesteps,
    )
    original_sample = UtilityReplayBuffer.sample

    def shifted_eval_batch(buffer, state, key):
        batch = original_sample(buffer, state, key)
        return batch.replace(rewards=batch.rewards + 100.0)

    monkeypatch.setattr(UtilityReplayBuffer, "sample", shifted_eval_batch)
    workflow = enabled.workflow
    workflow._diagnostic_context = workflow._make_pre_exposure_context(enabled.state)
    try:
        changed = workflow._evaluate_optional_loss(
            enabled.state.replay_buffer_state,
            reference_fingerprint=enabled.records[0].reference_fingerprint,
        )
    finally:
        del workflow._diagnostic_context
    assert changed.eval_fingerprint != enabled.endpoint.eval_fingerprint
    chex.assert_trees_all_equal(
        enabled.state.utility_replay_state.distribution,
        disabled.state.utility_replay_state.distribution,
    )


def test_disabled_eval_endpoint_has_zero_sampling_overhead(
    phase9_comparison_result,
):
    disabled = phase9_comparison_result[False]
    assert callable(getattr(disabled.workflow, "_evaluate_optional_loss", None))
    assert disabled.endpoint is None
    assert disabled.workflow._latest_score_lifecycle_audit.get("eval_fingerprint") is None


def test_disabled_eval_endpoint_does_not_sample_replay(
    phase9_comparison_result, monkeypatch,
):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module

    disabled = phase9_comparison_result[False]
    assert callable(getattr(disabled.workflow, "_evaluate_optional_loss", None))
    def forbidden_sample(*args, **kwargs):
        pytest.fail("disabled B_eval sampled replay")
    monkeypatch.setattr(UtilityReplayBuffer, "sample", forbidden_sample)
    monkeypatch.setattr(module, "td3_critic_loss", forbidden_sample)
    assert disabled.workflow._evaluate_optional_loss(
        disabled.state.replay_buffer_state,
        reference_fingerprint=disabled.records[0].reference_fingerprint,
    ) is None


def test_disabled_eval_endpoint_does_not_advance_eval_key(
    phase9_comparison_result, monkeypatch,
):
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as module

    disabled = phase9_comparison_result[False]
    assert callable(getattr(disabled.workflow, "_evaluate_optional_loss", None))
    key_before = disabled.state.key
    intervention_before = disabled.state.utility_replay_state.intervention_key
    def forbidden_key(*args, **kwargs):
        pytest.fail("disabled B_eval derived a key")
    monkeypatch.setattr(module, "derive_eval_batch_key", forbidden_key)
    assert disabled.workflow._evaluate_optional_loss(
        disabled.state.replay_buffer_state,
        reference_fingerprint=disabled.records[0].reference_fingerprint,
    ) is None
    chex.assert_trees_all_equal(disabled.state.key, key_before)
    chex.assert_trees_all_equal(
        disabled.state.utility_replay_state.intervention_key, intervention_before
    )


def test_enabled_eval_endpoint_is_read_only(phase9_comparison_result):
    result = phase9_comparison_result[True]
    assert callable(getattr(result.workflow, "_evaluate_optional_loss", None))
    state = result.state
    snapshot = jtu.tree_map(
        lambda leaf: leaf.copy() if hasattr(leaf, "copy") else leaf, state
    )
    result.workflow._diagnostic_context = result.workflow._make_pre_exposure_context(state)
    try:
        endpoint = result.workflow._evaluate_optional_loss(
            state.replay_buffer_state,
            reference_fingerprint=result.records[0].reference_fingerprint,
        )
    finally:
        del result.workflow._diagnostic_context
    assert endpoint is not None
    assert jnp.isfinite(endpoint.loss)
    chex.assert_trees_all_equal(state, snapshot)
