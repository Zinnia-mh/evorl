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
    init_utility_replay_state,
    make_ea_rollout_provenance,
    weighted_replay_draws,
    weighted_replay_sample,
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
    config = SimpleNamespace(alpha=0.5, actor_update_interval=2, num_rl_agents=1)
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
            overrides=["agent=erl/erl-utility-replay", "alpha=0.8", *overrides],
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
    config = SimpleNamespace(alpha=0.8, actor_update_interval=2, num_rl_agents=1)
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
            overrides=["agent=erl/erl-utility-replay", "alpha=0.8", *overrides],
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
    table = init_source_score_table(len(tokens))._replace(
        source_token=jnp.asarray(tokens, dtype=jnp.int32),
        rank_normalized_score=jnp.linspace(1.0, 0.0, len(tokens)),
        resident_count=counts,
        weighted_eligible=jnp.ones((len(tokens),), dtype=jnp.bool_),
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
            distribution=distribution,
            distribution_prepared=jnp.asarray(True),
            distribution_version=jnp.asarray(1, dtype=jnp.int32),
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
                "agent=erl/erl-utility-replay", "alpha=0.8", *overrides
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
