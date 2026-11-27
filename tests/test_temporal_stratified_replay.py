import chex
import jax
import jax.numpy as jnp
import numpy as np

from evorl.algorithms.erl.erl_td3.erl_utility_replay import weighted_replay_sample
from evorl.algorithms.erl.erl_td3.temporal_replay import (
    MODE_RECENT_30,
    MODE_STRATIFIED_5_25,
    build_temporal_ea_distribution,
)
from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    RL_ORIGIN,
    SourceAwareReplayBufferState,
    UNSET_ORIGIN,
)


def _replay(origin, generation, *, size=None):
    origin = jnp.asarray(origin, dtype=jnp.int8)
    generation = jnp.asarray(generation, dtype=jnp.int32)
    capacity = len(origin)
    return SourceAwareReplayBufferState(
        data=jnp.arange(capacity, dtype=jnp.int32),
        current_index=jnp.asarray(0, dtype=jnp.int32),
        buffer_size=jnp.asarray(capacity if size is None else size, dtype=jnp.int32),
        origin_type=origin,
        source_token=jnp.arange(capacity, dtype=jnp.int32),
        source_generation=generation,
        score_version=jnp.full((capacity,), -1, dtype=jnp.int32),
    )


def test_recent30_is_uniform_only_over_valid_window():
    replay = _replay(
        [EA_ORIGIN, EA_ORIGIN, EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, UNSET_ORIGIN],
        [40, 11, 10, -1, 39, 40],
    )
    result = build_temporal_ea_distribution(
        replay, current_generation=40, mode=MODE_RECENT_30
    )

    probability = np.asarray(result.distribution.transition_probability)
    np.testing.assert_allclose(probability, [1 / 3, 1 / 3, 0, 0, 1 / 3, 0])
    assert bool(result.prepared)
    assert int(result.window_count) == 3


def test_stratified_5_25_assigns_half_mass_to_each_stratum():
    replay = _replay(
        [EA_ORIGIN] * 8 + [RL_ORIGIN],
        [40, 39, 38, 37, 35, 34, 20, 10, -1],
    )
    result = build_temporal_ea_distribution(
        replay, current_generation=40, mode=MODE_STRATIFIED_5_25
    )
    probability = np.asarray(result.distribution.transition_probability)

    np.testing.assert_allclose(probability[:4].sum(), 0.5, rtol=0, atol=1e-7)
    np.testing.assert_allclose(probability[4:7].sum(), 0.5, rtol=0, atol=1e-7)
    assert probability[7] == 0
    assert probability[8] == 0
    assert int(result.recent_count) == 4
    assert int(result.history_count) == 3


def test_incomplete_30_generation_window_preserves_baseline_path():
    replay = _replay([EA_ORIGIN, EA_ORIGIN, RL_ORIGIN], [12, 1, -1])
    for mode in (MODE_RECENT_30, MODE_STRATIFIED_5_25):
        result = build_temporal_ea_distribution(
            replay, current_generation=12, mode=mode
        )
        assert not bool(result.prepared)
        chex.assert_trees_all_equal(
            result.distribution.transition_probability, jnp.zeros((3,), jnp.float32)
        )


def test_empty_required_stratum_preserves_baseline_path():
    replay = _replay([EA_ORIGIN, EA_ORIGIN, RL_ORIGIN], [20, 19, -1])
    result = build_temporal_ea_distribution(
        replay, current_generation=40, mode=MODE_STRATIFIED_5_25
    )
    assert not bool(result.prepared)


def test_ring_invalid_slots_never_receive_probability():
    replay = _replay(
        [EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, EA_ORIGIN, EA_ORIGIN, EA_ORIGIN],
        [40, -1, 35, 39, 38, 37],
        size=3,
    )
    result = build_temporal_ea_distribution(
        replay, current_generation=40, mode=MODE_RECENT_30
    )
    probability = np.asarray(result.distribution.transition_probability)
    np.testing.assert_allclose(probability, [0.5, 0, 0.5, 0, 0, 0])
    assert float(probability.sum()) == 1.0


def test_overlay_changes_ea_indices_and_preserves_each_rl_index():
    replay = _replay(
        [EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, EA_ORIGIN],
        [40, -1, 10, -1, 39, 38],
    )
    result = build_temporal_ea_distribution(
        replay, current_generation=40, mode=MODE_RECENT_30
    )
    sampled = weighted_replay_sample(
        replay,
        jax.random.key(17),
        jax.random.key(23),
        result.distribution,
        result.prepared,
        sample_batch_size=256,
    )
    baseline = np.asarray(sampled.baseline_indices)
    final = np.asarray(sampled.final_indices)
    rl = np.asarray(sampled.rl_slot_mask)
    ea = np.asarray(sampled.ea_slot_mask)

    np.testing.assert_array_equal(final[rl], baseline[rl])
    assert np.any(final[ea] != baseline[ea])
    assert np.all(np.isin(final[ea], [0, 4, 5]))
