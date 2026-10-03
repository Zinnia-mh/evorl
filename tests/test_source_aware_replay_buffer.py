import chex
import jax
import jax.numpy as jnp
import pytest

from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    RL_ORIGIN,
    RL_SCORE_VERSION,
    RL_SOURCE_GENERATION,
    RL_SOURCE_TOKEN,
    ReplayProvenance,
    SourceAwareReplayBuffer,
    encode_source_token,
    make_ea_provenance,
    make_rl_provenance,
    sidecar_nbytes,
)


def _buffer(capacity=8):
    replay = SourceAwareReplayBuffer(
        capacity=capacity,
        min_sample_timesteps=1,
        sample_batch_size=2,
    )
    return replay, replay.init(jnp.zeros((), dtype=jnp.int32))


def _provenance(origin, token, generation, version):
    return ReplayProvenance(
        origin_type=jnp.asarray(origin, dtype=jnp.int8),
        source_token=jnp.asarray(token, dtype=jnp.int32),
        source_generation=jnp.asarray(generation, dtype=jnp.int32),
        score_version=jnp.asarray(version, dtype=jnp.int32),
    )


def test_masked_add_keeps_payload_and_provenance_aligned():
    replay, state = _buffer(capacity=6)
    payload = jnp.asarray([10, 20, 30, 40], dtype=jnp.int32)
    metadata = _provenance(
        [EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, RL_ORIGIN],
        [101, RL_SOURCE_TOKEN, 303, RL_SOURCE_TOKEN],
        [1, RL_SOURCE_GENERATION, 3, RL_SOURCE_GENERATION],
        [11, RL_SCORE_VERSION, 33, RL_SCORE_VERSION],
    )
    mask = jnp.asarray([True, False, True, False])

    result = jax.jit(replay.add)(state, payload, metadata, mask)

    assert int(result.current_index) == 2
    assert int(result.buffer_size) == 2
    chex.assert_trees_all_equal(result.data[:2], jnp.asarray([10, 30]))
    chex.assert_trees_all_equal(
        result.origin_type[:2], jnp.asarray([EA_ORIGIN, EA_ORIGIN], dtype=jnp.int8)
    )
    chex.assert_trees_all_equal(
        result.source_token[:2], jnp.asarray([101, 303], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        result.source_generation[:2], jnp.asarray([1, 3], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        result.score_version[:2], jnp.asarray([11, 33], dtype=jnp.int32)
    )


def test_wraparound_overwrite_keeps_payload_and_provenance_aligned():
    replay, state = _buffer(capacity=4)
    first = _provenance(
        [EA_ORIGIN] * 3,
        [2] * 3,
        [1] * 3,
        [10] * 3,
    )
    state = replay.add(
        state, jnp.asarray([1, 2, 3], dtype=jnp.int32), first
    )
    second = _provenance(
        [EA_ORIGIN] * 3,
        [5] * 3,
        [2] * 3,
        [20] * 3,
    )

    result = replay.add(
        state, jnp.asarray([10, 11, 12], dtype=jnp.int32), second
    )

    assert int(result.current_index) == 2
    assert int(result.buffer_size) == 4
    chex.assert_trees_all_equal(result.data, jnp.asarray([11, 12, 3, 10]))
    chex.assert_trees_all_equal(
        result.source_token, jnp.asarray([5, 5, 2, 5], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        result.source_generation, jnp.asarray([2, 2, 1, 2], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        result.score_version, jnp.asarray([20, 20, 10, 20], dtype=jnp.int32)
    )


def test_ea_add_encodes_generation_and_individual_token():
    metadata = make_ea_provenance(
        generation=7,
        individual_id=5,
        population_size=32,
        score_version=4,
        batch_size=3,
    )

    assert encode_source_token(7, 5, 32) == 229
    chex.assert_trees_all_equal(
        metadata.origin_type, jnp.full((3,), EA_ORIGIN, dtype=jnp.int8)
    )
    chex.assert_trees_all_equal(
        metadata.source_token, jnp.full((3,), 229, dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        metadata.source_generation, jnp.full((3,), 7, dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        metadata.score_version, jnp.full((3,), 4, dtype=jnp.int32)
    )


def test_rl_add_uses_reserved_source_sentinel():
    metadata = make_rl_provenance(batch_size=3)

    chex.assert_trees_all_equal(
        metadata.origin_type, jnp.full((3,), RL_ORIGIN, dtype=jnp.int8)
    )
    chex.assert_trees_all_equal(
        metadata.source_token,
        jnp.full((3,), RL_SOURCE_TOKEN, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        metadata.source_generation,
        jnp.full((3,), RL_SOURCE_GENERATION, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        metadata.score_version,
        jnp.full((3,), RL_SCORE_VERSION, dtype=jnp.int32),
    )


def test_source_token_int32_overflow_is_rejected():
    max_int32 = jnp.iinfo(jnp.int32).max
    assert encode_source_token(max_int32 // 2, 1, 2) == max_int32

    with pytest.raises(ValueError, match="signed int32"):
        encode_source_token(max_int32 // 2 + 1, 0, 2)

    with pytest.raises(ValueError, match="individual_id"):
        encode_source_token(0, 2, 2)


def test_sidecar_memory_estimate_stays_within_registered_limit():
    capacity = 1_000_000
    dtypes = {
        "origin_type": jnp.dtype(jnp.int8),
        "source_token": jnp.dtype(jnp.int32),
        "source_generation": jnp.dtype(jnp.int32),
        "score_version": jnp.dtype(jnp.int32),
    }
    assert {name: dtype.itemsize for name, dtype in dtypes.items()} == {
        "origin_type": 1,
        "source_token": 4,
        "source_generation": 4,
        "score_version": 4,
    }
    assert sidecar_nbytes(capacity) == 13_000_000
    assert sidecar_nbytes(capacity) / 1024**2 == pytest.approx(12.39776611328125)
    assert sidecar_nbytes(capacity) <= 24 * 1024**2


def test_jitted_add_matches_eager_add():
    replay, state = _buffer(capacity=5)
    payload = jnp.asarray([7, 8, 9], dtype=jnp.int32)
    metadata = make_ea_provenance(
        generation=3,
        individual_id=1,
        population_size=4,
        score_version=2,
        batch_size=3,
    )
    mask = jnp.asarray([True, False, True])

    eager = replay.add(state, payload, metadata, mask)
    jitted = jax.jit(replay.add)(state, payload, metadata, mask)

    chex.assert_trees_all_equal(eager, jitted)
