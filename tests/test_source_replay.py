from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp

from evorl.algorithms.erl.erl_td3.source_replay import (
    build_origin_slot_masks,
    draw_baseline_indices,
    overlay_ea_indices,
)
from evorl.replay_buffers.source_aware_replay_buffer import EA_ORIGIN, RL_ORIGIN


def _fixture():
    origin_type = jnp.asarray(
        [EA_ORIGIN, RL_ORIGIN, EA_ORIGIN, EA_ORIGIN, RL_ORIGIN, RL_ORIGIN],
        dtype=jnp.int8,
    )
    baseline_indices = jnp.asarray([1, 0, 4, 2, 5, 3, 1], dtype=jnp.int32)
    return origin_type, baseline_indices


def test_overlay_skeleton_preserves_all_rl_indices():
    origin_type, baseline_indices = _fixture()

    result = overlay_ea_indices(baseline_indices, origin_type)

    expected_rl_mask = jnp.asarray([True, False, True, False, True, False, True])
    chex.assert_trees_all_equal(result.rl_slot_mask, expected_rl_mask)
    chex.assert_trees_all_equal(
        result.final_indices[result.rl_slot_mask],
        baseline_indices[result.rl_slot_mask],
    )


def test_overlay_skeleton_preserves_ea_slot_positions():
    origin_type, baseline_indices = _fixture()

    result = overlay_ea_indices(baseline_indices, origin_type)

    expected_ea_mask = jnp.asarray([False, True, False, True, False, True, False])
    chex.assert_trees_all_equal(result.ea_slot_mask, expected_ea_mask)
    chex.assert_trees_all_equal(
        jnp.flatnonzero(result.ea_slot_mask), jnp.asarray([1, 3, 5])
    )
    chex.assert_trees_all_equal(
        result.final_indices[result.ea_slot_mask],
        baseline_indices[result.ea_slot_mask],
    )


def test_overlay_skeleton_preserves_realized_ea_rl_counts():
    origin_type, baseline_indices = _fixture()
    before_ea, before_rl = build_origin_slot_masks(origin_type, baseline_indices)

    result = overlay_ea_indices(baseline_indices, origin_type)
    after_ea, after_rl = build_origin_slot_masks(origin_type, result.final_indices)

    assert int(before_ea.sum()) == int(after_ea.sum()) == 3
    assert int(before_rl.sum()) == int(after_rl.sum()) == 4
    chex.assert_trees_all_equal(before_ea, after_ea)
    chex.assert_trees_all_equal(before_rl, after_rl)


def test_overlay_skeleton_returns_baseline_indices_before_weighting():
    origin_type, baseline_indices = _fixture()

    result = overlay_ea_indices(baseline_indices, origin_type)

    chex.assert_trees_all_equal(result.baseline_indices, baseline_indices)
    chex.assert_trees_all_equal(result.final_indices, baseline_indices)


def test_pilot_baseline_draw_matches_original_randint_exactly():
    replay_state = SimpleNamespace(buffer_size=jnp.asarray(6, dtype=jnp.int32))
    key = jax.random.key(2026)
    batch_size = 17
    expected = jax.random.randint(
        key, (batch_size,), minval=0, maxval=replay_state.buffer_size
    )

    actual = draw_baseline_indices(replay_state, key, batch_size)

    chex.assert_trees_all_equal(actual, expected)


def test_overlay_skeleton_eager_and_jit_are_exactly_equal():
    origin_type, baseline_indices = _fixture()

    eager = overlay_ea_indices(baseline_indices, origin_type)
    jitted = jax.jit(overlay_ea_indices)(baseline_indices, origin_type)

    chex.assert_trees_all_equal(eager, jitted)
