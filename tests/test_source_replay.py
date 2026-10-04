from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp

from evorl.algorithms.erl.erl_td3.source_replay import (
    SCORE_KIND_FITNESS,
    SCORE_KIND_NONE,
    SCORE_KIND_TD,
    SCORE_KIND_UTILITY,
    SCORE_TABLE_ERROR_CAPACITY,
    SCORE_TABLE_ERROR_NEGATIVE_COUNT,
    SCORE_TABLE_ERROR_VERSION,
    SCORE_TABLE_OK,
    apply_checkpoint_score_update,
    build_checkpoint_score_update,
    build_origin_slot_masks,
    draw_baseline_indices,
    init_source_score_table,
    overlay_ea_indices,
    raise_for_score_table_status,
    rank_normalize_within_checkpoint,
    source_score_table_nbytes,
    update_resident_counts,
)
from evorl.algorithms.erl.erl_td3.erl_learner_utility import DiagnosticRecord
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


def _record(
    generation,
    individual_id,
    *,
    fitness=0.0,
    td=0.0,
    utility=0.0,
    eligible=True,
    checkpoint=None,
):
    return DiagnosticRecord(
        source_id=f"gen_{generation}_ind_{individual_id}",
        fitness=fitness,
        td_error_mean=td if eligible else None,
        td_error_median=(td + 100.0) if eligible else None,
        td_error_std=1.0 if eligible else None,
        utility=utility if eligible else None,
        loss_before=2.0 if eligible else None,
        loss_after=(2.0 - utility) if eligible else None,
        source_batch_size=4,
        reference_batch_size=4,
        reference_fingerprint="ref",
        eligible=eligible,
        reason=None if eligible else "insufficient_valid_transitions",
        generation=generation,
        individual_id=individual_id,
        checkpoint=generation if checkpoint is None else checkpoint,
    )


def _table_with_sources(tokens, counts):
    table = init_source_score_table(len(tokens))
    return table._replace(
        source_token=jnp.asarray(tokens, dtype=jnp.int32),
        resident_count=jnp.asarray(counts, dtype=jnp.int32),
    )


def test_score_table_stores_required_fields():
    table = init_source_score_table(5)
    expected = {
        "source_token",
        "raw_score",
        "rank_normalized_score",
        "measured_generation",
        "score_kind",
        "validity",
        "resident_count",
        "weighted_eligible",
        "version",
    }
    assert expected.issubset(table._fields)
    for field in expected:
        assert getattr(table, field).shape == (5,)
    assert table.source_token.dtype == jnp.int32
    assert table.raw_score.dtype == jnp.float32
    assert table.validity.dtype == jnp.bool_

    update = build_checkpoint_score_update(
        [_record(4, 1, fitness=7.0)],
        score_kind=SCORE_KIND_FITNESS,
        population_size=4,
        version=0,
    )
    stored = apply_checkpoint_score_update(table, update).table
    slot = int(jnp.argmax(stored.source_token == 17))
    assert float(stored.raw_score[slot]) == 7.0
    assert int(stored.measured_generation[slot]) == 4
    assert int(stored.score_kind[slot]) == SCORE_KIND_FITNESS
    assert bool(stored.validity[slot])
    assert not bool(stored.weighted_eligible[slot])
    assert int(stored.version[slot]) == 0


def test_rank_normalization_is_within_checkpoint():
    records = [_record(10, 0), _record(11, 1)]
    try:
        build_checkpoint_score_update(
            records,
            score_kind=SCORE_KIND_FITNESS,
            population_size=4,
            version=0,
        )
    except ValueError as exc:
        assert "checkpoint" in str(exc).lower()
    else:
        raise AssertionError("cross-checkpoint ranking must fail explicitly")


def test_rank_normalization_uses_deterministic_average_ties():
    scores = jnp.asarray([10.0, 10.0, 5.0], dtype=jnp.float32)
    actual = rank_normalize_within_checkpoint(scores, jnp.ones(3, dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        actual, jnp.asarray([0.75, 0.75, 0.0], dtype=jnp.float32)
    )


def test_rank_normalization_maps_higher_score_to_higher_priority():
    scores = jnp.asarray([-2.0, 7.0, 1.0], dtype=jnp.float32)
    ranks = rank_normalize_within_checkpoint(scores, jnp.ones(3, dtype=jnp.bool_))
    assert float(ranks[1]) > float(ranks[2]) > float(ranks[0])


def test_score_table_supports_fitness_td_and_utility_kinds():
    records = [
        _record(3, 0, fitness=1.0, td=2.0, utility=3.0),
        _record(3, 1, fitness=4.0, td=5.0, utility=6.0),
    ]
    expectations = {
        SCORE_KIND_FITNESS: [1.0, 4.0],
        SCORE_KIND_TD: [2.0, 5.0],
        SCORE_KIND_UTILITY: [3.0, 6.0],
    }
    for kind, raw in expectations.items():
        update = build_checkpoint_score_update(
            records, score_kind=kind, population_size=4, version=0
        )
        chex.assert_trees_all_equal(
            update.raw_score, jnp.asarray(raw, dtype=jnp.float32)
        )
        chex.assert_trees_all_equal(
            update.score_kind, jnp.full((2,), kind, dtype=jnp.int8)
        )


def test_score_table_version_is_monotonic():
    records = [_record(3, 0, fitness=1.0)]
    update = build_checkpoint_score_update(
        records, score_kind=SCORE_KIND_FITNESS, population_size=4, version=0
    )
    first = apply_checkpoint_score_update(init_source_score_table(2), update)
    assert int(first.status) == SCORE_TABLE_OK
    assert int(first.table.current_version) == 0
    repeated = apply_checkpoint_score_update(first.table, update)
    assert int(repeated.status) == SCORE_TABLE_ERROR_VERSION
    try:
        raise_for_score_table_status(repeated.status)
    except RuntimeError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("a repeated score version must fail explicitly")


def test_utility_scores_reuse_frozen_diagnostic_records():
    record = _record(8, 2, fitness=99.0, td=88.0, utility=-0.375)
    update = build_checkpoint_score_update(
        [record], score_kind=SCORE_KIND_UTILITY, population_size=4, version=0
    )
    assert float(update.raw_score[0]) == -0.375
    assert int(update.source_token[0]) == 8 * 4 + 2


def test_td_scores_use_frozen_record_mean_field():
    record = _record(8, 2, td=1.25)
    update = build_checkpoint_score_update(
        [record], score_kind=SCORE_KIND_TD, population_size=4, version=0
    )
    assert float(update.raw_score[0]) == 1.25
    assert float(update.raw_score[0]) != record.td_error_median


def test_ring_overwrite_updates_resident_counts():
    table = _table_with_sources([10, 20, -1, -1], [2, 1, 0, 0])
    resident_tokens = jnp.asarray([10, 10, 20, -1], dtype=jnp.int32)
    result = update_resident_counts(
        table,
        resident_tokens,
        write_indices=jnp.asarray([1, 3], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([20, 30], dtype=jnp.int32),
        mask=jnp.asarray([True, True]),
    )
    assert int(result.status) == SCORE_TABLE_OK
    counts = {
        int(token): int(count)
        for token, count in zip(result.table.source_token, result.table.resident_count)
        if int(token) >= 0
    }
    assert counts == {10: 1, 20: 2, 30: 1}
    chex.assert_trees_all_equal(
        result.resident_source_tokens,
        jnp.asarray([10, 20, 20, 30], dtype=jnp.int32),
    )


def test_resident_count_update_handles_repeated_wraparound_and_mask():
    table = _table_with_sources([10, 20, -1, -1], [1, 1, 0, 0])
    resident_tokens = jnp.asarray([10, 20, -1, -1], dtype=jnp.int32)
    result = update_resident_counts(
        table,
        resident_tokens,
        write_indices=jnp.asarray([3, 0, 0, 1], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([30, 30, 40, 99], dtype=jnp.int32),
        mask=jnp.asarray([True, True, True, False]),
    )
    assert int(result.status) == SCORE_TABLE_OK
    counts = {
        int(token): int(count)
        for token, count in zip(result.table.source_token, result.table.resident_count)
        if int(token) >= 0
    }
    assert counts == {20: 1, 30: 1, 40: 1}
    assert 99 not in counts
    chex.assert_trees_all_equal(
        result.resident_source_tokens,
        jnp.asarray([40, 20, -1, 30], dtype=jnp.int32),
    )


def test_score_slot_reclaimed_only_after_resident_count_zero():
    table = _table_with_sources([10, 20, -1], [1, 0, 0])
    result = update_resident_counts(
        table,
        jnp.asarray([-1], dtype=jnp.int32),
        write_indices=jnp.asarray([0], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([30], dtype=jnp.int32),
        mask=jnp.asarray([True]),
    )
    assert int(result.status) == SCORE_TABLE_OK
    assert int(result.table.source_token[0]) == 10
    assert int(result.table.source_token[1]) == 30


def test_score_slot_reuse_clears_all_stale_metadata():
    table = _table_with_sources([10], [0])._replace(
        raw_score=jnp.asarray([9.0], dtype=jnp.float32),
        rank_normalized_score=jnp.asarray([1.0], dtype=jnp.float32),
        measured_generation=jnp.asarray([7], dtype=jnp.int32),
        score_kind=jnp.asarray([SCORE_KIND_UTILITY], dtype=jnp.int8),
        validity=jnp.asarray([True]),
        weighted_eligible=jnp.asarray([True]),
        version=jnp.asarray([5], dtype=jnp.int32),
    )
    result = update_resident_counts(
        table,
        jnp.asarray([-1], dtype=jnp.int32),
        write_indices=jnp.asarray([0], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([30], dtype=jnp.int32),
        mask=jnp.asarray([True]),
    )
    assert int(result.table.source_token[0]) == 30
    assert float(result.table.raw_score[0]) == 0.0
    assert float(result.table.rank_normalized_score[0]) == 0.0
    assert int(result.table.measured_generation[0]) == -1
    assert int(result.table.score_kind[0]) == SCORE_KIND_NONE
    assert not bool(result.table.validity[0])
    assert not bool(result.table.weighted_eligible[0])
    assert int(result.table.version[0]) == -1


def test_score_slot_allocation_is_deterministic():
    table = _table_with_sources([10, 20, 30, 40], [1, 0, 0, 1])
    result = update_resident_counts(
        table,
        jnp.asarray([-1], dtype=jnp.int32),
        write_indices=jnp.asarray([0], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([50], dtype=jnp.int32),
        mask=jnp.asarray([True]),
    )
    assert int(result.table.source_token[1]) == 50
    assert int(result.table.source_token[2]) == 30


def test_score_table_full_with_all_sources_resident_has_explicit_failure():
    table = _table_with_sources([10, 20], [1, 1])
    result = update_resident_counts(
        table,
        jnp.asarray([-1], dtype=jnp.int32),
        write_indices=jnp.asarray([0], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([30], dtype=jnp.int32),
        mask=jnp.asarray([True]),
    )
    assert int(result.status) == SCORE_TABLE_ERROR_CAPACITY
    try:
        raise_for_score_table_status(result.status)
    except RuntimeError as exc:
        assert "capacity" in str(exc).lower()
    else:
        raise AssertionError("full resident table must fail explicitly")


def test_checkpoint_score_update_fails_when_all_slots_are_resident():
    table = _table_with_sources([10, 20], [1, 1])
    update = build_checkpoint_score_update(
        [_record(8, 0, fitness=1.0)],
        score_kind=SCORE_KIND_FITNESS,
        population_size=4,
        version=0,
    )
    result = apply_checkpoint_score_update(table, update)
    assert int(result.status) == SCORE_TABLE_ERROR_CAPACITY
    chex.assert_trees_all_equal(result.table, table)


def test_score_table_rejects_negative_resident_count():
    table = _table_with_sources([10], [0])
    result = update_resident_counts(
        table,
        jnp.asarray([10], dtype=jnp.int32),
        write_indices=jnp.asarray([0], dtype=jnp.int32),
        new_source_tokens=jnp.asarray([-1], dtype=jnp.int32),
        mask=jnp.asarray([True]),
    )
    assert int(result.status) == SCORE_TABLE_ERROR_NEGATIVE_COUNT


def test_score_table_capacity_is_fixed_before_jit():
    build = jax.jit(init_source_score_table, static_argnames=("capacity",))
    table = build(capacity=7)
    assert table.source_token.shape == (7,)
    assert source_score_table_nbytes(7) == 7 * 27 + 4


def test_score_table_measured_generation_is_exact():
    records = [_record(12, 0, fitness=1.0), _record(12, 1, fitness=2.0)]
    update = build_checkpoint_score_update(
        records, score_kind=SCORE_KIND_FITNESS, population_size=4, version=0
    )
    chex.assert_trees_all_equal(
        update.measured_generation, jnp.asarray([12, 12], dtype=jnp.int32)
    )


def test_score_table_eager_and_jit_updates_are_exactly_equal():
    records = [_record(3, 0, utility=1.0), _record(3, 1, utility=2.0)]
    update = build_checkpoint_score_update(
        records, score_kind=SCORE_KIND_UTILITY, population_size=4, version=0
    )
    table = init_source_score_table(4)
    eager_score = apply_checkpoint_score_update(table, update)
    jit_score = jax.jit(apply_checkpoint_score_update)(table, update)
    chex.assert_trees_all_equal(eager_score, jit_score)

    resident = jnp.asarray([-1, -1], dtype=jnp.int32)
    args = (
        eager_score.table,
        resident,
        jnp.asarray([0, 1], dtype=jnp.int32),
        update.source_token,
        jnp.asarray([True, True]),
    )
    eager_resident = update_resident_counts(*args)
    jit_resident = jax.jit(update_resident_counts)(*args)
    chex.assert_trees_all_equal(eager_resident, jit_resident)
