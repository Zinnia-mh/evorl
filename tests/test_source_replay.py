from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import pytest

from evorl.algorithms.erl.erl_td3.source_replay import (
    SCORE_KIND_FITNESS,
    SCORE_KIND_NONE,
    SCORE_KIND_TD,
    SCORE_KIND_UTILITY,
    SCORE_TABLE_ERROR_CAPACITY,
    SCORE_TABLE_ERROR_NEGATIVE_COUNT,
    SCORE_TABLE_ERROR_VERSION,
    SCORE_TABLE_OK,
    apply_weighted_ea_overlay,
    apply_checkpoint_score_update,
    build_weighted_source_distribution,
    build_checkpoint_score_update,
    build_origin_slot_masks,
    compute_source_eligibility,
    draw_baseline_indices,
    init_source_score_table,
    overlay_ea_indices,
    raise_for_score_table_status,
    rank_normalize_within_checkpoint,
    source_score_table_nbytes,
    sample_weighted_ea_indices,
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


def _eligibility_table(
    *,
    tokens=(10, 20, 30, -1),
    raw=(1.0, 2.0, 3.0, 0.0),
    rank=(0.0, 0.5, 1.0, 0.0),
    measured=(10, 10, 10, -1),
    kinds=(SCORE_KIND_FITNESS, SCORE_KIND_TD, SCORE_KIND_UTILITY, SCORE_KIND_NONE),
    validity=(True, True, True, False),
    counts=(4, 4, 4, 0),
    versions=(0, 0, 0, -1),
    weighted=(False, False, False, False),
):
    # Malformed score metadata is an explicit eligibility input class.  Build
    # it without the process-wide NaN trap rejecting the fixture itself.
    with jax.debug_nans(False):
        table = init_source_score_table(len(tokens))
        return table._replace(
            source_token=jnp.asarray(tokens, dtype=jnp.int32),
            raw_score=jnp.asarray(raw, dtype=jnp.float32),
            rank_normalized_score=jnp.asarray(rank, dtype=jnp.float32),
            measured_generation=jnp.asarray(measured, dtype=jnp.int32),
            score_kind=jnp.asarray(kinds, dtype=jnp.int8),
            validity=jnp.asarray(validity, dtype=jnp.bool_),
            resident_count=jnp.asarray(counts, dtype=jnp.int32),
            weighted_eligible=jnp.asarray(weighted, dtype=jnp.bool_),
            version=jnp.asarray(versions, dtype=jnp.int32),
            current_version=jnp.asarray(0, dtype=jnp.int32),
        )


def _eligibility(table, *, current=10, window=3, minimum=4):
    return compute_source_eligibility(
        table,
        current_generation=current,
        freshness_window=window,
        min_resident=minimum,
    )


def test_fresh_valid_source_with_minimum_residency_is_eligible():
    result = _eligibility(_eligibility_table())
    assert bool(result.eligible[0])


def test_age_equal_to_window_is_still_eligible():
    result = _eligibility(_eligibility_table(measured=(7, 10, 10, -1)))
    assert int(result.age[0]) == 3
    assert bool(result.eligible[0])


def test_age_one_past_window_is_ineligible():
    result = _eligibility(_eligibility_table(measured=(6, 10, 10, -1)))
    assert int(result.age[0]) == 4
    assert not bool(result.eligible[0])


def test_future_measured_generation_is_ineligible():
    result = _eligibility(
        _eligibility_table(measured=(11, 10, 10, -1)),
        current=jnp.asarray(10, dtype=jnp.uint32),
    )
    assert int(result.age[0]) == -1
    assert not bool(result.eligible[0])


def test_insufficient_resident_source_is_neutral():
    result = _eligibility(_eligibility_table(counts=(3, 4, 4, 0)))
    assert not bool(result.eligible[0])


def test_resident_count_equal_to_minimum_is_eligible():
    result = _eligibility(_eligibility_table(counts=(4, 4, 4, 0)))
    assert bool(result.eligible[0])


def test_missing_score_version_is_neutral():
    result = _eligibility(_eligibility_table(versions=(-1, 0, 0, -1)))
    assert not bool(result.eligible[0])


def test_missing_measured_generation_is_neutral():
    result = _eligibility(_eligibility_table(measured=(-1, 10, 10, -1)))
    assert not bool(result.eligible[0])


def test_invalid_score_is_neutral():
    result = _eligibility(_eligibility_table(validity=(False, True, True, False)))
    assert not bool(result.eligible[0])


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_raw_score_is_neutral(value):
    result = _eligibility(_eligibility_table(raw=(value, 2.0, 3.0, 0.0)))
    assert not bool(result.eligible[0])


def test_nan_score_is_neutral():
    result = _eligibility(_eligibility_table(raw=(jnp.nan, 2.0, 3.0, 0.0)))
    assert not bool(result.eligible[0])


def test_inf_score_is_neutral():
    result = _eligibility(_eligibility_table(rank=(jnp.inf, 0.5, 1.0, 0.0)))
    assert not bool(result.eligible[0])


def test_empty_source_slot_is_neutral():
    result = _eligibility(_eligibility_table())
    assert not bool(result.eligible[3])


def test_zero_resident_source_is_neutral():
    result = _eligibility(_eligibility_table(counts=(0, 4, 4, 0)))
    assert not bool(result.eligible[0])


def test_empty_eligible_set_has_no_weighted_sources():
    table = _eligibility_table(validity=(False, False, False, False))
    result = _eligibility(table)
    assert int(result.eligible_count) == 0
    assert not bool(jnp.any(result.table.weighted_eligible))


def test_single_eligible_source_marks_only_that_source():
    table = _eligibility_table(counts=(4, 3, 0, 0))
    result = _eligibility(table)
    chex.assert_trees_all_equal(
        result.eligible, jnp.asarray([True, False, False, False])
    )
    chex.assert_trees_all_equal(result.table.resident_count, table.resident_count)


def test_mixed_eligible_ineligible_sources_have_exact_mask():
    table = _eligibility_table(
        tokens=(10, 20, 30, 40, -1, -2),
        raw=(1.0, 2.0, 3.0, jnp.nan, 0.0, 5.0),
        rank=(0.1, 0.2, 0.3, 0.4, 0.0, 0.5),
        measured=(10, 10, 5, 10, -1, 10),
        kinds=(
            SCORE_KIND_FITNESS,
            SCORE_KIND_TD,
            SCORE_KIND_UTILITY,
            SCORE_KIND_FITNESS,
            SCORE_KIND_NONE,
            SCORE_KIND_TD,
        ),
        validity=(True, True, True, True, False, True),
        counts=(4, 3, 4, 4, 0, 4),
        versions=(0, 0, 0, 0, -1, 0),
    )
    result = _eligibility(table)
    chex.assert_trees_all_equal(
        result.eligible,
        jnp.asarray([True, False, False, False, False, False]),
    )


def test_all_score_kinds_share_identical_window_and_min_resident_rules():
    table = _eligibility_table(
        measured=(7, 7, 7, -1),
        counts=(4, 4, 4, 0),
    )
    result = _eligibility(table, current=10, window=3, minimum=4)
    chex.assert_trees_all_equal(
        result.eligible, jnp.asarray([True, True, True, False])
    )


def test_eligibility_refresh_clears_previously_true_stale_flag():
    table = _eligibility_table(
        measured=(1, 1, 1, -1), weighted=(True, True, True, True)
    )
    result = _eligibility(table, current=10, window=3)
    assert not bool(jnp.any(result.table.weighted_eligible))


def test_eligibility_does_not_modify_score_or_resident_fields():
    table = _eligibility_table(weighted=(True, False, True, False))
    result = _eligibility(table)
    for field in table._fields:
        if field != "weighted_eligible":
            chex.assert_trees_all_equal(getattr(result.table, field), getattr(table, field))


def test_eligibility_eager_and_jit_are_exactly_equal():
    table = _eligibility_table(measured=(7, 8, 11, -1))
    eager = _eligibility(table, current=jnp.asarray(10, dtype=jnp.uint32))
    compiled = jax.jit(
        compute_source_eligibility,
        static_argnames=("freshness_window", "min_resident"),
    )(
        table,
        current_generation=jnp.asarray(10, dtype=jnp.uint32),
        freshness_window=3,
        min_resident=4,
    )
    chex.assert_trees_all_equal(eager, compiled)


def test_negative_window_is_rejected():
    with pytest.raises(ValueError, match="freshness_window"):
        _eligibility(_eligibility_table(), window=-1)


def test_nonpositive_min_resident_is_rejected():
    with pytest.raises(ValueError, match="min_resident"):
        _eligibility(_eligibility_table(), minimum=0)


def _sampler_fixture(
    *,
    counts=(4, 2, 4, 0),
    ranks=(0.0, 1.0, 0.5, 0.0),
    eligible=(True, True, False, False),
):
    table = _eligibility_table(
        tokens=(10, 20, 30, -1),
        rank=ranks,
        counts=counts,
        weighted=eligible,
    )
    resident_tokens = jnp.asarray(
        [10, 10, 10, 10, 20, 20, 30, 30, 30, 30, -1, -1],
        dtype=jnp.int32,
    )
    resident_mask = jnp.asarray(
        [True] * 11 + [False],
        dtype=jnp.bool_,
    )
    return table, resident_tokens, resident_mask


def _distribution(*, alpha=0.5, temperature=1.0, **fixture_kwargs):
    table, resident_tokens, resident_mask = _sampler_fixture(**fixture_kwargs)
    return build_weighted_source_distribution(
        table,
        resident_source_tokens=resident_tokens,
        resident_mask=resident_mask,
        alpha=alpha,
        replay_temperature=temperature,
    )


def test_source_probabilities_sum_to_one_and_are_finite():
    result = _distribution()
    assert jnp.isclose(result.source_probability.sum(), 1.0)
    assert bool(jnp.all(jnp.isfinite(result.source_probability)))
    assert bool(jnp.all(result.source_probability >= 0))


def test_transition_probabilities_sum_to_one_and_are_finite():
    result = _distribution()
    assert jnp.isclose(result.transition_probability.sum(), 1.0)
    assert bool(jnp.all(jnp.isfinite(result.transition_probability)))
    assert bool(jnp.all(result.transition_probability >= 0))


def test_alpha_zero_distribution_is_exact_baseline_distribution():
    result = _distribution(alpha=0.0)
    chex.assert_trees_all_equal(
        result.source_probability,
        jnp.asarray([0.4, 0.2, 0.4, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.transition_probability[:10],
        jnp.full((10,), 0.1, dtype=jnp.float32),
    )


def test_uniform_floor_is_preserved_per_transition():
    result = _distribution(alpha=0.5)
    assert float(result.uniform_floor_per_transition) == pytest.approx(0.05)
    assert bool(jnp.all(result.transition_probability[:10] >= 0.05))


def test_ineligible_source_transition_probability_is_exactly_baseline():
    result = _distribution(alpha=0.75, temperature=0.4)
    chex.assert_trees_all_equal(
        result.transition_probability[6:10],
        jnp.full((4,), 0.1, dtype=jnp.float32),
    )


def test_tiny_source_below_m_min_is_not_amplified():
    result = _distribution(counts=(4, 2, 4, 0), eligible=(True, False, True, False))
    assert float(result.source_probability[1]) == pytest.approx(0.2)
    chex.assert_trees_all_equal(
        result.transition_probability[4:6],
        jnp.full((2,), 0.1, dtype=jnp.float32),
    )


def test_empty_eligible_set_is_uniform_over_ea_transitions():
    result = _distribution(eligible=(False, False, False, False))
    chex.assert_trees_all_equal(
        result.transition_probability[:10],
        jnp.full((10,), 0.1, dtype=jnp.float32),
    )


def test_single_eligible_source_reduces_to_baseline_mass():
    result = _distribution(eligible=(False, True, False, False))
    chex.assert_trees_all_equal(
        result.source_probability,
        jnp.asarray([0.4, 0.2, 0.4, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.transition_probability[:10],
        jnp.full((10,), 0.1, dtype=jnp.float32),
    )


def test_multiple_eligible_sources_preserve_total_eligible_mass():
    result = _distribution(alpha=0.8, temperature=0.3)
    assert float(result.weighted_source_probability[:2].sum()) == pytest.approx(0.6)
    assert float(result.source_probability[:2].sum()) == pytest.approx(0.6)


def test_mixed_eligible_ineligible_sources_preserve_ineligible_mass():
    result = _distribution(alpha=1.0, temperature=0.2)
    assert float(result.weighted_source_probability[2]) == pytest.approx(0.4)
    assert float(result.source_probability[2]) == pytest.approx(0.4)


def test_unequal_resident_counts_define_baseline_source_mass():
    table = _eligibility_table(
        tokens=(10, 20, 30, -1),
        counts=(100, 10, 2, 0),
        weighted=(True, True, False, False),
    )
    tokens = jnp.asarray([10] * 100 + [20] * 10 + [30] * 2, dtype=jnp.int32)
    result = build_weighted_source_distribution(
        table,
        resident_source_tokens=tokens,
        resident_mask=jnp.ones((112,), dtype=jnp.bool_),
        alpha=0.5,
        replay_temperature=1.0,
    )
    chex.assert_trees_all_close(
        result.baseline_source_probability,
        jnp.asarray([100 / 112, 10 / 112, 2 / 112, 0], dtype=jnp.float32),
    )


def test_sampling_is_uniform_within_selected_source():
    result = _distribution(alpha=0.7, temperature=0.5)
    assert float(result.transition_probability[0]) == float(
        result.transition_probability[3]
    )
    assert float(result.transition_probability[4]) == float(
        result.transition_probability[5]
    )


def test_sampler_is_deterministic_for_fixed_key():
    distribution = _distribution(alpha=0.7, temperature=0.5)
    key = jax.random.key(2026)
    first = sample_weighted_ea_indices(distribution, key, sample_size=128)
    second = sample_weighted_ea_indices(distribution, key, sample_size=128)
    chex.assert_trees_all_equal(first, second)
    assert not bool(jnp.array_equal(first.next_intervention_key, key))


def test_empirical_sampling_matches_analytical_transition_probability():
    # Fixed before implementation: seed=2026, draws=50_000, tolerance=0.015.
    distribution = _distribution(alpha=0.7, temperature=0.5)
    sampled = sample_weighted_ea_indices(
        distribution, jax.random.key(2026), sample_size=50_000
    )
    empirical = jnp.bincount(sampled.sampled_indices, length=12) / 50_000
    chex.assert_trees_all_close(
        empirical,
        distribution.transition_probability,
        atol=0.015,
        rtol=0.0,
    )


def test_extreme_finite_logits_remain_finite():
    result = _distribution(alpha=1.0, temperature=1e-20)
    assert bool(jnp.all(jnp.isfinite(result.source_probability)))
    assert bool(jnp.all(jnp.isfinite(result.transition_probability)))
    assert jnp.isclose(result.transition_probability.sum(), 1.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        _distribution(alpha=alpha)


@pytest.mark.parametrize(
    "temperature", [0.0, -1.0, float("nan"), float("inf")]
)
def test_nonpositive_replay_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="replay_temperature"):
        _distribution(temperature=temperature)


def test_no_resident_ea_transition_is_handled_without_nan():
    table = init_source_score_table(4)
    result = build_weighted_source_distribution(
        table,
        resident_source_tokens=jnp.full((8,), -1, dtype=jnp.int32),
        resident_mask=jnp.asarray([True] * 4 + [False] * 4),
        alpha=0.5,
        replay_temperature=1.0,
    )
    assert not bool(result.has_resident_ea)
    assert float(result.source_probability.sum()) == 0.0
    assert float(result.transition_probability.sum()) == 0.0
    assert bool(jnp.all(jnp.isfinite(result.transition_probability)))


def test_sampler_reports_required_audit_metrics():
    result = _distribution(alpha=0.5)
    assert int(result.n_ea) == 10
    assert int(result.eligible_source_count) == 2
    assert int(result.resident_source_count) == 3
    assert float(result.effective_source_count) == pytest.approx(
        1.0 / float(jnp.sum(result.source_probability**2))
    )
    assert float(result.max_source_probability) == float(
        jnp.max(result.source_probability)
    )
    assert float(result.uniform_component_total_mass) == pytest.approx(0.5)
    assert float(result.eligible_baseline_mass) == pytest.approx(0.6)
    assert result.transition_cdf.shape == result.transition_probability.shape
    assert float(result.transition_cdf[-1]) == pytest.approx(1.0)
    assert bool(jnp.all(jnp.diff(result.transition_cdf) >= 0))


def test_weighted_sampler_eager_and_jit_are_exactly_equal():
    table, tokens, mask = _sampler_fixture()
    eager_distribution = build_weighted_source_distribution(
        table,
        resident_source_tokens=tokens,
        resident_mask=mask,
        alpha=0.7,
        replay_temperature=0.5,
    )
    compiled_distribution = jax.jit(
        build_weighted_source_distribution,
        static_argnames=("alpha", "replay_temperature"),
    )(
        table,
        resident_source_tokens=tokens,
        resident_mask=mask,
        alpha=0.7,
        replay_temperature=0.5,
    )
    chex.assert_trees_all_equal(eager_distribution, compiled_distribution)

    key = jax.random.key(11)
    eager_sample = sample_weighted_ea_indices(
        eager_distribution, key, sample_size=64
    )
    compiled_sample = jax.jit(
        sample_weighted_ea_indices, static_argnames=("sample_size",)
    )(compiled_distribution, key, sample_size=64)
    chex.assert_trees_all_equal(eager_sample, compiled_sample)


def _overlay_fixture():
    distribution = _distribution(alpha=0.8, temperature=0.4)
    origin_type = jnp.asarray(
        [EA_ORIGIN] * 10 + [RL_ORIGIN, RL_ORIGIN], dtype=jnp.int8
    )
    baseline_indices = jnp.asarray([10, 0, 10, 4, 7, 10], dtype=jnp.int32)
    return distribution, origin_type, baseline_indices


def test_real_overlay_changes_only_ea_slots():
    distribution, origin_type, baseline_indices = _overlay_fixture()
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, jax.random.key(7)
    )

    chex.assert_trees_all_equal(
        result.final_indices[result.rl_slot_mask],
        baseline_indices[result.rl_slot_mask],
    )
    assert bool(jnp.any(
        result.final_indices[result.ea_slot_mask]
        != baseline_indices[result.ea_slot_mask]
    ))


def test_real_overlay_preserves_positions_counts_and_batch_size():
    distribution, origin_type, baseline_indices = _overlay_fixture()
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, jax.random.key(9)
    )

    chex.assert_trees_all_equal(
        result.ea_slot_mask,
        jnp.asarray([False, True, False, True, True, False]),
    )
    chex.assert_trees_all_equal(result.rl_slot_mask, ~result.ea_slot_mask)
    assert result.final_indices.shape == baseline_indices.shape
    assert int(result.ea_slot_mask.sum()) == 3
    assert int(result.rl_slot_mask.sum()) == 3
    assert int(result.replacement_count) == int(
        jnp.sum(result.final_indices != baseline_indices)
    )


def test_real_overlay_replacements_are_resident_ea_indices_only():
    distribution, origin_type, baseline_indices = _overlay_fixture()
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, jax.random.key(11)
    )
    replacements = result.final_indices[result.ea_slot_mask]

    assert bool(jnp.all(replacements >= 0))
    assert bool(jnp.all(replacements < 11))
    chex.assert_trees_all_equal(
        origin_type[replacements],
        jnp.full(replacements.shape, EA_ORIGIN, dtype=jnp.int8),
    )


def test_real_overlay_allows_duplicate_replacement():
    distribution, origin_type, _ = _overlay_fixture()
    baseline_indices = jnp.zeros((20,), dtype=jnp.int32)
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, jax.random.key(12)
    )
    replacements = result.final_indices[result.ea_slot_mask]

    assert replacements.shape[0] == 20
    assert int(jnp.unique(replacements).shape[0]) < replacements.shape[0]


def test_real_overlay_fixed_key_is_deterministic_and_advances_key():
    distribution, origin_type, baseline_indices = _overlay_fixture()
    key = jax.random.key(13)
    first = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, key
    )
    second = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, key
    )

    chex.assert_trees_all_equal(first, second)
    assert not bool(jnp.array_equal(first.next_intervention_key, key))


def test_real_overlay_no_ea_slot_keeps_indices_and_key():
    distribution, origin_type, _ = _overlay_fixture()
    baseline_indices = jnp.asarray([10, 11, 10, 11], dtype=jnp.int32)
    key = jax.random.key(17)
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, key
    )

    chex.assert_trees_all_equal(result.final_indices, baseline_indices)
    chex.assert_trees_all_equal(result.next_intervention_key, key)
    assert int(result.replacement_count) == 0


def test_real_overlay_no_resident_ea_keeps_indices_and_key():
    table = init_source_score_table(4)
    distribution = build_weighted_source_distribution(
        table,
        resident_source_tokens=jnp.full((12,), -1, dtype=jnp.int32),
        resident_mask=jnp.ones((12,), dtype=jnp.bool_),
        alpha=0.5,
        replay_temperature=1.0,
    )
    origin_type = jnp.full((12,), EA_ORIGIN, dtype=jnp.int8)
    baseline_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    key = jax.random.key(19)
    result = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, key
    )

    chex.assert_trees_all_equal(result.final_indices, baseline_indices)
    chex.assert_trees_all_equal(result.next_intervention_key, key)


def test_real_overlay_eager_and_jit_are_exactly_equal():
    distribution, origin_type, baseline_indices = _overlay_fixture()
    key = jax.random.key(23)
    eager = apply_weighted_ea_overlay(
        baseline_indices, origin_type, distribution, key
    )
    compiled = jax.jit(apply_weighted_ea_overlay)(
        baseline_indices, origin_type, distribution, key
    )
    chex.assert_trees_all_equal(eager, compiled)
