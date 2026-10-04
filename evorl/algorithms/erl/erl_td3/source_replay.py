"""Pure source-score and baseline-draw overlay primitives for the replay pilot.

Phase 4 adds fixed-shape source-score bookkeeping.  It still deliberately
performs no replacement, weighting, freshness, eligibility, or intervention
random draw.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    RL_ORIGIN,
    encode_source_token,
)


SCORE_KIND_NONE = 0
SCORE_KIND_FITNESS = 1
SCORE_KIND_TD = 2
SCORE_KIND_UTILITY = 3

SCORE_TABLE_OK = 0
SCORE_TABLE_ERROR_CAPACITY = 1
SCORE_TABLE_ERROR_NEGATIVE_COUNT = 2
SCORE_TABLE_ERROR_VERSION = 3

EMPTY_SOURCE_TOKEN = -1
NO_MEASURED_GENERATION = -1
NO_SCORE_VERSION = -1


class SourceScoreTable(NamedTuple):
    """Fixed-capacity JAX registry for resident EA source metadata."""

    source_token: chex.Array
    raw_score: chex.Array
    rank_normalized_score: chex.Array
    measured_generation: chex.Array
    score_kind: chex.Array
    validity: chex.Array
    resident_count: chex.Array
    weighted_eligible: chex.Array
    version: chex.Array
    current_version: chex.Array


class CheckpointScoreUpdate(NamedTuple):
    """Device-ready arrays extracted from one host diagnostic checkpoint."""

    source_token: chex.Array
    raw_score: chex.Array
    rank_normalized_score: chex.Array
    measured_generation: chex.Array
    score_kind: chex.Array
    validity: chex.Array
    version: chex.Array


class ScoreTableUpdateResult(NamedTuple):
    table: SourceScoreTable
    status: chex.Array


class ResidentCountUpdateResult(NamedTuple):
    table: SourceScoreTable
    resident_source_tokens: chex.Array
    status: chex.Array


def init_source_score_table(capacity: int) -> SourceScoreTable:
    """Allocate the complete fixed-shape table before JIT compilation."""
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("score_table_capacity must be positive")
    return SourceScoreTable(
        source_token=jnp.full((capacity,), EMPTY_SOURCE_TOKEN, dtype=jnp.int32),
        raw_score=jnp.zeros((capacity,), dtype=jnp.float32),
        rank_normalized_score=jnp.zeros((capacity,), dtype=jnp.float32),
        measured_generation=jnp.full(
            (capacity,), NO_MEASURED_GENERATION, dtype=jnp.int32
        ),
        score_kind=jnp.full((capacity,), SCORE_KIND_NONE, dtype=jnp.int8),
        validity=jnp.zeros((capacity,), dtype=jnp.bool_),
        resident_count=jnp.zeros((capacity,), dtype=jnp.int32),
        weighted_eligible=jnp.zeros((capacity,), dtype=jnp.bool_),
        version=jnp.full((capacity,), NO_SCORE_VERSION, dtype=jnp.int32),
        current_version=jnp.asarray(NO_SCORE_VERSION, dtype=jnp.int32),
    )


def source_score_table_nbytes(capacity: int) -> int:
    """Return raw array bytes, separate from replay provenance sidecars."""
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    per_slot = sum(
        np.dtype(dtype).itemsize
        for dtype in (
            np.int32,
            np.float32,
            np.float32,
            np.int32,
            np.int8,
            np.bool_,
            np.int32,
            np.bool_,
            np.int32,
        )
    )
    return capacity * per_slot + np.dtype(np.int32).itemsize


def rank_normalize_within_checkpoint(
    raw_score: chex.Array, validity: chex.Array
) -> chex.Array:
    """Return bounded average-tie ranks using only one checkpoint's rows."""
    raw_score = jnp.asarray(raw_score, dtype=jnp.float32)
    validity = jnp.asarray(validity, dtype=jnp.bool_) & jnp.isfinite(raw_score)
    lhs = raw_score[:, None]
    rhs = raw_score[None, :]
    pair_valid = validity[:, None] & validity[None, :]
    lower_count = jnp.sum(pair_valid & (rhs < lhs), axis=1)
    tie_count = jnp.sum(pair_valid & (rhs == lhs), axis=1)
    average_rank = lower_count + (tie_count - 1) * 0.5
    valid_count = jnp.sum(validity)
    denominator = jnp.maximum(valid_count - 1, 1)
    normalized = average_rank / denominator
    normalized = jnp.where(valid_count == 1, 0.5, normalized)
    return jnp.where(validity, normalized, 0.0).astype(jnp.float32)


def build_checkpoint_score_update(
    records,
    *,
    score_kind: int,
    population_size: int,
    version: int,
) -> CheckpointScoreUpdate:
    """Convert frozen host DiagnosticRecords into one device update batch."""
    records = tuple(records)
    if not records:
        raise ValueError("checkpoint records must not be empty")
    if score_kind not in (
        SCORE_KIND_FITNESS,
        SCORE_KIND_TD,
        SCORE_KIND_UTILITY,
    ):
        raise ValueError("unknown score_kind")

    generations = {int(record.generation) for record in records}
    checkpoints = {record.checkpoint for record in records}
    if len(generations) != 1 or len(checkpoints) != 1:
        raise ValueError("records must belong to one diagnostic checkpoint")
    if int(version) < 0:
        raise ValueError("real score versions start at zero")

    source_tokens = [
        encode_source_token(
            int(record.generation), int(record.individual_id), population_size
        )
        for record in records
    ]
    if len(set(source_tokens)) != len(source_tokens):
        raise ValueError("checkpoint records contain duplicate source identity")

    if score_kind == SCORE_KIND_FITNESS:
        values = [record.fitness for record in records]
    elif score_kind == SCORE_KIND_TD:
        values = [record.td_error for record in records]
    else:
        values = [record.utility for record in records]

    validity = np.asarray(
        [
            bool(record.eligible)
            and value is not None
            and np.isfinite(float(value))
            for record, value in zip(records, values)
        ],
        dtype=np.bool_,
    )
    raw_score = np.asarray(
        [float(value) if is_valid else 0.0 for value, is_valid in zip(values, validity)],
        dtype=np.float32,
    )
    raw_score_jax = jnp.asarray(raw_score)
    validity_jax = jnp.asarray(validity)
    count = len(records)
    generation = next(iter(generations))
    return CheckpointScoreUpdate(
        source_token=jnp.asarray(source_tokens, dtype=jnp.int32),
        raw_score=raw_score_jax,
        rank_normalized_score=rank_normalize_within_checkpoint(
            raw_score_jax, validity_jax
        ),
        measured_generation=jnp.full((count,), generation, dtype=jnp.int32),
        score_kind=jnp.full((count,), score_kind, dtype=jnp.int8),
        validity=validity_jax,
        version=jnp.asarray(version, dtype=jnp.int32),
    )


def _clear_slot(
    table: SourceScoreTable, slot: chex.Array, source_token: chex.Array
) -> SourceScoreTable:
    """Install a source identity while clearing every stale score field."""
    return table._replace(
        source_token=table.source_token.at[slot].set(source_token),
        raw_score=table.raw_score.at[slot].set(0.0),
        rank_normalized_score=table.rank_normalized_score.at[slot].set(0.0),
        measured_generation=table.measured_generation.at[slot].set(
            NO_MEASURED_GENERATION
        ),
        score_kind=table.score_kind.at[slot].set(SCORE_KIND_NONE),
        validity=table.validity.at[slot].set(False),
        resident_count=table.resident_count.at[slot].set(0),
        weighted_eligible=table.weighted_eligible.at[slot].set(False),
        version=table.version.at[slot].set(NO_SCORE_VERSION),
    )


def _find_source_slot(table: SourceScoreTable, source_token: chex.Array):
    matches = table.source_token == source_token
    return jnp.any(matches), jnp.argmax(matches).astype(jnp.int32)


def apply_checkpoint_score_update(
    table: SourceScoreTable, update: CheckpointScoreUpdate
) -> ScoreTableUpdateResult:
    """Upsert one checkpoint atomically with deterministic slot allocation."""
    expected_version = table.current_version + jnp.int32(1)
    version_ok = update.version == expected_version
    initial_status = jnp.where(
        version_ok, SCORE_TABLE_OK, SCORE_TABLE_ERROR_VERSION
    ).astype(jnp.int8)
    reserved = jnp.zeros_like(table.validity)

    def apply_one(carry, row):
        working, reserved_slots, status = carry

        def do_update(current):
            current_table, current_reserved, current_status = current
            token = update.source_token[row]
            found, existing_slot = _find_source_slot(current_table, token)
            reclaimable = (current_table.resident_count == 0) & ~current_reserved
            has_reclaimable = jnp.any(reclaimable)
            reclaimed_slot = jnp.argmax(reclaimable).astype(jnp.int32)
            slot = jnp.where(found, existing_slot, reclaimed_slot)
            can_write = found | has_reclaimable

            def write_slot(inner):
                inner_table, inner_reserved, inner_status = inner
                prepared = jax.lax.cond(
                    found,
                    lambda value: value,
                    lambda value: _clear_slot(value, slot, token),
                    inner_table,
                )
                prepared = prepared._replace(
                    raw_score=prepared.raw_score.at[slot].set(update.raw_score[row]),
                    rank_normalized_score=prepared.rank_normalized_score.at[slot].set(
                        update.rank_normalized_score[row]
                    ),
                    measured_generation=prepared.measured_generation.at[slot].set(
                        update.measured_generation[row]
                    ),
                    score_kind=prepared.score_kind.at[slot].set(
                        update.score_kind[row]
                    ),
                    validity=prepared.validity.at[slot].set(update.validity[row]),
                    weighted_eligible=prepared.weighted_eligible.at[slot].set(False),
                    version=prepared.version.at[slot].set(update.version),
                )
                return (
                    prepared,
                    inner_reserved.at[slot].set(True),
                    inner_status,
                )

            return jax.lax.cond(
                can_write,
                write_slot,
                lambda inner: (
                    inner[0],
                    inner[1],
                    jnp.asarray(SCORE_TABLE_ERROR_CAPACITY, dtype=jnp.int8),
                ),
                current,
            )

        carry = jax.lax.cond(
            status == SCORE_TABLE_OK,
            do_update,
            lambda current: current,
            (working, reserved_slots, status),
        )
        return carry, None

    (candidate, _, status), _ = jax.lax.scan(
        apply_one,
        (table, reserved, initial_status),
        jnp.arange(update.source_token.shape[0], dtype=jnp.int32),
    )
    candidate = candidate._replace(current_version=update.version)
    final_table = jax.lax.cond(
        status == SCORE_TABLE_OK, lambda _: candidate, lambda _: table, operand=None
    )
    return ScoreTableUpdateResult(table=final_table, status=status)


def update_resident_counts(
    table: SourceScoreTable,
    resident_source_tokens: chex.Array,
    write_indices: chex.Array,
    new_source_tokens: chex.Array,
    mask: chex.Array,
) -> ResidentCountUpdateResult:
    """Apply actual replay overwrites sequentially, including repeated indices."""
    resident_source_tokens = jnp.asarray(resident_source_tokens, dtype=jnp.int32)
    write_indices = jnp.asarray(write_indices, dtype=jnp.int32)
    new_source_tokens = jnp.asarray(new_source_tokens, dtype=jnp.int32)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    initial_status = jnp.asarray(SCORE_TABLE_OK, dtype=jnp.int8)

    def apply_one(carry, row):
        working, resident, status = carry

        def do_write(current):
            current_table, current_resident, current_status = current
            index = write_indices[row]
            old_token = current_resident[index]
            new_token = new_source_tokens[row]

            def decrement_old(inner):
                inner_table, inner_status = inner
                found, slot = _find_source_slot(inner_table, old_token)
                count = inner_table.resident_count[slot]
                valid_decrement = found & (count > 0)
                updated = inner_table._replace(
                    resident_count=inner_table.resident_count.at[slot].set(count - 1)
                )
                return jax.lax.cond(
                    valid_decrement,
                    lambda _: (updated, inner_status),
                    lambda _: (
                        inner_table,
                        jnp.asarray(
                            SCORE_TABLE_ERROR_NEGATIVE_COUNT, dtype=jnp.int8
                        ),
                    ),
                    operand=None,
                )

            decremented, current_status = jax.lax.cond(
                old_token >= 0,
                decrement_old,
                lambda inner: inner,
                (current_table, current_status),
            )

            def increment_new(inner):
                inner_table, inner_status = inner
                found, existing_slot = _find_source_slot(inner_table, new_token)
                reclaimable = inner_table.resident_count == 0
                has_reclaimable = jnp.any(reclaimable)
                reclaimed_slot = jnp.argmax(reclaimable).astype(jnp.int32)
                slot = jnp.where(found, existing_slot, reclaimed_slot)
                can_write = found | has_reclaimable

                def increment(_):
                    prepared = jax.lax.cond(
                        found,
                        lambda value: value,
                        lambda value: _clear_slot(value, slot, new_token),
                        inner_table,
                    )
                    return (
                        prepared._replace(
                            resident_count=prepared.resident_count.at[slot].add(1)
                        ),
                        inner_status,
                    )

                return jax.lax.cond(
                    can_write,
                    increment,
                    lambda _: (
                        inner_table,
                        jnp.asarray(SCORE_TABLE_ERROR_CAPACITY, dtype=jnp.int8),
                    ),
                    operand=None,
                )

            incremented, current_status = jax.lax.cond(
                (current_status == SCORE_TABLE_OK) & (new_token >= 0),
                increment_new,
                lambda inner: inner,
                (decremented, current_status),
            )
            updated_resident = jax.lax.cond(
                current_status == SCORE_TABLE_OK,
                lambda values: values[0].at[index].set(values[1]),
                lambda values: values[0],
                (current_resident, new_token),
            )
            return incremented, updated_resident, current_status

        return (
            jax.lax.cond(
                mask[row] & (status == SCORE_TABLE_OK),
                do_write,
                lambda current: current,
                (working, resident, status),
            ),
            None,
        )

    (candidate_table, candidate_resident, status), _ = jax.lax.scan(
        apply_one,
        (table, resident_source_tokens, initial_status),
        jnp.arange(write_indices.shape[0], dtype=jnp.int32),
    )
    final_table, final_resident = jax.lax.cond(
        status == SCORE_TABLE_OK,
        lambda _: (candidate_table, candidate_resident),
        lambda _: (table, resident_source_tokens),
        operand=None,
    )
    return ResidentCountUpdateResult(
        table=final_table,
        resident_source_tokens=final_resident,
        status=status,
    )


def raise_for_score_table_status(status) -> None:
    """Translate a device status into an explicit host-side safe failure."""
    status = int(jax.device_get(status))
    if status == SCORE_TABLE_OK:
        return
    if status == SCORE_TABLE_ERROR_CAPACITY:
        raise RuntimeError("score table capacity exhausted with all slots resident")
    if status == SCORE_TABLE_ERROR_NEGATIVE_COUNT:
        raise RuntimeError("resident_count would become negative")
    if status == SCORE_TABLE_ERROR_VERSION:
        raise RuntimeError("score table version must advance monotonically")
    raise RuntimeError(f"unknown score table status: {status}")


class ReplayOverlay(NamedTuple):
    """The unchanged baseline draw and its exact origin slot masks."""

    baseline_indices: chex.Array
    final_indices: chex.Array
    ea_slot_mask: chex.Array
    rl_slot_mask: chex.Array


def draw_baseline_indices(
    replay_buffer_state, key: chex.PRNGKey, sample_batch_size: int
) -> chex.Array:
    """Reproduce ``ReplayBuffer.sample`` index generation exactly."""
    return jax.random.randint(
        key,
        (sample_batch_size,),
        minval=0,
        maxval=replay_buffer_state.buffer_size,
    )


def build_origin_slot_masks(
    origin_type: chex.Array, baseline_indices: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Look up the exact realized EA and RL positions in a baseline draw."""
    sampled_origins = origin_type[baseline_indices]
    return sampled_origins == EA_ORIGIN, sampled_origins == RL_ORIGIN


def overlay_ea_indices(
    baseline_indices: chex.Array, origin_type: chex.Array
) -> ReplayOverlay:
    """Build the Phase-2 overlay seam without replacing any transition."""
    ea_slot_mask, rl_slot_mask = build_origin_slot_masks(
        origin_type, baseline_indices
    )
    return ReplayOverlay(
        baseline_indices=baseline_indices,
        final_indices=baseline_indices,
        ea_slot_mask=ea_slot_mask,
        rl_slot_mask=rl_slot_mask,
    )
