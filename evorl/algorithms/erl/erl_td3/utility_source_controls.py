"""Pure primitives for learner-utility source-control diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from evorl.algorithms.erl.erl_td3.learner_utility import (
    UtilityResult,
    td3_critic_loss,
)
from evorl.utils.jax_utils import right_shift_with_padding


LossFn = Callable[[chex.ArrayTree, Any, chex.PRNGKey], chex.Array]
SOURCE_CONTROL_UPDATE_NOISE_TAG = 0x53435550
SOURCE_CONTROL_REFERENCE_EVAL_NOISE_TAG = 0x53435245


def derive_source_update_key(
    baseline_key: chex.PRNGKey, generation: int
) -> chex.PRNGKey:
    key = jax.random.fold_in(baseline_key, SOURCE_CONTROL_UPDATE_NOISE_TAG)
    return jax.random.fold_in(key, generation)


def derive_reference_eval_key(
    baseline_key: chex.PRNGKey, generation: int, reference_index: int
) -> chex.PRNGKey:
    key = jax.random.fold_in(
        baseline_key, SOURCE_CONTROL_REFERENCE_EVAL_NOISE_TAG
    )
    key = jax.random.fold_in(key, generation)
    return jax.random.fold_in(key, reference_index)


class TDStats(NamedTuple):
    """Sealed-compatible TD loss summary."""

    mean: chex.Array
    median: chex.Array
    std: chex.Array


def compute_source_td_stats(
    *,
    agent: Any,
    frozen_agent_state: Any,
    source_batch: Any,
    source_update_key: chex.PRNGKey,
) -> TDStats:
    """Compatibility adapter for the sealed per-transition TD statistics."""
    batch_size = jtu.tree_leaves(source_batch)[0].shape[0]

    def per_transition_loss(single_transition, loss_key):
        singleton_batch = jtu.tree_map(
            lambda leaf: leaf[None, ...], single_transition
        )
        return td3_critic_loss(
            agent, frozen_agent_state, singleton_batch, loss_key
        )

    td_losses = jax.vmap(per_transition_loss)(
        source_batch, jax.random.split(source_update_key, batch_size)
    )
    return TDStats(
        mean=jnp.mean(td_losses),
        median=jnp.median(td_losses),
        std=jnp.std(td_losses),
    )


@dataclass(frozen=True)
class SourceControlBatch:
    source_type: str
    source_id: str
    batch: Any | None
    eligible: bool = True
    reason: str | None = None
    valid_transition_count: int = 0
    evolutionary_fitness: float | None = None


@dataclass(frozen=True)
class FrozenDiagnosticSnapshot:
    generation: int
    learner_agent_state: Any
    critic_opt_state: chex.ArrayTree
    baseline_key: chex.PRNGKey
    baseline_key_fingerprint: str
    critic_fingerprint: str
    target_critic_fingerprint: str
    optimizer_fingerprint: str
    replay_metadata_fingerprint: str
    counter_fingerprint: str
    reference_batches: tuple[Any, ...] = ()
    reference_fingerprints: tuple[str, ...] = ()
    replay_control_batches: tuple[Any, ...] = ()
    replay_control_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceUtilityRecord:
    checkpoint: int
    source_type: str
    source_id: str
    reference_id: str
    step_scale: float
    loss_ref_before: float
    loss_ref_after: float
    utility: float
    numerical_tolerance: float
    evolutionary_fitness: float | None = None
    generation: int = 0
    reference_fingerprint: str = ""
    source_batch_size: int = 0
    td_error_mean: float | None = None
    td_error_median: float | None = None
    td_error_std: float | None = None
    eligible: bool = True
    ineligible_reason: str | None = None
    temporal_alignment: str = "exact"


@dataclass(frozen=True)
class ReferenceAggregate:
    mean: float
    median: float
    std: float
    negative_fraction: float
    near_zero_fraction: float
    sign_consistency: float | None
    all_near_zero: bool


@dataclass(frozen=True)
class CheckpointDiagnostic:
    checkpoint: int
    temporal_alignment: str
    records: tuple[SourceUtilityRecord, ...] = ()
    valid: bool = True
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    zero_step_results: tuple[SourceUtilityRecord, ...] = ()

    @classmethod
    def from_records(
        cls,
        *,
        checkpoint: int,
        temporal_alignment: str,
        records: Iterable[SourceUtilityRecord],
        reasons: Iterable[str] = (),
        warnings: Iterable[str] = (),
        zero_step_results: Iterable[SourceUtilityRecord] = (),
    ) -> "CheckpointDiagnostic":
        records = tuple(records)
        collected_reasons = list(reasons)
        if any(
            not np.isfinite(value)
            for record in records
            for value in (
                record.loss_ref_before,
                record.loss_ref_after,
                record.utility,
            )
        ):
            collected_reasons.append("non_finite_utility")
        unique_reasons = tuple(dict.fromkeys(collected_reasons))
        return cls(
            checkpoint=checkpoint,
            temporal_alignment=temporal_alignment,
            records=records,
            valid=not unique_reasons,
            reasons=unique_reasons,
            warnings=tuple(warnings),
            zero_step_results=tuple(zero_step_results),
        )


def build_rl_own_source_batch(
    raw_trajectory: Any,
    *,
    generation: int,
    source_batch_size: int,
    key: chex.PRNGKey,
) -> SourceControlBatch:
    """Select a matched batch from the one real baseline RL rollout."""
    if raw_trajectory.dones is None:
        raise ValueError("raw RL trajectory must include dones")
    if raw_trajectory.dones.ndim < 3:
        raise ValueError("raw RL trajectory must have [agent, time, env] dones")

    valid_mask = jax.vmap(
        lambda dones: jnp.logical_not(right_shift_with_padding(dones, 1))
    )(raw_trajectory.dones)
    valid_indices = np.flatnonzero(
        np.asarray(jax.device_get(valid_mask)).reshape(-1)
    )
    valid_count = int(valid_indices.size)
    source_id = f"gen_{generation}_rl_own"
    if valid_count < source_batch_size:
        return SourceControlBatch(
            source_type="RL-own",
            source_id=source_id,
            batch=None,
            eligible=False,
            reason="insufficient_rl_own_samples",
            valid_transition_count=valid_count,
        )

    chosen = jax.random.choice(
        key,
        jnp.asarray(valid_indices),
        (source_batch_size,),
        replace=False,
    )

    def _select(leaf):
        if leaf is None:
            return None
        flattened = leaf.reshape((-1, *leaf.shape[3:]))
        return flattened[chosen]

    return SourceControlBatch(
        source_type="RL-own",
        source_id=source_id,
        batch=jtu.tree_map(_select, raw_trajectory),
        eligible=True,
        reason=None,
        valid_transition_count=valid_count,
    )


def tree_fingerprint(tree: Any) -> str:
    """Hash leaf shape, dtype and bytes for checkpoint audit evidence."""
    digest = hashlib.sha256()
    for leaf in jtu.tree_leaves(tree):
        device_leaf = jax.device_get(leaf)
        try:
            array = np.asarray(device_leaf)
        except TypeError:
            array = np.asarray(jax.random.key_data(device_leaf))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def resolve_numerical_tolerance(
    dtype: Any, before: float, after: float, configured: float | None = None
) -> float:
    """Resolve an explicit positive tolerance or a dtype-aware default."""
    if configured is not None:
        if configured <= 0:
            raise ValueError("configured numerical tolerance must be positive")
        return float(configured)
    np_dtype = np.dtype(dtype)
    if not np.issubdtype(np_dtype, np.floating):
        raise TypeError("numerical tolerance requires a floating dtype")
    scale = max(1.0, abs(float(before)), abs(float(after)))
    return float(32 * np.finfo(np_dtype).eps * scale)


def scaled_virtual_critic_utility(
    *,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    source_batch: Any,
    reference_batch: Any,
    source_update_key: chex.PRNGKey,
    reference_eval_key: chex.PRNGKey,
    step_scale: float,
) -> UtilityResult:
    """Run the real optimizer path and scale only its final temporary update."""
    if step_scale < 0:
        raise ValueError("step_scale must be non-negative")
    loss_ref_before = loss_fn(params, reference_batch, reference_eval_key)
    grads = jax.grad(loss_fn)(params, source_batch, source_update_key)
    updates, _ = optimizer.update(grads, opt_state, params)
    scaled_updates = jtu.tree_map(lambda update: update * step_scale, updates)
    virtual_params = optax.apply_updates(params, scaled_updates)
    loss_ref_after = loss_fn(
        virtual_params, reference_batch, reference_eval_key
    )
    return UtilityResult(
        utility=loss_ref_before - loss_ref_after,
        loss_ref_before=loss_ref_before,
        loss_ref_after=loss_ref_after,
    )


def evaluate_frozen_source_controls(
    *,
    checkpoint: int,
    generation: int,
    sources: Iterable[SourceControlBatch],
    reference_batches: Iterable[Any],
    reference_fingerprints: Iterable[str],
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    baseline_key: chex.PRNGKey,
    step_scales: Iterable[float],
    numerical_tolerance: float,
    temporal_alignment: str,
    td_stats_by_source: dict[str, TDStats] | None = None,
) -> CheckpointDiagnostic:
    """Evaluate every source from one immutable optimizer checkpoint."""
    sources = tuple(source for source in sources if source.eligible)
    references = tuple(reference_batches)
    fingerprints = tuple(reference_fingerprints)
    scales = tuple(step_scales)
    if len(references) != len(fingerprints):
        raise ValueError("reference batches and fingerprints must align")
    batch_sizes = {
        int(jtu.tree_leaves(batch)[0].shape[0])
        for batch in (
            tuple(source.batch for source in sources) + references
        )
    }
    if len(batch_sizes) > 1:
        return CheckpointDiagnostic.from_records(
            checkpoint=checkpoint,
            temporal_alignment=temporal_alignment,
            records=(),
            reasons=("batch_size_mismatch",),
        )

    update_key = derive_source_update_key(baseline_key, generation)
    td_stats_by_source = td_stats_by_source or {}

    def make_record(
        source: SourceControlBatch,
        reference_index: int,
        step_scale: float,
    ) -> SourceUtilityRecord:
        try:
            result = scaled_virtual_critic_utility(
                loss_fn=loss_fn,
                optimizer=optimizer,
                params=params,
                opt_state=opt_state,
                source_batch=source.batch,
                reference_batch=references[reference_index],
                source_update_key=update_key,
                reference_eval_key=derive_reference_eval_key(
                    baseline_key, generation, reference_index
                ),
                step_scale=step_scale,
            )
        except FloatingPointError:
            result = UtilityResult(
                utility=jnp.asarray(jnp.nan),
                loss_ref_before=jnp.asarray(jnp.nan),
                loss_ref_after=jnp.asarray(jnp.nan),
            )
        stats = td_stats_by_source.get(source.source_id)
        batch_size = int(jtu.tree_leaves(source.batch)[0].shape[0])
        return SourceUtilityRecord(
            checkpoint=checkpoint,
            source_type=source.source_type,
            source_id=source.source_id,
            reference_id=f"reference_{reference_index}",
            step_scale=step_scale,
            loss_ref_before=float(result.loss_ref_before),
            loss_ref_after=float(result.loss_ref_after),
            utility=float(result.utility),
            numerical_tolerance=numerical_tolerance,
            evolutionary_fitness=source.evolutionary_fitness,
            generation=generation,
            reference_fingerprint=fingerprints[reference_index],
            source_batch_size=batch_size,
            td_error_mean=None if stats is None else float(stats.mean),
            td_error_median=None if stats is None else float(stats.median),
            td_error_std=None if stats is None else float(stats.std),
            temporal_alignment=temporal_alignment,
        )

    ordinary_pairs = tuple(
        (source, reference_index)
        for source in sources
        for reference_index in range(len(references))
    )
    self_sources = tuple(
        SourceControlBatch(
            source_type="Reference-self",
            source_id=f"reference_self_{reference_index}",
            batch=reference,
        )
        for reference_index, reference in enumerate(references)
    )
    self_pairs = tuple(
        (source, reference_index)
        for reference_index, source in enumerate(self_sources)
    )
    pairs = ordinary_pairs + self_pairs
    records = tuple(
        make_record(source, reference_index, scale)
        for source, reference_index in pairs
        for scale in scales
    )
    zero_step_results = tuple(
        make_record(source, reference_index, 0.0)
        for source, reference_index in pairs
    )

    reasons = []
    if any(
        not np.isfinite(record.utility)
        or abs(record.utility) > record.numerical_tolerance
        for record in zero_step_results
    ):
        reasons.append("zero_step_failure")
    for reference_index in range(len(references)):
        before = np.asarray(
            [
                record.loss_ref_before
                for record in records
                if record.reference_id == f"reference_{reference_index}"
            ],
            dtype=float,
        )
        if before.size and (
            not np.all(np.isfinite(before))
            or float(np.max(before) - np.min(before)) > numerical_tolerance
        ):
            reasons.append("common_reference_before_mismatch")

    warnings = []
    if any(
        record.source_type == "Reference-self"
        and record.utility < -record.numerical_tolerance
        for record in records
        if np.isfinite(record.utility)
    ):
        warnings.append("reference_self_negative")
    return CheckpointDiagnostic.from_records(
        checkpoint=checkpoint,
        temporal_alignment=temporal_alignment,
        records=records,
        reasons=reasons,
        warnings=warnings,
        zero_step_results=zero_step_results,
    )


def aggregate_reference_utilities(
    values: Iterable[float], tolerance: float
) -> ReferenceAggregate:
    """Summarize references while excluding near-zero values from sign voting."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("at least one utility value is required")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return ReferenceAggregate(
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            negative_fraction=float("nan"),
            near_zero_fraction=float("nan"),
            sign_consistency=None,
            all_near_zero=False,
        )
    near_zero = np.abs(finite) <= tolerance
    votes = finite[~near_zero]
    if votes.size:
        positive = int(np.count_nonzero(votes > 0))
        negative = int(np.count_nonzero(votes < 0))
        sign_consistency = max(positive, negative) / votes.size
        all_near_zero = False
    else:
        sign_consistency = None
        all_near_zero = True
    return ReferenceAggregate(
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        negative_fraction=float(np.mean(finite < -tolerance)),
        near_zero_fraction=float(np.mean(near_zero)),
        sign_consistency=sign_consistency,
        all_near_zero=all_near_zero,
    )
