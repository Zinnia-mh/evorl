"""Pure primitives for learner-utility source-control diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from evorl.algorithms.erl.erl_td3.learner_utility import UtilityResult
from evorl.utils.jax_utils import right_shift_with_padding


LossFn = Callable[[chex.ArrayTree, Any, chex.PRNGKey], chex.Array]


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
    key: chex.PRNGKey,
    step_scale: float,
) -> UtilityResult:
    """Run the real optimizer path and scale only its final temporary update."""
    if step_scale < 0:
        raise ValueError("step_scale must be non-negative")
    source_key, reference_key = jax.random.split(key)
    loss_ref_before = loss_fn(params, reference_batch, reference_key)
    grads = jax.grad(loss_fn)(params, source_batch, source_key)
    updates, _ = optimizer.update(grads, opt_state, params)
    scaled_updates = jtu.tree_map(lambda update: update * step_scale, updates)
    virtual_params = optax.apply_updates(params, scaled_updates)
    loss_ref_after = loss_fn(virtual_params, reference_batch, reference_key)
    return UtilityResult(
        utility=loss_ref_before - loss_ref_after,
        loss_ref_before=loss_ref_before,
        loss_ref_after=loss_ref_after,
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
