"""Pilot-only gradient alignment and actual virtual-displacement diagnostics.

First-order predicted utility is local loss geometry, not observed return and
not the measured finite-step utility. A cache belongs to exactly one frozen
checkpoint and must never be shared across learner states or evaluation keys.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.tree_util as jtu
import numpy as np

from .utility_source_controls import tree_fingerprint


def _flat(tree):
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([
        np.asarray(jax.device_get(leaf), dtype=np.float64).reshape(-1)
        for leaf in leaves
    ])


def _norm(tree):
    return float(np.linalg.norm(_flat(tree)))


def _dot(left, right):
    if jtu.tree_structure(left) != jtu.tree_structure(right):
        raise ValueError("gradient_parameter_tree_mismatch")
    return float(np.dot(_flat(left), _flat(right)))


def _machine_epsilon(params):
    """Floating-point precision of the frozen parameter tree."""
    leaves = jtu.tree_leaves(params)
    dtypes = [np.asarray(jax.device_get(leaf)).dtype for leaf in leaves]
    floating = [dtype for dtype in dtypes if np.issubdtype(dtype, np.floating)]
    try:
        dtype = np.result_type(*floating) if floating else np.dtype(np.float32)
        machine_epsilon = float(np.finfo(dtype).eps)
    except (TypeError, ValueError):
        machine_epsilon = float(np.finfo(np.float32).eps)
    return 32.0 * machine_epsilon


def _epsilon(params):
    """Parameter displacement resolution for the frozen parameter scale."""
    return _machine_epsilon(params) * max(1.0, _norm(params))


def _cosine(left, right, epsilon):
    left_norm, right_norm = _norm(left), _norm(right)
    if not np.isfinite(left_norm) or not np.isfinite(right_norm):
        return None, "non_finite_norm"
    if left_norm <= epsilon or right_norm <= epsilon:
        return None, "near_zero_norm"
    value = _dot(left, right) / (left_norm * right_norm)
    if not np.isfinite(value):
        return None, "non_finite_norm"
    return float(np.clip(value, -1.0, 1.0)), "ok"


@dataclass(frozen=True)
class ReferenceGradient:
    gradient: object
    gradient_fingerprint: str
    reference_fingerprint: str
    reference_eval_key_fingerprint: str
    loss_ref_before: float
    params_fingerprint: str
    gradient_flat: np.ndarray
    gradient_norm: float


class ReferenceGradientCache:
    """One-instance-per-checkpoint cache, bound to parameters, data and noise."""
    def __init__(
        self, *, checkpoint, frozen_learner_fingerprint,
        frozen_critic_fingerprint, tolerance,
    ):
        if checkpoint < 0 or not frozen_learner_fingerprint or not frozen_critic_fingerprint:
            raise ValueError("a frozen checkpoint identity is required")
        if tolerance <= 0:
            raise ValueError("positive numerical tolerance required")
        self.checkpoint = checkpoint
        self.frozen_learner_fingerprint = frozen_learner_fingerprint
        self.frozen_critic_fingerprint = frozen_critic_fingerprint
        self.tolerance = tolerance
        self.params_fingerprint = None
        self.entries: dict[tuple[str, int], ReferenceGradient] = {}

    def get(
        self, *, family, index, params, reference_batch, reference_eval_key,
        loss_fn, loss_ref_before, params_fingerprint=None,
    ):
        current_params = params_fingerprint or tree_fingerprint(params)
        if self.params_fingerprint is None:
            self.params_fingerprint = current_params
        if current_params != self.params_fingerprint:
            raise ValueError("reference_cache_mismatch: frozen critic changed")
        reference_fingerprint = tree_fingerprint(reference_batch)
        key_fingerprint = tree_fingerprint(reference_eval_key)
        cache_key = (family, index)
        existing = self.entries.get(cache_key)
        if existing is not None:
            if (existing.reference_fingerprint != reference_fingerprint
                    or existing.reference_eval_key_fingerprint != key_fingerprint
                    or existing.params_fingerprint != current_params
                    or not np.isclose(existing.loss_ref_before, loss_ref_before,
                                      rtol=0, atol=self.tolerance, equal_nan=True)):
                raise ValueError("reference_cache_mismatch: batch, key or before")
            return existing
        gradient = jax.grad(loss_fn)(params, reference_batch, reference_eval_key)
        gradient_flat = _flat(gradient)
        entry = ReferenceGradient(
            gradient=gradient,
            gradient_fingerprint=tree_fingerprint(gradient),
            reference_fingerprint=reference_fingerprint,
            reference_eval_key_fingerprint=key_fingerprint,
            loss_ref_before=float(loss_ref_before),
            params_fingerprint=current_params,
            gradient_flat=gradient_flat,
            gradient_norm=float(np.linalg.norm(gradient_flat)),
        )
        self.entries[cache_key] = entry
        return entry


@dataclass(frozen=True)
class MechanismRecord:
    reference_family: str
    reference_index: int
    step_scale: float
    source_gradient_norm: float
    reference_gradient_norm: float
    raw_gradient_cosine: float | None
    optax_update_norm: float
    actual_parameter_displacement_norm: float
    negative_update_reference_cosine: float | None
    first_order_predicted_utility: float
    actual_utility: float
    first_order_residual: float
    cosine_status: str
    raw_cosine_status: str
    update_cosine_status: str
    numerical_status: str
    eligible: bool
    ineligible_reason: str | None
    loss_ref_before: float
    source_gradient_fingerprint: str
    source_update_fingerprint: str
    reference_gradient_fingerprint: str
    reference_eval_key_fingerprint: str
    reference_fingerprint: str
    frozen_critic_fingerprint: str


def _cosine_flat(left, right, left_norm, right_norm, epsilon):
    if not np.isfinite(left_norm) or not np.isfinite(right_norm):
        return None, "non_finite_norm"
    if left_norm <= epsilon or right_norm <= epsilon:
        return None, "near_zero_norm"
    value = float(np.dot(left, right) / (left_norm * right_norm))
    if not np.isfinite(value):
        return None, "non_finite_norm"
    return float(np.clip(value, -1.0, 1.0)), "ok"


def measure_mechanisms(
    *, loss_fn, params, source_gradient, optax_update,
    virtual_params_by_scale, utility_records, reference_batches,
    reference_keys, cache, update_fingerprint,
):
    """Match measured Utilities with cached frozen-reference gradients."""
    source_flat = _flat(source_gradient)
    update_flat = _flat(optax_update)
    source_norm = float(np.linalg.norm(source_flat))
    update_norm = float(np.linalg.norm(update_flat))
    source_fingerprint = tree_fingerprint(source_gradient)
    params_fingerprint = tree_fingerprint(params)
    epsilon = _epsilon(params)
    direction_epsilon = _machine_epsilon(params)
    displacements = {
        scale: _flat(jtu.tree_map(
            lambda after, before: after - before, virtual, params
        ))
        for scale, virtual in virtual_params_by_scale.items()
    }
    displacements[0.0] = np.zeros_like(_flat(params))
    displacement_norms = {
        scale: float(np.linalg.norm(vector))
        for scale, vector in displacements.items()
    }
    records = []
    reasons = []
    local_references = {}
    for utility in utility_records:
        family, index, scale = (
            utility.reference_family, utility.reference_index, utility.step_scale
        )
        key = (family, index)
        if key not in local_references:
            reference_batch = reference_batches[family][index]
            reference_key = reference_keys[family][index]
            entry = cache.get(
                family=family, index=index, params=params,
                params_fingerprint=params_fingerprint,
                reference_batch=reference_batch,
                reference_eval_key=reference_key,
                loss_fn=loss_fn, loss_ref_before=utility.loss_ref_before,
            )
            raw_cosine, raw_status = _cosine_flat(
                source_flat, entry.gradient_flat, source_norm,
                entry.gradient_norm, direction_epsilon,
            )
            update_cosine, update_status = _cosine_flat(
                -update_flat, entry.gradient_flat, update_norm,
                entry.gradient_norm, direction_epsilon,
            )
            local_references[key] = (
                entry, raw_cosine, raw_status, update_cosine, update_status,
            )
        else:
            entry = local_references[key][0]
            if not np.isclose(
                entry.loss_ref_before, utility.loss_ref_before,
                rtol=0, atol=cache.tolerance, equal_nan=True,
            ):
                raise ValueError("reference_cache_mismatch: before by scale")
        entry, raw_cosine, raw_status, update_cosine, update_status = (
            local_references[key]
        )
        displacement = displacements[scale]
        displacement_norm = displacement_norms[scale]
        predicted = -float(np.dot(entry.gradient_flat, displacement))
        actual = utility.utility
        residual = actual - predicted
        nonfinite_reason = None
        if not np.isfinite(source_norm) or not np.all(np.isfinite(source_flat)):
            nonfinite_reason = "non_finite_gradient"
        elif (not np.isfinite(entry.gradient_norm)
              or not np.all(np.isfinite(entry.gradient_flat))):
            nonfinite_reason = "non_finite_reference_gradient"
        elif not np.isfinite(update_norm) or not np.all(np.isfinite(update_flat)):
            nonfinite_reason = "non_finite_update"
        elif not all(np.isfinite(x) for x in (
            displacement_norm, predicted, actual, residual, utility.loss_ref_before
        )):
            nonfinite_reason = "non_finite_mechanism"
        if nonfinite_reason:
            status = nonfinite_reason
            reasons.append(nonfinite_reason)
        elif raw_status != "ok" or update_status != "ok":
            status = "near_zero_norm"
        elif displacement_norm <= epsilon:
            status = "near_zero_displacement"
        else:
            status = "ok"
        records.append(MechanismRecord(
            reference_family=family, reference_index=index, step_scale=scale,
            source_gradient_norm=source_norm,
            reference_gradient_norm=entry.gradient_norm,
            raw_gradient_cosine=raw_cosine,
            optax_update_norm=update_norm,
            actual_parameter_displacement_norm=displacement_norm,
            negative_update_reference_cosine=update_cosine,
            first_order_predicted_utility=predicted,
            actual_utility=actual,
            first_order_residual=residual,
            cosine_status="near_zero_norm" if status == "near_zero_norm" else (
                "non_finite_norm" if nonfinite_reason else "ok"
            ),
            raw_cosine_status=raw_status,
            update_cosine_status=update_status,
            numerical_status=status,
            eligible=nonfinite_reason is None,
            ineligible_reason=nonfinite_reason,
            loss_ref_before=utility.loss_ref_before,
            source_gradient_fingerprint=source_fingerprint,
            source_update_fingerprint=update_fingerprint,
            reference_gradient_fingerprint=entry.gradient_fingerprint,
            reference_eval_key_fingerprint=entry.reference_eval_key_fingerprint,
            reference_fingerprint=entry.reference_fingerprint,
            frozen_critic_fingerprint=cache.frozen_critic_fingerprint,
        ))
    return tuple(records), tuple(dict.fromkeys(reasons))
