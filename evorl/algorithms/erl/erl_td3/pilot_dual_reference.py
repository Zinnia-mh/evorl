"""Pilot-only paired source utility and held-out fresh selection."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from evorl.utils.jax_utils import right_shift_with_padding
from .pilot_transition_identity import (
    FreshCapture, FreshIdentity, IdentityBatch, check_disjointness,
    sample_fresh_source,
)
from .utility_source_controls import tree_fingerprint


@dataclass(frozen=True)
class PairedRecord:
    reference_family: str
    reference_index: int
    step_scale: float
    loss_ref_before: float
    loss_ref_after: float
    utility: float


@dataclass(frozen=True)
class PairedResult:
    records: tuple[PairedRecord, ...]
    zero_step_results: tuple[PairedRecord, ...]
    update_fingerprint: str
    valid: bool
    reasons: tuple[str, ...]


def evaluate_paired_utility(
    *, loss_fn, optimizer, params, opt_state, source_batch, references,
    source_update_key, reference_keys, step_scales, tolerance,
):
    """One exact source gradient and Optax update, shared by all families."""
    if tolerance <= 0 or any(scale <= 0 for scale in step_scales):
        raise ValueError("positive tolerance and scales required")
    families = ("Replay", "RL-fresh", "EA-fresh")
    if set(references) != set(families) or set(reference_keys) != set(families):
        raise ValueError("three complete reference families required")
    if any(len(references[f]) != len(reference_keys[f]) or not references[f] for f in families):
        raise ValueError("reference batches and keys must align")
    grad = jax.grad(loss_fn)(params, source_batch, source_update_key)
    updates, _ = optimizer.update(grad, opt_state, params)
    fingerprint = tree_fingerprint(updates)
    records, zero = [], []
    reasons = []
    scales = tuple(step_scales)
    virtual = {scale: optax.apply_updates(
        params, jtu.tree_map(lambda x: x * scale, updates)
    ) for scale in scales}
    for family in families:
        for index, (batch, key) in enumerate(zip(references[family], reference_keys[family], strict=True)):
            before = float(loss_fn(params, batch, key))
            for scale in scales:
                after = float(loss_fn(virtual[scale], batch, key))
                records.append(PairedRecord(family, index, scale, before, after, before - after))
            after_zero = float(loss_fn(params, batch, key))
            zero.append(PairedRecord(family, index, 0.0, before, after_zero, before - after_zero))
            if not np.isfinite(before) or any(not np.isfinite(r.utility) for r in records[-len(scales):]):
                reasons.append("non_finite_utility")
            if abs(before - after_zero) > tolerance:
                reasons.append("zero_step_failure")
    for family in families:
        for index in range(len(references[family])):
            before_values = [r.loss_ref_before for r in records if r.reference_family == family and r.reference_index == index]
            if max(before_values) - min(before_values) > tolerance:
                reasons.append("common_reference_before_mismatch")
    return PairedResult(tuple(records), tuple(zero), fingerprint, not reasons, tuple(dict.fromkeys(reasons)))


def _fresh_batch_at(raw, *, run_id, generation, producer, positions, valid_count):
    capture = FreshCapture(raw, run_id, generation, producer)
    shape = raw.dones.shape
    identities = []
    for pos in positions:
        actor, remainder = divmod(pos, shape[1] * shape[2])
        time, episode = divmod(remainder, shape[2])
        identities.append(FreshIdentity(run_id, generation, producer, actor, episode, time, capture.capture_id))
    batch = jtu.tree_map(
        lambda leaf: None if leaf is None else leaf.reshape((-1, *leaf.shape[3:]))[jnp.asarray(positions)],
        raw,
    )
    return IdentityBatch(
        batch=batch, identities=tuple(identities), selected_indices=tuple(positions),
        valid_sample_count=valid_count, overlap_count=0, fresh_capture=capture,
    )


def sample_rl_heldout(raw, own, *, run_id, generation, batch_size, key):
    """Keep the exact legacy own draw; split 512 different valid positions."""
    if not own.eligible:
        return (own, own)
    mask = jax.vmap(lambda d: jnp.logical_not(right_shift_with_padding(d, 1)))(raw.dones)
    valid = np.flatnonzero(np.asarray(jax.device_get(mask)).reshape(-1))
    available = np.setdiff1d(valid, np.asarray(own.selected_indices), assume_unique=True)
    if len(valid) < 3 * batch_size or len(available) < 2 * batch_size:
        hold = IdentityBatch(eligible=False, ineligible_reason="insufficient_fresh_samples", valid_sample_count=len(available))
        return (hold, hold)
    selected = np.asarray(jax.device_get(jax.random.choice(
        key, jnp.asarray(available), (2 * batch_size,), replace=False
    )), dtype=int)
    refs = tuple(_fresh_batch_at(
        raw, run_id=run_id, generation=generation, producer="RL",
        positions=tuple(map(int, selected[i:i + batch_size])), valid_count=len(valid),
    ) for i in (0, batch_size))
    if not check_disjointness((own,), refs, batch_size=batch_size).eligible:
        hold = IdentityBatch(eligible=False, ineligible_reason="source_reference_overlap")
        return (hold, hold)
    return refs


def choose_ea_candidates_and_donors(raw, fitnesses, *, run_id, generation, batch_size, source_key, donor_key):
    """Rank by fitness then actor ID; reserve two different held-out actors."""
    fitness = np.asarray(jax.device_get(fitnesses), dtype=float)
    if not np.all(np.isfinite(fitness)):
        return (), (), "non_finite_fitness"
    ranks = sorted(range(len(fitness)), key=lambda i: (-fitness[i], i))
    mid = ranks[(len(ranks) - 1) // 2] if ranks else None
    candidate_ids = tuple(dict.fromkeys((ranks[0], mid, ranks[-1]))) if ranks else ()
    donor_ids = tuple(i for i in ranks if i not in candidate_ids)[:2]
    if len(candidate_ids) != 3 or len(donor_ids) != 2:
        return (), (), "insufficient_ea_donor_actors"
    candidates = tuple(sample_fresh_source(
        raw, run_id=run_id, generation=generation, producer="EA", actor_id=i,
        batch_size=batch_size, key=source_key,
    ) for i in candidate_ids)
    donors = tuple(sample_fresh_source(
        raw, run_id=run_id, generation=generation, producer="EA", actor_id=i,
        batch_size=batch_size, key=donor_key,
    ) for i in donor_ids)
    gate = check_disjointness(candidates, donors, batch_size=batch_size)
    if not gate.eligible:
        return (tuple(zip(candidate_ids, candidates, strict=True)),
                tuple(zip(donor_ids, donors, strict=True)),
                gate.ineligible_reason)
    return tuple(zip(candidate_ids, candidates, strict=True)), tuple(zip(donor_ids, donors, strict=True)), None


def paired_aggregates(records, *, scale, replay_source_id=None):
    """Return raw-family summaries and a paired replay-control contrast."""
    grouped = {}
    for source_id, result in records.items():
        values = {(family, index): r.utility for r in result.records
                  if r.step_scale == scale for family, index in ((r.reference_family, r.reference_index),)}
        grouped[source_id] = {
            "U_replay": median(values[("Replay", i)] for i in range(4)),
            "U_RL_fresh": median(values[("RL-fresh", i)] for i in range(2)),
            "U_EA_fresh": median(values[("EA-fresh", i)] for i in range(2)),
            "U_balanced_fresh": median([values[("RL-fresh", i)] for i in range(2)] + [values[("EA-fresh", i)] for i in range(2)]),
        }
    if replay_source_id in grouped:
        base = grouped[replay_source_id]
        for metrics in grouped.values():
            for family in ("RL_fresh", "EA_fresh", "balanced_fresh"):
                key = "U_" + family
                metrics["delta_" + family] = (metrics[key] - base[key]) - (metrics["U_replay"] - base["U_replay"])
    return grouped
