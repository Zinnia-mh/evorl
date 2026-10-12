"""Opt-in P1-A identity primitives; never called by the sealed training path.

Replay identities are meaningful only within one immutable frozen capture.
Fresh episode IDs are the actual EpisodeCollector output columns (each column
is one episode, including multiple collection iterations); time is row index.
EA/RL denotes physical producer, not a source/reference role: relabelling
RL-own as RL-fresh cannot make the same transition independent.

These eager, host-side gates do not evaluate Utility or collect environments.
Callers must keep the returned provenance and use a unique run_id per run.
Cross fresh/replay comparisons require collection/insertion ordering evidence
from the workflow hooks; differing identity namespaces alone prove nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from evorl.replay_buffers.replay_buffer import ReplayBuffer, ReplayBufferState
from evorl.utils.jax_utils import right_shift_with_padding, tree_get

PILOT_REPLAY_NAMESPACE = 0x50314152


@dataclass(frozen=True)
class ReplayIdentity:
    snapshot_id: str
    slot: int


@dataclass(frozen=True)
class FreshIdentity:
    run_id: str
    generation: int
    source_type: str
    actor_id: int
    episode_id: int
    time_index: int
    capture_id: str


@dataclass(frozen=True)
class FreshCapture:
    """Retain the actual immutable collector output, not a role-labelled copy.

    Object identity is a process-local capture token. Retaining raw prevents
    address reuse during this diagnostic. It is not a content hash or PRNG draw;
    selected transition identity additionally includes actor/episode/time.
    Only one raw capture per producer/generation is accepted by a gate.
    """

    raw: object
    run_id: str
    generation: int
    producer: str

    @property
    def capture_id(self):
        return (f"{self.run_id}/checkpoint/{self.generation}/"
                f"{self.producer}/raw_capture/{id(self.raw):x}")


@dataclass(frozen=True)
class ReplaySnapshot:
    state: ReplayBufferState
    run_id: str
    generation: int
    snapshot_id: str
    state_identity: int


@dataclass(frozen=True)
class IdentityGate:
    eligible: bool = True
    ineligible_reason: str | None = None
    snapshot_id: str | None = None
    overlap_count: int | None = None
    valid_sample_count: int = 0

    @property
    def measurement_eligible(self):
        return self.eligible

    @property
    def comparison_eligible(self):
        return self.eligible


@dataclass(frozen=True)
class IdentityBatch(IdentityGate):
    batch: object = None
    identities: tuple[ReplayIdentity | FreshIdentity, ...] = ()
    selected_indices: tuple[int, ...] = ()
    replay_snapshot: ReplaySnapshot | None = None
    fresh_capture: FreshCapture | None = None


@dataclass(frozen=True)
class ReplayGroups(IdentityGate):
    references: tuple[IdentityBatch, ...] = ()
    controls: tuple[IdentityBatch, ...] = ()
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionOrdering:
    """Hook event ordinals, not wall time or inferred content equality.

    replay_insert_event is the insertion of THIS fresh trajectory, not an
    unrelated earlier replay write. One evidence object covers one fresh
    producer capture; callers must check separate captures separately.
    """

    run_id: str
    generation: int
    snapshot_id: str
    replay_capture_event: int
    fresh_capture_event: int
    replay_insert_event: int
    fresh_capture_id: str | None = None


def freeze_replay_snapshot(state, *, run_id, generation):
    """Retain the actual immutable state, scoped by run/checkpoint/capture."""
    if not isinstance(run_id, str) or not run_id.strip() or generation < 0:
        raise ValueError("a nonempty unique run_id and generation are required")
    if not isinstance(state, ReplayBufferState):
        raise TypeError("snapshot requires the actual ReplayBufferState")
    # UUID is capture identity, NOT an insertion version and NOT baseline RNG.
    return ReplaySnapshot(
        state=state, state_identity=id(state), run_id=run_id, generation=int(generation),
        snapshot_id=f"{run_id}/checkpoint/{generation}/capture/{uuid.uuid4().hex}",
    )


def derive_pilot_sampling_key(baseline_key, *, generation):
    key = jax.random.fold_in(baseline_key, PILOT_REPLAY_NAMESPACE)
    return jax.random.fold_in(key, generation)


def sample_disjoint_replay_groups(
    buffer, snapshot, key, *, batch_size=256, num_references=4, num_controls=8,
):
    """Joint uniform draw without replacement, split in a fixed group order.

    Only uniform ReplayBuffer is supported. No data-dependent selection,
    retries, partial batches, or changes to baseline sampling are performed.
    The frozen state is read now; later live replay replacements are irrelevant.
    """
    if type(buffer) is not ReplayBuffer:
        raise TypeError("Pilot sampler requires uniform ReplayBuffer")
    if any(type(value) is not int or value <= 0
           for value in (batch_size, num_references, num_controls)):
        raise ValueError("group counts and batch_size must be positive integers")
    size = int(jax.device_get(snapshot.state.buffer_size))
    leaves = jtu.tree_leaves(snapshot.state.data)
    if not leaves or any(leaf.shape[0] != buffer.capacity for leaf in leaves):
        raise ValueError("snapshot data does not match buffer capacity")
    if not 0 <= size <= buffer.capacity:
        raise ValueError("invalid replay size")
    required = batch_size * (num_references + num_controls)
    provenance = {
        "snapshot_id": snapshot.snapshot_id,
        "run_id": snapshot.run_id,
        "generation": snapshot.generation,
        "buffer_size": size,
        "reference_slot_indices": [],
        "replay_source_slot_indices": [],
        "overlap_count": None,
        "sampling_namespace": "pilot_joint_uniform_without_replacement",
        "sampling_namespace_tag": PILOT_REPLAY_NAMESPACE,
        "sampling_key": np.asarray(jax.random.key_data(key)).tolist(),
    }
    reason = None
    if size == 0 or not bool(jax.device_get(buffer.can_sample(snapshot.state))):
        reason = "replay_not_ready"
    elif size < required:
        reason = "insufficient_disjoint_replay_slots"
    if reason:
        return ReplayGroups(
            eligible=False, ineligible_reason=reason,
            snapshot_id=snapshot.snapshot_id, valid_sample_count=size,
            provenance=provenance,
        )

    slots = jax.random.choice(key, size, (required,), replace=False)
    host_slots = np.asarray(jax.device_get(slots))
    groups = []
    for offset in range(0, required, batch_size):
        indices = slots[offset:offset + batch_size]
        positions = tuple(map(int, host_slots[offset:offset + batch_size]))
        groups.append(IdentityBatch(
            batch=tree_get(snapshot.state.data, indices),
            identities=tuple(ReplayIdentity(snapshot.snapshot_id, i) for i in positions),
            selected_indices=positions, snapshot_id=snapshot.snapshot_id,
            overlap_count=0, valid_sample_count=batch_size,
            replay_snapshot=snapshot,
        ))
    references = tuple(groups[:num_references])
    controls = tuple(groups[num_references:])
    gate = check_disjointness(controls, references, batch_size=batch_size)
    # Also verify every control against every other control.
    all_ids = [i for group in groups for i in group.identities]
    duplicates = len(all_ids) - len(set(all_ids))
    if not gate.eligible or duplicates:
        return ReplayGroups(
            eligible=False, ineligible_reason="source_reference_overlap",
            snapshot_id=snapshot.snapshot_id, overlap_count=duplicates,
            valid_sample_count=size, provenance=provenance,
        )
    provenance.update(
        reference_slot_indices=[list(g.selected_indices) for g in references],
        replay_source_slot_indices=[list(g.selected_indices) for g in controls],
        overlap_count=0,
    )
    return ReplayGroups(
        snapshot_id=snapshot.snapshot_id, overlap_count=0,
        valid_sample_count=required, references=references, controls=controls,
        provenance=provenance,
    )


def sample_fresh_source(
    raw, *, run_id, generation, producer, actor_id, batch_size, key,
):
    """Select exactly the legacy EA or RL source positions with real identity.

    No-autoreset EpisodeCollector dones must stay true after termination.
    This validates layout without changing the sealed right-shift mask.
    EA uses the legacy per-actor fold_in key; RL uses its original key.
    """
    if producer not in ("EA", "RL"):
        raise ValueError("producer must be physical EA or RL, not a role label")
    if not isinstance(run_id, str) or not run_id.strip() or generation < 0:
        raise ValueError("a nonempty unique run_id and generation are required")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    dones = raw.dones
    leaves = jtu.tree_leaves(raw)
    if (dones is None or dones.ndim != 3
            or any(leaf.shape[:3] != dones.shape for leaf in leaves)
            or any(dim == 0 for dim in dones.shape)):
        return IdentityBatch(eligible=False, ineligible_reason="invalid_transition_mask")
    host_dones = np.asarray(jax.device_get(dones))
    if (host_dones.dtype.kind not in "biuf"
            or not np.all(np.isfinite(host_dones))
            or not np.all((host_dones == 0) | (host_dones == 1))):
        return IdentityBatch(eligible=False, ineligible_reason="invalid_transition_mask")
    done_mask = host_dones.astype(bool)
    if np.any(done_mask[:, :-1, :] & ~done_mask[:, 1:, :]):
        return IdentityBatch(eligible=False, ineligible_reason="invalid_transition_mask")
    actors, time, episodes = dones.shape
    if producer == "EA" and (type(actor_id) is not int or not 0 <= actor_id < actors):
        raise ValueError("EA actor_id must identify a real rollout actor")
    if producer == "RL" and actor_id is not None:
        raise ValueError("legacy RL-own selects across all real RL agents")
    valid_mask = jax.vmap(
        lambda d: jnp.logical_not(right_shift_with_padding(d, 1))
    )(dones)
    mask = np.asarray(jax.device_get(valid_mask))
    if producer == "EA":
        valid = np.flatnonzero(mask[actor_id].reshape(-1))
        sample_key = jax.random.fold_in(key, jnp.uint32(actor_id))
        base_offset = actor_id * time * episodes
    else:
        valid = np.flatnonzero(mask.reshape(-1))
        sample_key = key
        base_offset = 0
    count = int(valid.size)
    if count < batch_size:
        return IdentityBatch(
            eligible=False, ineligible_reason="insufficient_fresh_samples",
            valid_sample_count=count,
        )
    chosen = jax.random.choice(sample_key, jnp.asarray(valid), (batch_size,), replace=False)
    positions = chosen + base_offset
    host_positions = tuple(map(int, np.asarray(jax.device_get(positions))))
    capture = FreshCapture(raw, run_id, int(generation), producer)
    identities = []
    for position in host_positions:
        actor, remainder = divmod(position, time * episodes)
        timestep, episode = divmod(remainder, episodes)
        identities.append(FreshIdentity(
            run_id, int(generation), producer, actor, episode, timestep, capture.capture_id,
        ))
    batch = jtu.tree_map(
        lambda leaf: leaf.reshape((-1, *leaf.shape[3:]))[positions], raw,
    )
    return IdentityBatch(
        batch=batch, identities=tuple(identities),
        selected_indices=host_positions, valid_sample_count=count, overlap_count=0,
        fresh_capture=capture,
    )


def check_disjointness(sources, references, *, batch_size, ordering=None):
    """Fail closed before any future Pilot comparison.

    Check each source against all references, duplicates within each batch,
    and duplicates between references. Sources need not be mutually disjoint
    (except the joint Replay sampler enforces that additional control).
    Reference-self diagonal is intentionally not a held-out comparison.
    """
    sources, references = tuple(sources), tuple(references)
    groups = sources + references
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    def hold(reason, overlap=None):
        return IdentityGate(
            eligible=False, ineligible_reason=reason, overlap_count=overlap,
        )

    if not sources or not references:
        return hold("missing_transition_identity")
    for group in groups:
        if not group.eligible:
            return hold(group.ineligible_reason or "missing_transition_identity")
        if group.batch is None:
            return hold("missing_transition_identity")
        leaves = jtu.tree_leaves(group.batch)
        if (not leaves or any(leaf.ndim == 0 or leaf.shape[0] != batch_size for leaf in leaves)
                or group.valid_sample_count < batch_size):
            return hold("insufficient_fresh_samples" if any(
                isinstance(i, FreshIdentity) for i in group.identities
            ) else "insufficient_disjoint_replay_slots")
        if len(group.identities) != batch_size:
            return hold("missing_transition_identity")
        if any(not isinstance(i, (ReplayIdentity, FreshIdentity)) for i in group.identities):
            return hold("missing_transition_identity")

    replay_scopes = set()
    replay_state = None
    fresh_captures = {}
    physical_producers = {}
    for group in groups:
        if any(isinstance(i, ReplayIdentity) for i in group.identities):
            capture = group.replay_snapshot
            if capture is None:
                return hold("missing_transition_identity")
            if (id(capture.state) != capture.state_identity
                    or group.snapshot_id != capture.snapshot_id or any(
                    not isinstance(i, ReplayIdentity) or i.snapshot_id != capture.snapshot_id
                    for i in group.identities)):
                return hold("snapshot_mismatch")
            if any(i.slot >= int(jax.device_get(capture.state.buffer_size))
                   for i in group.identities):
                return hold("missing_transition_identity")
            if replay_state is not None and capture.state is not replay_state:
                return hold("snapshot_mismatch")
            replay_state = capture.state
            replay_scopes.add((capture.run_id, capture.generation))
        else:
            capture = group.fresh_capture
            if capture is None:
                return hold("missing_transition_identity")
            scope = (capture.run_id, capture.generation, capture.producer)
            if scope in fresh_captures and fresh_captures[scope].raw is not capture.raw:
                return hold("snapshot_mismatch")
            fresh_captures[scope] = capture
            physical = id(capture.raw)
            if physical in physical_producers and physical_producers[physical] != capture.producer:
                return hold("missing_transition_identity")
            physical_producers[physical] = capture.producer
            if any((i.run_id, i.generation, i.source_type, i.capture_id) != (
                    capture.run_id, capture.generation, capture.producer, capture.capture_id)
                   for i in group.identities):
                return hold("snapshot_mismatch")

    replay_ids = [i for g in groups for i in g.identities if isinstance(i, ReplayIdentity)]
    fresh_ids = [i for g in groups for i in g.identities if isinstance(i, FreshIdentity)]
    snapshots = {i.snapshot_id for i in replay_ids}
    scopes = {(i.run_id, i.generation) for i in fresh_ids}
    if len(snapshots) > 1 or len(scopes) > 1:
        return hold("snapshot_mismatch")
    if any(not i.snapshot_id or i.slot < 0 for i in replay_ids):
        return hold("missing_transition_identity")
    if any(not i.run_id or i.generation < 0 or i.actor_id < 0 or i.episode_id < 0
           or i.time_index < 0 or i.source_type not in ("EA", "RL") for i in fresh_ids):
        return hold("missing_transition_identity")
    if replay_ids and fresh_ids:
        if ordering is None or len({i.source_type for i in fresh_ids}) != 1:
            return hold("missing_transition_identity")
        if (snapshots != {ordering.snapshot_id}
                or scopes != {(ordering.run_id, ordering.generation)}
                or replay_scopes != scopes
                or {i.capture_id for i in fresh_ids} != {ordering.fresh_capture_id}):
            return hold("snapshot_mismatch")
        if not (ordering.replay_capture_event < ordering.fresh_capture_event
                < ordering.replay_insert_event):
            return hold("source_reference_overlap")

    # This is an integrity read-back, not content-based transition identity.
    # Deduplication below uses scoped physical positions, even if data are equal.
    for group in groups:
        positions = group.selected_indices
        if len(positions) != batch_size or any(type(i) is not int for i in positions):
            return hold("missing_transition_identity")
        if group.replay_snapshot is not None:
            if tuple(i.slot for i in group.identities) != positions:
                return hold("missing_transition_identity")
            actual = tree_get(group.replay_snapshot.state.data, jnp.asarray(positions))
        else:
            capture = group.fresh_capture
            raw = capture.raw
            shape = raw.dones.shape
            if any(i < 0 or i >= int(np.prod(shape)) for i in positions):
                return hold("missing_transition_identity")
            expected = []
            for position in positions:
                actor, remainder = divmod(position, shape[1] * shape[2])
                time_index, episode = divmod(remainder, shape[2])
                expected.append(FreshIdentity(
                    capture.run_id, capture.generation, capture.producer,
                    actor, episode, time_index, capture.capture_id,
                ))
            if tuple(expected) != group.identities:
                return hold("missing_transition_identity")
            valid = np.asarray(jax.device_get(jax.vmap(
                lambda d: jnp.logical_not(right_shift_with_padding(d, 1))
            )(raw.dones))).reshape(-1)
            if not np.all(valid[np.asarray(positions)]):
                return hold("invalid_transition_mask")
            actual = jtu.tree_map(
                lambda leaf: leaf.reshape((-1, *leaf.shape[3:]))[jnp.asarray(positions)],
                raw,
            )
        if jtu.tree_structure(group.batch) != jtu.tree_structure(actual):
            return hold("missing_transition_identity")
        if any(not np.array_equal(np.asarray(jax.device_get(a)),
                                  np.asarray(jax.device_get(b)), equal_nan=True)
               for a, b in zip(jtu.tree_leaves(group.batch), jtu.tree_leaves(actual), strict=True)):
            return hold("missing_transition_identity")

    overlap = sum(len(g.identities) - len(set(g.identities)) for g in groups)
    reference_ids = [i for g in references for i in g.identities]
    overlap += len(reference_ids) - len(set(reference_ids))
    reference_set = set(reference_ids)
    overlap += sum(len(set(g.identities) & reference_set) for g in sources)
    if overlap:
        return hold("source_reference_overlap", overlap)
    # Held-out EA donors must be different actors, even with disjoint times.
    candidate_actors = {i.actor_id for g in sources for i in g.identities
                        if isinstance(i, FreshIdentity) and i.source_type == "EA"}
    donor_actors = {i.actor_id for g in references for i in g.identities
                   if isinstance(i, FreshIdentity) and i.source_type == "EA"}
    if candidate_actors & donor_actors:
        return hold("source_reference_overlap", 0)
    return IdentityGate(
        snapshot_id=next(iter(snapshots), None), overlap_count=0,
        valid_sample_count=sum(batch_size for _ in groups),
    )
