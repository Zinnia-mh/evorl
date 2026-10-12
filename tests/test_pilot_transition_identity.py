"""P1-A contracts: real slots, frozen scopes, fresh positions and baseline parity."""

import importlib
import importlib.util
from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.algorithms.erl.erl_td3.learner_utility import build_source_batches
from evorl.algorithms.erl.erl_td3.utility_source_controls import (
    build_rl_own_source_batch,
)


def pilot():
    name = "evorl.algorithms.erl.erl_td3.pilot_transition_identity"
    assert importlib.util.find_spec(name) is not None, "P1-A identity module missing"
    return importlib.import_module(name)


def replay(size=40, capacity=48, batch_size=2):
    buffer = ReplayBuffer(capacity=capacity, sample_batch_size=batch_size)
    state = buffer.init(SampleBatch(obs=jnp.zeros((1,), dtype=jnp.int32)))
    state = buffer.add(state, SampleBatch(obs=jnp.arange(size)[:, None]))
    return buffer, state


@pytest.mark.parametrize("size", [5, 48, 53])
@pytest.mark.parametrize("compiled", [False, True])
def test_replay_indices_match_original_sample_and_physical_data(size, compiled):
    buffer, state = replay(size=size, batch_size=16)
    key = jax.random.key(11)
    assert hasattr(buffer, "sample_with_indices"), "Replay slot interface missing"
    sample = buffer.sample_with_indices
    if compiled:
        sample = jax.jit(sample)
    batch, indices = sample(state, key)
    chex.assert_trees_all_equal(batch, buffer.sample(state, key))
    np.testing.assert_array_equal(batch.obs, state.data.obs[indices])
    assert np.all(np.asarray(indices) >= 0)
    assert np.all(np.asarray(indices) < min(size, 48))


def test_masked_and_wrapped_add_preserves_original_state():
    buffer, state = replay(size=4, capacity=6)
    state = buffer.add(state, SampleBatch(obs=jnp.array([[10], [11], [12], [13]])),
                       jnp.array([True, False, True, True]))
    np.testing.assert_array_equal(state.data.obs[:, 0], [13, 1, 2, 3, 10, 12])
    assert int(state.current_index) == 1
    assert int(state.buffer_size) == 6
    state = buffer.add(state, SampleBatch(obs=jnp.array([[20], [21]])))
    assert hasattr(buffer, "sample_with_indices"), "Replay slot interface missing"
    before = jax.tree_util.tree_map(lambda x: x.copy(), state)
    buffer.sample_with_indices(state, jax.random.key(4))
    chex.assert_trees_all_equal(state, before)
    np.testing.assert_array_equal(state.data.obs[:, 0], [13, 20, 21, 3, 10, 12])
    assert int(state.current_index) == 3
    assert int(state.buffer_size) == 6


def groups(size=40, capacity=48, batch_size=2, refs=4, controls=8):
    p = pilot()
    buffer, state = replay(size, capacity, batch_size)
    snapshot = p.freeze_replay_snapshot(state, run_id="run-a", generation=7)
    result = p.sample_disjoint_replay_groups(
        buffer, snapshot, jax.random.key(12),
        batch_size=batch_size, num_references=refs, num_controls=controls,
    )
    return p, buffer, state, snapshot, result


@pytest.mark.parametrize("size,capacity,batch_size", [(40, 48, 2), (3072, 4096, 256), (53, 48, 2)])
def test_disjoint_groups_are_exact_real_slots_and_reproducible(size, capacity, batch_size):
    p, buffer, state, snapshot, result = groups(size, capacity, batch_size)
    assert result.eligible and result.measurement_eligible and result.comparison_eligible
    assert len(result.references) == 4 and len(result.controls) == 8
    indices = []
    for group in result.references + result.controls:
        slots = [identity.slot for identity in group.identities]
        assert len(set(slots)) == batch_size
        assert all(identity.snapshot_id == snapshot.snapshot_id for identity in group.identities)
        np.testing.assert_array_equal(group.batch.obs, state.data.obs[jnp.array(slots)])
        indices.extend(slots)
    assert len(set(indices)) == 12 * batch_size
    assert result.overlap_count == 0
    again = p.sample_disjoint_replay_groups(
        buffer, snapshot, jax.random.key(12), batch_size=batch_size,
    )
    assert [g.identities for g in again.references + again.controls] == [
        g.identities for g in result.references + result.controls
    ]
    assert result.provenance["buffer_size"] == min(size, capacity)
    assert result.provenance["snapshot_id"] == snapshot.snapshot_id
    assert len(result.provenance["reference_slot_indices"]) == 4
    assert len(result.provenance["replay_source_slot_indices"]) == 8
    assert result.provenance["overlap_count"] == 0
    assert result.provenance["sampling_namespace"]


@pytest.mark.parametrize("size,reason", [
    (0, "replay_not_ready"),
    (23, "insufficient_disjoint_replay_slots"),
])
def test_replay_shortage_fails_closed_without_partial_groups(size, reason):
    _, _, _, _, result = groups(size=size)
    assert not result.eligible
    assert not result.measurement_eligible and not result.comparison_eligible
    assert result.ineligible_reason == reason
    assert result.references == result.controls == ()
    assert result.valid_sample_count == size


def test_replay_overlap_missing_identity_and_snapshot_mismatch():
    p, _, _, snapshot, result = groups()
    source = result.controls[0]
    reference = result.references[0]
    assert p.check_disjointness((source,), (reference,), batch_size=2).eligible
    slots = (source.identities[0].slot, reference.identities[1].slot)
    overlap = replace(reference,
        identities=(source.identities[0], reference.identities[1]),
        selected_indices=slots,
        batch=SampleBatch(obs=snapshot.state.data.obs[jnp.array(slots)]),
    )
    gate = p.check_disjointness((source,), (overlap,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "source_reference_overlap"
    assert gate.overlap_count == 1
    missing = replace(source, identities=())
    assert p.check_disjointness((missing,), (reference,), batch_size=2).ineligible_reason == "missing_transition_identity"
    another = replace(reference, identities=tuple(
        replace(identity, snapshot_id="other-snapshot") for identity in reference.identities
    ))
    assert p.check_disjointness((source,), (another,), batch_size=2).ineligible_reason == "snapshot_mismatch"
    second_snapshot = p.freeze_replay_snapshot(snapshot.state, run_id="run-a", generation=7)
    assert second_snapshot.snapshot_id != snapshot.snapshot_id


def test_joint_gate_checks_duplicates_between_and_within_reference_groups():
    p, _, _, _, result = groups()
    source = result.controls[0]
    ref = result.references[0]
    duplicate = replace(ref, identities=(ref.identities[0], ref.identities[0]))
    assert not p.check_disjointness((source,), (duplicate,), batch_size=2).eligible
    assert not p.check_disjointness((source,), (ref, ref), batch_size=2).eligible


def raw(actors=2, time=4, episodes=2):
    markers = jnp.arange(actors * time * episodes, dtype=jnp.float32).reshape(
        actors, time, episodes
    )
    return SampleBatch(obs=markers[..., None], rewards=markers,
                       dones=jnp.zeros(markers.shape, dtype=jnp.bool_))


@pytest.mark.parametrize("producer", ["EA", "RL"])
def test_fresh_sampler_matches_legacy_selected_positions(producer):
    p = pilot()
    trajectory = raw()
    key = jax.random.key(17)
    selected = p.sample_fresh_source(
        trajectory, run_id="run-a", generation=7, producer=producer,
        actor_id=1 if producer == "EA" else None, batch_size=4, key=key,
    )
    legacy = (
        build_source_batches(trajectory, generation=7, source_batch_size=4, key=key)[1]
        if producer == "EA" else
        build_rl_own_source_batch(trajectory, generation=7, source_batch_size=4, key=key)
    )
    chex.assert_trees_all_equal(selected.batch, legacy.batch)
    assert selected.eligible and selected.valid_sample_count == legacy.valid_transition_count
    for row, identity in zip(selected.batch.obs, selected.identities, strict=True):
        assert float(row[0]) == float(trajectory.obs[
            identity.actor_id, identity.time_index, identity.episode_id, 0
        ])
        assert identity.run_id == "run-a" and identity.generation == 7
        assert identity.source_type == producer


def test_fresh_actor_episode_time_and_scope_are_distinct():
    p = pilot()
    selected = p.sample_fresh_source(
        raw(), run_id="run-a", generation=7, producer="RL",
        actor_id=None, batch_size=16, key=jax.random.key(2),
    )
    assert len(set(selected.identities)) == 16
    assert {(i.actor_id, i.episode_id, i.time_index) for i in selected.identities} == {
        (actor, episode, time) for actor in (0, 1) for episode in (0, 1) for time in range(4)
    }


def test_fresh_shifted_mask_and_shortage():
    p = pilot()
    trajectory = raw(actors=1, time=4, episodes=1).replace(
        dones=jnp.array([[[False], [True], [True], [True]]])
    )
    source = p.sample_fresh_source(
        trajectory, run_id="r", generation=1, producer="RL", actor_id=None,
        batch_size=2, key=jax.random.key(1),
    )
    assert {i.time_index for i in source.identities} == {0, 1}
    short = p.sample_fresh_source(
        trajectory, run_id="r", generation=1, producer="RL", actor_id=None,
        batch_size=3, key=jax.random.key(1),
    )
    assert not short.eligible and short.ineligible_reason == "insufficient_fresh_samples"
    assert short.batch is None and short.identities == ()
    assert short.valid_sample_count == 2


@pytest.mark.parametrize("dones", [
    jnp.full((1, 4, 1), 0.5),                  # non-binary numeric mask
    jnp.array([[[True], [False], [False], [False]]]),  # non-absorbing no-autoreset layout
])
def test_invalid_transition_mask_fails_closed(dones):
    p = pilot()
    selected = p.sample_fresh_source(
        raw(1, 4, 1).replace(dones=dones), run_id="r", generation=1,
        producer="RL", actor_id=None, batch_size=2, key=jax.random.key(1),
    )
    assert not selected.eligible
    assert selected.ineligible_reason == "invalid_transition_mask"


def test_fresh_overlap_actor_donor_and_scope_gates():
    p = pilot()
    trajectory = raw()
    source = p.sample_fresh_source(trajectory, run_id="r", generation=1,
        producer="EA", actor_id=0, batch_size=2, key=jax.random.key(1))
    donor = p.sample_fresh_source(trajectory, run_id="r", generation=1,
        producer="EA", actor_id=1, batch_size=2, key=jax.random.key(1))
    assert p.check_disjointness((source,), (donor,), batch_size=2).eligible
    assert p.check_disjointness((source,), (source,), batch_size=2).ineligible_reason == "source_reference_overlap"
    # A real split from one EA actor stays prohibited even with disjoint times.
    full = p.sample_fresh_source(trajectory, run_id="r", generation=1,
        producer="EA", actor_id=0, batch_size=8, key=jax.random.key(1))
    candidate, same_actor = subset(full, 0, 2), subset(full, 2, 4)
    assert not set(candidate.identities) & set(same_actor.identities)
    assert not p.check_disjointness((candidate,), (same_actor,), batch_size=2).eligible
    other_run = replace(donor, identities=tuple(replace(i, run_id="other") for i in donor.identities))
    assert p.check_disjointness((source,), (other_run,), batch_size=2).ineligible_reason == "snapshot_mismatch"


def subset(group, start, end):
    return replace(group,
        batch=jax.tree_util.tree_map(lambda leaf: leaf[start:end], group.batch),
        identities=group.identities[start:end],
        selected_indices=group.selected_indices[start:end],
    )


def test_rl_heldout_times_use_exact_identity_intersection():
    p = pilot()
    full = p.sample_fresh_source(raw(1, 8, 1), run_id="r", generation=1,
        producer="RL", actor_id=None, batch_size=8, key=jax.random.key(1))
    source, ref = subset(full, 0, 2), subset(full, 2, 4)
    assert p.check_disjointness((source,), (ref,), batch_size=2).eligible
    overlap = subset(full, 1, 3)
    gate = p.check_disjointness((source,), (overlap,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "source_reference_overlap"


def test_mixed_namespaces_need_temporal_ordering_evidence():
    p, _, _, snapshot, result = groups()
    fresh = p.sample_fresh_source(raw(), run_id="run-a", generation=7,
        producer="EA", actor_id=0, batch_size=2, key=jax.random.key(1))
    refs = result.references
    assert not p.check_disjointness((fresh,), refs, batch_size=2).eligible
    ordering = p.CollectionOrdering(
        run_id="run-a", generation=7, snapshot_id=snapshot.snapshot_id,
        replay_capture_event=0, fresh_capture_event=1, replay_insert_event=2,
        fresh_capture_id=fresh.fresh_capture.capture_id,
    )
    assert p.check_disjointness((fresh,), refs, batch_size=2, ordering=ordering).eligible
    invalid = replace(ordering, replay_capture_event=3)
    assert not p.check_disjointness((fresh,), refs, batch_size=2, ordering=invalid).eligible


def test_disjoint_sampling_does_not_mutate_baseline_state_key_or_normal_sample():
    p, buffer, state, snapshot, _ = groups()
    baseline_key = jax.random.key(31)
    before_state = jax.tree_util.tree_map(lambda x: x.copy(), state)
    before_key = jax.random.key_data(baseline_key).copy()
    before_batch = buffer.sample(state, baseline_key)
    diagnostic_key = p.derive_pilot_sampling_key(baseline_key, generation=7)
    assert not np.array_equal(jax.random.key_data(diagnostic_key), before_key)
    p.sample_disjoint_replay_groups(buffer, snapshot, diagnostic_key, batch_size=2)
    chex.assert_trees_all_equal(state, before_state)
    np.testing.assert_array_equal(jax.random.key_data(baseline_key), before_key)
    chex.assert_trees_all_equal(buffer.sample(state, baseline_key), before_batch)


def test_gate_rejects_identity_count_not_matching_batch():
    p, _, _, _, result = groups()
    source = replace(result.controls[0], identities=result.controls[0].identities[:1])
    gate = p.check_disjointness((source,), result.references, batch_size=2)
    assert not gate.measurement_eligible and not gate.comparison_eligible
    assert gate.ineligible_reason == "missing_transition_identity"


def test_replay_group_cannot_relabel_its_frozen_snapshot():
    p, _, _, _, result = groups()
    original = result.references[0]
    relabelled = replace(
        original, snapshot_id="forged",
        identities=tuple(replace(i, snapshot_id="forged") for i in original.identities),
    )
    assert p.check_disjointness(
        (relabelled,), (relabelled,), batch_size=2
    ).ineligible_reason == "snapshot_mismatch"


def test_temporal_evidence_cannot_relabel_replay_run_scope():
    p, _, _, snapshot, result = groups()
    fresh = p.sample_fresh_source(raw(), run_id="other-run", generation=7,
        producer="RL", actor_id=None, batch_size=2, key=jax.random.key(1))
    wrong = p.CollectionOrdering(
        run_id="other-run", generation=7, snapshot_id=snapshot.snapshot_id,
        replay_capture_event=0, fresh_capture_event=1, replay_insert_event=2,
        fresh_capture_id=fresh.fresh_capture.capture_id,
    )
    gate = p.check_disjointness((fresh,), result.references, batch_size=2, ordering=wrong)
    assert not gate.eligible and gate.ineligible_reason == "snapshot_mismatch"


def test_one_ordering_record_does_not_prove_two_producer_captures():
    p, _, _, snapshot, result = groups()
    ea = p.sample_fresh_source(raw(), run_id="run-a", generation=7,
        producer="EA", actor_id=0, batch_size=2, key=jax.random.key(1))
    rl = p.sample_fresh_source(raw(), run_id="run-a", generation=7,
        producer="RL", actor_id=None, batch_size=2, key=jax.random.key(2))
    ordering = p.CollectionOrdering(
        run_id="run-a", generation=7, snapshot_id=snapshot.snapshot_id,
        replay_capture_event=0, fresh_capture_event=1, replay_insert_event=2,
        fresh_capture_id=ea.fresh_capture.capture_id,
    )
    gate = p.check_disjointness((ea, rl), result.references, batch_size=2, ordering=ordering)
    assert not gate.eligible and gate.ineligible_reason == "missing_transition_identity"


def test_indexed_sampler_does_not_misrepresent_prioritized_sampling():
    from evorl.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer

    buffer = PrioritizedReplayBuffer(capacity=8, sample_batch_size=2)
    state = buffer.init(SampleBatch(obs=jnp.zeros((1,), dtype=jnp.int32)))
    with pytest.raises(NotImplementedError, match="uniform"):
        buffer.sample_with_indices(state, jax.random.key(1))


def test_replay_data_cannot_be_relabelled_with_disjoint_slot_ids():
    p, _, _, _, result = groups()
    forged = replace(result.references[0], batch=result.controls[0].batch)
    gate = p.check_disjointness((result.controls[0],), (forged,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "missing_transition_identity"


def test_same_snapshot_label_cannot_bind_two_different_captured_states():
    p, buffer, state, snapshot, result = groups()
    later = buffer.add(state, SampleBatch(obs=jnp.array([[100], [101]])))
    forged = replace(result.references[0],
        replay_snapshot=replace(snapshot, state=later))
    gate = p.check_disjointness((result.controls[0],), (forged,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "snapshot_mismatch"


def test_fresh_data_and_times_are_bound_to_the_actual_capture():
    p = pilot()
    full = p.sample_fresh_source(raw(1, 8, 1), run_id="r", generation=1,
        producer="RL", actor_id=None, batch_size=8, key=jax.random.key(1))
    source, ref = subset(full, 0, 2), subset(full, 2, 4)
    forged = replace(ref, batch=source.batch)
    gate = p.check_disjointness((source,), (forged,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "missing_transition_identity"
    forged_time = replace(ref, identities=tuple(
        replace(i, time_index=i.time_index + 8) for i in ref.identities))
    assert not p.check_disjointness((source,), (forged_time,), batch_size=2).eligible


@pytest.mark.parametrize("other_producer", ["EA", "RL"])
def test_ordering_evidence_cannot_authorize_a_different_fresh_capture(other_producer):
    p, _, _, snapshot, result = groups()
    first = p.sample_fresh_source(raw(), run_id="run-a", generation=7,
        producer="EA", actor_id=0, batch_size=2, key=jax.random.key(1))
    other = p.sample_fresh_source(raw(), run_id="run-a", generation=7,
        producer=other_producer, actor_id=0 if other_producer == "EA" else None,
        batch_size=2, key=jax.random.key(1))
    assert hasattr(first, "fresh_capture"), "Fresh capture provenance missing"
    ordering = p.CollectionOrdering(
        run_id="run-a", generation=7, snapshot_id=snapshot.snapshot_id,
        replay_capture_event=0, fresh_capture_event=1, replay_insert_event=2,
        fresh_capture_id=first.fresh_capture.capture_id,
    )
    gate = p.check_disjointness((other,), result.references, batch_size=2, ordering=ordering)
    assert not gate.eligible and gate.ineligible_reason == "snapshot_mismatch"


def test_fresh_ids_are_reproducible_for_the_same_real_capture_and_key():
    p = pilot()
    trajectory = raw()
    kwargs = dict(run_id="r", generation=1, producer="EA",
                  actor_id=0, batch_size=2, key=jax.random.key(1))
    first = p.sample_fresh_source(trajectory, **kwargs)
    second = p.sample_fresh_source(trajectory, **kwargs)
    assert first.identities == second.identities
    assert first.selected_indices == second.selected_indices


def test_fresh_producer_names_cannot_make_the_same_raw_capture_independent():
    p = pilot()
    trajectory = raw(1, 8, 1)
    ea = p.sample_fresh_source(trajectory, run_id="r", generation=1,
        producer="EA", actor_id=0, batch_size=2, key=jax.random.key(1))
    rl = p.sample_fresh_source(trajectory, run_id="r", generation=1,
        producer="RL", actor_id=None, batch_size=2, key=jax.random.key(2))
    gate = p.check_disjointness((ea,), (rl,), batch_size=2)
    assert not gate.eligible


def test_all_replay_groups_cannot_change_state_under_the_same_capture_id():
    p, buffer, state, snapshot, result = groups()
    later = buffer.add(state, SampleBatch(obs=jnp.array([[100], [101]])))
    capture = replace(snapshot, state=later)
    source = replace(result.controls[0], replay_snapshot=capture)
    ref = replace(result.references[0], replay_snapshot=capture)
    gate = p.check_disjointness((source,), (ref,), batch_size=2)
    assert not gate.eligible and gate.ineligible_reason == "snapshot_mismatch"

def test_binary_float_brax_done_mask_matches_legacy_fresh_selection():
    p = pilot()
    trajectory = raw(1, 8, 1).replace(
        dones=jnp.array([[[0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]]])
    )
    key = jax.random.key(17)
    source = p.sample_fresh_source(
        trajectory, run_id="r", generation=11, producer="RL",
        actor_id=None, batch_size=4, key=key,
    )
    legacy = build_rl_own_source_batch(
        trajectory, generation=11, source_batch_size=4, key=key,
    )
    assert source.eligible
    assert source.valid_sample_count == 7
    chex.assert_trees_all_equal(source.batch, legacy.batch)
    assert {i.time_index for i in source.identities} <= set(range(7))

def test_nonfinite_numeric_done_mask_fails_closed():
    p = pilot()
    with jax.debug_nans(False):
        trajectory = raw(1, 4, 1).replace(
            dones=jnp.array([[[0.], [jnp.nan], [1.], [1.]]])
        )
        selected = p.sample_fresh_source(
            trajectory, run_id="r", generation=11, producer="RL",
            actor_id=None, batch_size=2, key=jax.random.key(1),
        )
    assert not selected.eligible
    assert selected.ineligible_reason == "invalid_transition_mask"
