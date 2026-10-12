import jax
import jax.numpy as jnp
import optax
from evorl.algorithms.erl.erl_td3.pilot_dual_reference import evaluate_paired_utility

def test_paired_utility_reuses_one_update_and_common_before():
    optimizer = optax.sgd(0.1)
    def loss(params, batch, key):
        return jnp.mean((params - batch) ** 2)
    result = evaluate_paired_utility(
        loss_fn=loss, optimizer=optimizer, params=jnp.array([1.0]),
        opt_state=optimizer.init(jnp.array([1.0])), source_batch=jnp.array([2.0]),
        references={"Replay": (jnp.array([3.0]),), "RL-fresh": (jnp.array([4.0]),), "EA-fresh": (jnp.array([5.0]),)},
        source_update_key=jax.random.key(1),
        reference_keys={"Replay": (jax.random.key(2),), "RL-fresh": (jax.random.key(3),), "EA-fresh": (jax.random.key(4),)},
        step_scales=(1.0, 0.5, 0.1), tolerance=1e-6,
    )
    assert len(result.records) == 9
    assert len(result.zero_step_results) == 3
    assert result.valid

def test_paired_replay_matches_legacy_and_one_optimizer_step():
    from evorl.algorithms.erl.erl_td3.utility_source_controls import scaled_virtual_critic_utility
    updates = []
    base = optax.sgd(0.2)
    optimizer = optax.GradientTransformation(
        init=base.init,
        update=lambda grads, state, params=None: (updates.append(1) or base.update(grads, state, params)),
    )
    params = jnp.array([1.0])
    opt_state = optimizer.init(params)
    source = jnp.array([2.0])
    replay = jnp.array([3.0])
    update_key = jax.random.key(21)
    reference_key = jax.random.key(22)
    def loss(p, b, key):
        return jnp.mean((p-b)**2) + 0.01 * jax.random.normal(key)
    result = evaluate_paired_utility(
        loss_fn=loss, optimizer=optimizer, params=params, opt_state=opt_state,
        source_batch=source,
        references={"Replay": (replay,), "RL-fresh": (jnp.array([4.0]),), "EA-fresh": (jnp.array([5.0]),)},
        source_update_key=update_key,
        reference_keys={"Replay": (reference_key,), "RL-fresh": (jax.random.key(23),), "EA-fresh": (jax.random.key(24),)},
        step_scales=(1.0, 0.5, 0.1), tolerance=1e-6,
    )
    assert len(updates) == 1
    for record in result.records[:3]:
        legacy = scaled_virtual_critic_utility(
            loss_fn=loss, optimizer=base, params=params, opt_state=opt_state,
            source_batch=source, reference_batch=replay,
            source_update_key=update_key, reference_eval_key=reference_key,
            step_scale=record.step_scale,
        )
        assert abs(record.utility - float(legacy.utility)) < 1e-6
        assert abs(record.loss_ref_before - float(legacy.loss_ref_before)) < 1e-6
    assert all(abs(z.utility) <= 1e-6 for z in result.zero_step_results)


def _raw(actors, time=4, episodes=2):
    from evorl.sample_batch import SampleBatch
    values = jnp.arange(actors*time*episodes, dtype=jnp.float32).reshape(actors,time,episodes)
    return SampleBatch(obs=values[...,None], rewards=values, dones=jnp.zeros(values.shape, dtype=jnp.bool_))


def test_rl_heldout_keeps_legacy_source_and_is_deterministically_disjoint():
    from evorl.algorithms.erl.erl_td3.pilot_transition_identity import sample_fresh_source, check_disjointness
    from evorl.algorithms.erl.erl_td3.pilot_dual_reference import sample_rl_heldout
    from evorl.algorithms.erl.erl_td3.utility_source_controls import build_rl_own_source_batch
    import chex
    raw = _raw(2)
    key = jax.random.key(31)
    own = sample_fresh_source(raw, run_id="r", generation=5, producer="RL", actor_id=None, batch_size=4, key=key)
    legacy = build_rl_own_source_batch(raw, generation=5, source_batch_size=4, key=key)
    chex.assert_trees_all_equal(own.batch, legacy.batch)
    refs = sample_rl_heldout(raw, own, run_id="r", generation=5, batch_size=4, key=jax.random.key(32))
    assert check_disjointness((own,), refs, batch_size=4).eligible
    again = sample_rl_heldout(raw, own, run_id="r", generation=5, batch_size=4, key=jax.random.key(32))
    assert [x.selected_indices for x in refs] == [x.selected_indices for x in again]
    too_large = sample_fresh_source(raw, run_id="r", generation=5, producer="RL", actor_id=None, batch_size=6, key=key)
    assert all(not x.eligible for x in sample_rl_heldout(raw, too_large, run_id="r", generation=5, batch_size=6, key=jax.random.key(32)))


def test_ea_donors_are_distinct_from_top_mid_low_with_tie_break():
    from evorl.algorithms.erl.erl_td3.pilot_dual_reference import choose_ea_candidates_and_donors
    from evorl.algorithms.erl.erl_td3.pilot_transition_identity import check_disjointness
    candidates, donors, reason = choose_ea_candidates_and_donors(
        _raw(5), jnp.array([5.,5.,3.,1.,0.]),
        run_id="r", generation=5, batch_size=2,
        source_key=jax.random.key(41), donor_key=jax.random.key(42),
    )
    assert reason is None
    assert [i for i, _ in candidates] == [0,2,4]
    assert [i for i, _ in donors] == [1,3]
    assert check_disjointness([x for _, x in candidates], [x for _, x in donors], batch_size=2).eligible
    assert choose_ea_candidates_and_donors(
        _raw(4), jnp.array([4.,3.,2.,1.]), run_id="r", generation=5,
        batch_size=2, source_key=jax.random.key(41), donor_key=jax.random.key(42),
    )[2] == "insufficient_ea_donor_actors"


def test_pilot_writer_stays_under_root_and_persists_raw_results(tmp_path, monkeypatch):
    from evorl.algorithms.erl.erl_td3 import pilot_artifacts
    monkeypatch.setattr(pilot_artifacts, "PILOT_ROOT", tmp_path)
    writer = pilot_artifacts.PilotArtifactWriter(tmp_path, "run")
    with __import__("pytest").raises(ValueError):
        pilot_artifacts.PilotArtifactWriter(tmp_path.parent, "run")
    for bad_id in ("..", ".", "../outside", "folder/name", ""):
        with __import__("pytest").raises(ValueError):
            pilot_artifacts.PilotArtifactWriter(tmp_path, bad_id)
    optimizer = optax.sgd(0.1)
    result = evaluate_paired_utility(
        loss_fn=lambda p,b,k: jnp.mean((p-b)**2), optimizer=optimizer,
        params=jnp.array([1.]), opt_state=optimizer.init(jnp.array([1.])),
        source_batch=jnp.array([2.]),
        references={"Replay": (jnp.array([3.]),), "RL-fresh": (jnp.array([4.]),), "EA-fresh": (jnp.array([5.]),)},
        source_update_key=jax.random.key(1),
        reference_keys={"Replay": (jax.random.key(2),), "RL-fresh": (jax.random.key(3),), "EA-fresh": (jax.random.key(4),)},
        step_scales=(1.0,0.5,0.1), tolerance=1e-6,
    )
    writer.append_checkpoint(generation=5, status="PASS", reasons=(), source_results={"s":result}, aggregates={}, provenance={"snapshot_id":"x"})
    assert len((tmp_path/"run"/"reference_level.csv").read_text().splitlines()) == 10
    assert len((tmp_path/"run"/"zero_step.csv").read_text().splitlines()) == 4
    assert '"snapshot_id": "x"' in (tmp_path/"run"/"paired_summary.jsonl").read_text()
