"""P1-C mechanism definitions and matching gates, with synthetic losses only."""
from dataclasses import replace
import csv
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evorl.algorithms.erl.erl_td3.pilot_dual_reference import evaluate_paired_utility
from evorl.algorithms.erl.erl_td3.pilot_gradient_mechanisms import ReferenceGradientCache


def _evaluate(*, params=None, source=None, refs=None, optimizer=None, loss=None,
              scales=(1.0, 0.1), cache=None, keys=None):
    params = jnp.array([0.], dtype=jnp.float32) if params is None else params
    source = jnp.array([1.], dtype=jnp.float32) if source is None else source
    refs = (jnp.array([1.]), jnp.array([-1.]), jnp.array([1.])) if refs is None else refs
    optimizer = optax.sgd(1.) if optimizer is None else optimizer
    loss = (lambda p,b,k: jnp.sum((p-b)**2)) if loss is None else loss
    keys = (jax.random.key(10), jax.random.key(11), jax.random.key(12)) if keys is None else keys
    references = {"Replay": (refs[0],), "RL-fresh": (refs[1],), "EA-fresh": (refs[2],)}
    reference_keys = {"Replay": (keys[0],), "RL-fresh": (keys[1],), "EA-fresh": (keys[2],)}
    if cache is None:
        cache = ReferenceGradientCache(
            checkpoint=5, frozen_learner_fingerprint="frozen-learner",
            frozen_critic_fingerprint="frozen-critic", tolerance=1e-6,
        )
    result = evaluate_paired_utility(
        loss_fn=loss, optimizer=optimizer, params=params,
        opt_state=optimizer.init(params), source_batch=source,
        references=references, source_update_key=jax.random.key(9),
        reference_keys=reference_keys, step_scales=scales,
        tolerance=1e-6, mechanism_cache=cache,
    )
    return result, cache


def _record(result, family, scale):
    return next(x for x in result.mechanism_records
                if x.reference_family == family and x.step_scale == scale)


def test_quadratic_alignment_conflict_first_order_and_overshoot():
    result, cache = _evaluate()
    assert result.valid
    assert len(result.mechanism_records) == len(result.records) + len(result.zero_step_results)
    aligned = _record(result, "Replay", 1.0)
    assert aligned.source_gradient_norm == pytest.approx(2.)
    assert aligned.reference_gradient_norm == pytest.approx(2.)
    assert aligned.raw_gradient_cosine == pytest.approx(1.)
    assert aligned.negative_update_reference_cosine == pytest.approx(1.)
    assert aligned.optax_update_norm == pytest.approx(2.)
    assert aligned.actual_parameter_displacement_norm == pytest.approx(2.)
    assert aligned.first_order_predicted_utility == pytest.approx(4.)
    assert aligned.actual_utility == pytest.approx(0.)
    assert aligned.first_order_residual == pytest.approx(-4.)
    small = _record(result, "Replay", 0.1)
    assert small.actual_parameter_displacement_norm == pytest.approx(.2, abs=1e-6)
    assert small.first_order_predicted_utility == pytest.approx(.4, abs=1e-6)
    assert small.actual_utility == pytest.approx(.36, abs=1e-6)
    conflict = _record(result, "RL-fresh", 1.0)
    assert conflict.raw_gradient_cosine == pytest.approx(-1.)
    assert conflict.negative_update_reference_cosine == pytest.approx(-1.)
    assert conflict.first_order_predicted_utility == pytest.approx(-4.)
    assert conflict.actual_utility == pytest.approx(-8.)
    assert len(cache.entries) == 3


def test_reference_gradients_cached_across_sources_with_same_key_and_before():
    cache = ReferenceGradientCache(
        checkpoint=5, frozen_learner_fingerprint="frozen-learner",
        frozen_critic_fingerprint="frozen-critic", tolerance=1e-6,
    )
    real_grad = jax.grad
    calls = []
    def count_grad(fn):
        calls.append(1)
        return real_grad(fn)
    with patch("jax.grad", side_effect=count_grad):
        first, _ = _evaluate(cache=cache)
        second, _ = _evaluate(source=jnp.array([2.]), cache=cache)
    assert len(calls) == 5  # two source gradients and three reference gradients
    assert len(cache.entries) == 3
    assert first.update_fingerprint != second.update_fingerprint
    for family in ("Replay", "RL-fresh", "EA-fresh"):
        a = _record(first, family, 1.)
        b = _record(second, family, 1.)
        assert a.reference_gradient_norm == b.reference_gradient_norm
        assert a.reference_eval_key_fingerprint == b.reference_eval_key_fingerprint
        assert a.loss_ref_before == b.loss_ref_before


def test_reference_gradient_uses_exact_eval_key_and_no_cross_checkpoint_cache():
    def noisy_loss(p,b,key):
        target = b + jax.random.normal(key, shape=b.shape)
        return jnp.sum((p-target)**2)
    keys = (jax.random.key(3), jax.random.key(4), jax.random.key(5))
    result, cache = _evaluate(loss=noisy_loss, keys=keys)
    entry = cache.entries[("Replay", 0)]
    expected = jax.grad(noisy_loss)(jnp.array([0.]), jnp.array([1.]), keys[0])
    chex.assert_trees_all_close(entry.gradient, expected)
    assert entry.loss_ref_before == pytest.approx(
        float(noisy_loss(jnp.array([0.]), jnp.array([1.]), keys[0])))
    with pytest.raises(ValueError, match="reference_cache_mismatch"):
        _evaluate(cache=cache, keys=(jax.random.key(99), keys[1], keys[2]))
    other = ReferenceGradientCache(
        checkpoint=6, frozen_learner_fingerprint="other",
        frozen_critic_fingerprint="other", tolerance=1e-6,
    )
    assert other.entries == {}


def test_zero_and_tiny_gradients_have_null_cosine_not_zero():
    result, _ = _evaluate(source=jnp.array([0.]))
    zero = _record(result, "Replay", 1.)
    assert zero.source_gradient_norm == 0
    assert zero.raw_gradient_cosine is None
    assert zero.negative_update_reference_cosine is None
    assert zero.cosine_status == "near_zero_norm"
    tiny, _ = _evaluate(loss=lambda p,b,k: 1e-8*jnp.sum((p-b)**2))
    assert _record(tiny, "Replay", 1.).raw_gradient_cosine is None
    assert _record(tiny, "Replay", 1.).cosine_status == "near_zero_norm"
    for row in result.mechanism_records:
        if row.step_scale == 0:
            assert row.actual_parameter_displacement_norm == 0
            assert row.first_order_predicted_utility == 0
            assert abs(row.actual_utility) <= 1e-6


def test_nonfinite_gradient_reported_and_tiny_float32_displacement_measured():
    with jax.debug_nans(False):
        bad, _ = _evaluate(source=jnp.array([jnp.nan]))
    assert "non_finite_gradient" in bad.reasons
    assert all(not row.eligible and row.ineligible_reason for row in bad.mechanism_records)
    optimizer = optax.sgd(1e-8)
    small, _ = _evaluate(
        params=jnp.array([1e6], dtype=jnp.float32),
        source=jnp.array([999999.], dtype=jnp.float32),
        refs=(jnp.array([999999.]),)*3,
        optimizer=optimizer,
    )
    row = _record(small, "Replay", 1.)
    assert row.actual_parameter_displacement_norm == 0
    assert row.first_order_predicted_utility == 0
    assert row.numerical_status in ("near_zero_displacement", "near_zero_norm")


def test_multileaf_unequal_size_tree_and_actual_adam_direction():
    params = {"a": jnp.array([0.,0.]), "b": jnp.array([0.,0.,0.])}
    source = {"a": jnp.array([1.,2.]), "b": jnp.array([3.,4.,5.])}
    refs = (
        {"a": jnp.array([1.,0.]), "b": jnp.array([0.,0.,1.])},
        {"a": jnp.array([-1.,0.]), "b": jnp.array([0.,0.,-1.])},
        {"a": jnp.array([1.,0.]), "b": jnp.array([0.,0.,1.])},
    )
    loss = lambda p,b,k: sum(jnp.sum((p[name]-b[name])**2) for name in ("a","b"))
    result, _ = _evaluate(params=params, source=source, refs=refs,
                          optimizer=optax.adam(.1), loss=loss)
    row = _record(result, "Replay", 1.)
    assert result.valid
    assert row.source_gradient_norm == pytest.approx(float(np.linalg.norm([2,4,6,8,10])))
    assert row.raw_gradient_cosine != pytest.approx(row.negative_update_reference_cosine)
    assert row.actual_parameter_displacement_norm > 0


def test_mechanism_writer_has_unique_matched_primary_keys(tmp_path, monkeypatch):
    from evorl.algorithms.erl.erl_td3 import pilot_artifacts
    monkeypatch.setattr(pilot_artifacts, "PILOT_ROOT", tmp_path)
    writer = pilot_artifacts.PilotArtifactWriter(tmp_path, "run")
    result, _ = _evaluate()
    kwargs = dict(
        generation=5, status="PASS", reasons=(), source_results={"s":result},
        aggregates={}, provenance={"snapshot_id":"x"}, seed=7,
        source_types={"s":"EA"}, frozen_critic_fingerprint="frozen-critic",
    )
    writer.append_checkpoint(**kwargs)
    rows = list(csv.DictReader((tmp_path/"run"/"pilot_gradient_mechanisms.csv").open()))
    assert len(rows) == len(result.mechanism_records)
    keys = {(r["run_id"],r["checkpoint"],r["source_id"],r["reference_family"],
             r["reference_id"],r["step_scale"]) for r in rows}
    assert len(keys) == len(rows)
    assert {"first_order_residual", "reference_fingerprint", "source_update_fingerprint"} <= set(rows[0])
    with pytest.raises(ValueError, match="duplicate_mechanism_key"):
        writer.append_checkpoint(**kwargs)

def test_empty_parameter_gradient_is_zero_and_cosines_undefined():
    optimizer = optax.sgd(.1)
    cache = ReferenceGradientCache(
        checkpoint=5, frozen_learner_fingerprint="empty-learner",
        frozen_critic_fingerprint="empty-critic", tolerance=1e-6,
    )
    result = evaluate_paired_utility(
        loss_fn=lambda p,b,k: jnp.array(0., dtype=jnp.float32),
        optimizer=optimizer, params={}, opt_state=optimizer.init({}),
        source_batch=jnp.array([1.]),
        references={"Replay": (jnp.array([1.]),),
                    "RL-fresh": (jnp.array([1.]),),
                    "EA-fresh": (jnp.array([1.]),)},
        source_update_key=jax.random.key(2),
        reference_keys={"Replay": (jax.random.key(3),),
                        "RL-fresh": (jax.random.key(4),),
                        "EA-fresh": (jax.random.key(5),)},
        step_scales=(1., .5, .1), tolerance=1e-6,
        mechanism_cache=cache,
    )
    assert result.valid
    assert len(result.mechanism_records) == 12
    assert all(record.source_gradient_norm == 0
               and record.optax_update_norm == 0
               and record.raw_gradient_cosine is None
               and record.negative_update_reference_cosine is None
               and record.cosine_status == "near_zero_norm"
               for record in result.mechanism_records)

def test_reference_cache_rejects_changed_frozen_params_and_nonfinite_reference():
    cache = ReferenceGradientCache(
        checkpoint=5, frozen_learner_fingerprint="frozen-learner",
        frozen_critic_fingerprint="frozen-critic", tolerance=1e-6,
    )
    _evaluate(cache=cache)
    with pytest.raises(ValueError, match="reference_cache_mismatch"):
        _evaluate(cache=cache, params=jnp.array([2.]))
    with jax.debug_nans(False):
        bad, _ = _evaluate(refs=(jnp.array([jnp.nan]), jnp.array([-1.]), jnp.array([1.])))
    assert "non_finite_reference_gradient" in bad.reasons
    replay_rows = [r for r in bad.mechanism_records if r.reference_family == "Replay"]
    assert replay_rows and all(not r.eligible and
                               r.ineligible_reason == "non_finite_reference_gradient"
                               for r in replay_rows)


def test_writer_rejects_value_mismatch_before_creating_files(tmp_path, monkeypatch):
    from evorl.algorithms.erl.erl_td3 import pilot_artifacts
    monkeypatch.setattr(pilot_artifacts, "PILOT_ROOT", tmp_path)
    writer = pilot_artifacts.PilotArtifactWriter(tmp_path, "run")
    result, _ = _evaluate()
    wrong = replace(result, mechanism_records=(
        replace(result.mechanism_records[0],
                actual_utility=result.mechanism_records[0].actual_utility + 1.),
        *result.mechanism_records[1:],
    ))
    with pytest.raises(ValueError, match="mechanism_utility_value_mismatch"):
        writer.append_checkpoint(
            generation=5, status="PASS", reasons=(),
            source_results={"s":wrong}, aggregates={}, provenance={},
            seed=7, source_types={"s":"EA"},
            frozen_critic_fingerprint="frozen-critic",
        )
    assert not (tmp_path/"run").exists()

def test_large_parameters_do_not_obscure_finite_gradient_direction():
    result, _ = _evaluate(
        params=jnp.array([1e6], dtype=jnp.float32),
        source=jnp.array([999999.], dtype=jnp.float32),
        refs=(jnp.array([999999.]),)*3,
        optimizer=optax.sgd(.1),
    )
    row = _record(result, "Replay", 1.)
    assert row.source_gradient_norm == pytest.approx(2.)
    assert row.raw_gradient_cosine == pytest.approx(1.)
    assert row.negative_update_reference_cosine == pytest.approx(1.)


def test_writer_requires_mechanisms_and_persists_failed_source(tmp_path, monkeypatch):
    from evorl.algorithms.erl.erl_td3 import pilot_artifacts
    monkeypatch.setattr(pilot_artifacts, "PILOT_ROOT", tmp_path)
    writer = pilot_artifacts.PilotArtifactWriter(tmp_path, "run")
    result, _ = _evaluate()
    kwargs = dict(
        generation=5, status="HOLD", reasons=("non_finite_gradient_or_utility",),
        source_results={"s":replace(result, mechanism_records=())},
        aggregates={}, provenance={}, seed=7, source_types={"s":"EA"},
        frozen_critic_fingerprint="frozen-critic", require_mechanisms=True,
    )
    with pytest.raises(ValueError, match="mechanism_utility_key_mismatch"):
        writer.append_checkpoint(**kwargs)
    assert not (tmp_path/"run").exists()
    kwargs["source_results"] = {}
    kwargs["failed_sources"] = [{
        "source_id":"s", "reason":"non_finite_gradient_or_utility",
        "source_batch_fingerprint":"batch-hash",
    }]
    writer.append_checkpoint(**kwargs)
    failures = list(csv.DictReader((tmp_path/"run"/"pilot_gradient_failures.csv").open()))
    assert len(failures) == 1
    assert failures[0]["source_id"] == "s"
    assert failures[0]["source_batch_fingerprint"] == "batch-hash"
    assert failures[0]["frozen_critic_fingerprint"] == "frozen-critic"
    assert not (tmp_path/"run"/"pilot_gradient_mechanisms.csv").exists()
