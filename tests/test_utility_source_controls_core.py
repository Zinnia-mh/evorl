from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pytest

from evorl.algorithms.erl.erl_td3 import utility_source_controls as controls
from evorl.algorithms.erl.erl_td3.erl_learner_utility import (
    ERLLearnerUtilityWorkflow,
)
from evorl.algorithms.erl.erl_td3.utility_source_controls import (
    CheckpointDiagnostic,
    SourceUtilityRecord,
    aggregate_reference_utilities,
    build_rl_own_source_batch,
    compute_source_td_stats,
    derive_reference_eval_key,
    derive_source_update_key,
    evaluate_frozen_source_controls,
    scaled_virtual_critic_utility,
)
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.algorithms.td3 import make_mlp_td3_agent
from evorl.envs import Box
from types import SimpleNamespace


def _quadratic(params, batch, key):
    del key
    return jnp.square(params["w"] - batch).sum()


def test_step_scale_multiplies_final_optax_update_not_gradient():
    params = {"w": jnp.array([2.0])}
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)
    full = scaled_virtual_critic_utility(
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=opt_state,
        source_batch=jnp.array([0.0]),
        reference_batch=jnp.array([0.0]),
        source_update_key=jax.random.PRNGKey(0),
        reference_eval_key=jax.random.PRNGKey(1),
        step_scale=1.0,
    )
    half = scaled_virtual_critic_utility(
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=opt_state,
        source_batch=jnp.array([0.0]),
        reference_batch=jnp.array([0.0]),
        source_update_key=jax.random.PRNGKey(0),
        reference_eval_key=jax.random.PRNGKey(1),
        step_scale=0.5,
    )
    assert np.isclose(float(full.loss_ref_before), 4.0)
    assert np.isclose(float(full.loss_ref_after), 3.61, atol=1e-5)
    assert np.isclose(float(half.loss_ref_after), 3.8025, atol=1e-5)


def test_zero_step_reuses_optimizer_path_and_returns_zero_utility():
    calls = []
    base = optax.sgd(0.1)
    optimizer = optax.GradientTransformation(
        base.init,
        lambda updates, state, params=None: (
            calls.append(params) or base.update(updates, state, params)
        ),
    )
    params = {"w": jnp.array([2.0])}
    result = scaled_virtual_critic_utility(
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=optimizer.init(params),
        source_batch=jnp.array([0.0]),
        reference_batch=jnp.array([0.0]),
        source_update_key=jax.random.PRNGKey(1),
        reference_eval_key=jax.random.PRNGKey(2),
        step_scale=0.0,
    )
    assert len(calls) == 1
    assert float(result.utility) == 0.0
    assert float(result.loss_ref_before) == float(result.loss_ref_after)


def test_all_sources_start_from_identical_params_and_optimizer_state():
    seen = []
    params = {"w": jnp.array([2.0])}
    base = optax.sgd(0.1)
    optimizer = optax.GradientTransformation(
        base.init,
        lambda updates, state, params=None: (
            seen.append((float(params["w"][0]), state))
            or base.update(updates, state, params)
        ),
    )
    opt_state = optimizer.init(params)
    for source in (0.0, 1.0, 3.0):
        scaled_virtual_critic_utility(
            loss_fn=_quadratic,
            optimizer=optimizer,
            params=params,
            opt_state=opt_state,
            source_batch=jnp.array([source]),
            reference_batch=jnp.array([0.0]),
            source_update_key=jax.random.PRNGKey(2),
            reference_eval_key=jax.random.PRNGKey(3),
            step_scale=1.0,
        )
    assert [value for value, _ in seen] == [2.0, 2.0, 2.0]
    assert all(state == opt_state for _, state in seen)


def test_sign_consistency_ignores_near_zero_votes():
    result = aggregate_reference_utilities([2.0, 1.0, 1e-9, -1e-9], 1e-6)
    assert result.sign_consistency == 1.0
    assert result.near_zero_fraction == 0.5
    assert result.all_near_zero is False


def test_all_near_zero_returns_na_consistency():
    result = aggregate_reference_utilities([1e-9, -1e-9, 0.0], 1e-6)
    assert result.sign_consistency is None
    assert result.all_near_zero is True
    assert result.near_zero_fraction == 1.0


def test_non_finite_utility_invalidates_checkpoint():
    record = SourceUtilityRecord(
        checkpoint=1,
        source_type="EA",
        source_id="gen_1_ind_0",
        reference_id="reference_0",
        step_scale=1.0,
        loss_ref_before=1.0,
        loss_ref_after=0.0,
        utility=1.0,
        numerical_tolerance=1e-6,
    )
    checkpoint = CheckpointDiagnostic.from_records(
        checkpoint=1,
        temporal_alignment="exact",
        records=(record, replace(record, utility=float("nan"))),
    )
    assert checkpoint.valid is False
    assert "non_finite_utility" in checkpoint.reasons


def _rl_raw(num_agents, time, num_envs):
    markers = jnp.arange(num_agents * time * num_envs, dtype=jnp.float32).reshape(
        num_agents, time, num_envs
    )
    return SampleBatch(
        obs=markers[..., None],
        actions=markers[..., None],
        rewards=markers,
        dones=jnp.zeros((num_agents, time, num_envs), dtype=jnp.bool_),
    )


def _assert_rl_own_shape(num_agents, num_envs):
    raw = _rl_raw(num_agents, time=3, num_envs=num_envs)
    expected_count = num_agents * 3 * num_envs
    source = build_rl_own_source_batch(
        raw,
        generation=7,
        source_batch_size=expected_count,
        key=jax.random.key(4),
    )
    assert source.source_type == "RL-own"
    assert source.source_id == "gen_7_rl_own"
    assert source.valid_transition_count == expected_count
    assert source.batch.obs.shape == (expected_count, 1)
    assert set(map(float, source.batch.obs[:, 0])) == set(
        map(float, raw.obs.reshape(-1))
    )


def test_rl_own_single_agent_single_env_shape():
    _assert_rl_own_shape(1, 1)


def test_rl_own_multi_agent_single_env_shape():
    _assert_rl_own_shape(2, 1)


def test_rl_own_single_agent_multi_env_shape():
    _assert_rl_own_shape(1, 3)


def test_rl_own_multi_agent_multi_env_shape():
    _assert_rl_own_shape(2, 3)


def test_rl_own_shifted_done_excludes_post_terminal_transition():
    raw = _rl_raw(1, time=3, num_envs=1).replace(
        dones=jnp.array([[[False], [True], [False]]])
    )
    source = build_rl_own_source_batch(
        raw,
        generation=8,
        source_batch_size=2,
        key=jax.random.key(6),
    )
    assert source.valid_transition_count == 2
    assert set(map(float, source.batch.obs[:, 0])) == {0.0, 1.0}


def test_rl_own_insufficient_samples_is_ineligible():
    source = build_rl_own_source_batch(
        _rl_raw(1, time=2, num_envs=1),
        generation=9,
        source_batch_size=3,
        key=jax.random.key(5),
    )
    assert source.eligible is False
    assert source.reason == "insufficient_rl_own_samples"
    assert source.valid_transition_count == 2
    assert source.batch is None


def _td_fixture():
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = make_mlp_td3_agent(
        action_space,
        critic_hidden_layer_sizes=(8,),
        actor_hidden_layer_sizes=(8,),
        policy_noise=0.0,
    )
    agent_state = agent.init(obs_space, action_space, jax.random.key(1))
    batch = SampleBatch(
        obs=jnp.ones((4, 3)),
        actions=jnp.zeros((4, 2)),
        rewards=jnp.array([0.0, 1.0, 2.0, 4.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.full((4, 3), 0.5),
                termination=jnp.zeros(4),
            )
        ),
    )
    return agent, agent_state, batch


def test_source_control_td_adapter_matches_sealed_td_statistics():
    agent, agent_state, batch = _td_fixture()
    optimizer = optax.adam(1e-3)
    workflow = object.__new__(ERLLearnerUtilityWorkflow)
    workflow.agent = agent
    workflow.optimizer = optimizer
    workflow._diagnostic_context = SimpleNamespace(
        learner_agent_state=agent_state,
        critic_opt_state=optimizer.init(agent_state.params.critic_params),
    )
    root = jax.random.key(2)
    sealed, _ = workflow._evaluate_source_utility(batch, batch, root)
    td_key, _ = jax.random.split(root)
    adapted = compute_source_td_stats(
        agent=agent,
        frozen_agent_state=agent_state,
        source_batch=batch,
        source_update_key=td_key,
    )
    np.testing.assert_allclose(
        np.asarray(sealed), np.asarray(adapted), rtol=1e-6, atol=1e-6
    )


def test_td_adapter_uses_authoritative_td3_critic_loss(monkeypatch):
    calls = []

    def authoritative(agent, state, batch, key):
        del agent, state
        calls.append(jax.random.key_data(key))
        return batch.rewards.mean()

    monkeypatch.setattr(controls, "td3_critic_loss", authoritative)
    batch = SampleBatch(rewards=jnp.arange(4.0))
    stats = compute_source_td_stats(
        agent=object(),
        frozen_agent_state=object(),
        source_batch=batch,
        source_update_key=jax.random.key(3),
    )
    assert len(calls) == 1
    assert float(stats.mean) == pytest.approx(1.5)


def test_td_stats_share_common_update_key_across_sources(monkeypatch):
    def authoritative(agent, state, batch, key):
        del agent, state
        return jax.random.uniform(key) + 0.0 * batch.rewards.mean()

    monkeypatch.setattr(controls, "td3_critic_loss", authoritative)
    common = jax.random.key(9)
    results = []
    for offset in (0.0, 10.0, 20.0, 30.0):
        results.append(
            compute_source_td_stats(
                agent=object(),
                frozen_agent_state=object(),
                source_batch=SampleBatch(rewards=jnp.arange(4.0) + offset),
                source_update_key=common,
            )
        )
    for result in results[1:]:
        chex.assert_trees_all_equal(result, results[0])


def test_common_source_update_key_does_not_depend_on_source():
    baseline = jax.random.key(11)
    first = derive_source_update_key(baseline, generation=7)
    second = derive_source_update_key(baseline, generation=7)
    chex.assert_trees_all_equal(first, second)


def test_common_source_update_key_does_not_depend_on_scale():
    baseline = jax.random.key(11)
    keys = [derive_source_update_key(baseline, 7) for _ in (1.0, 0.5, 0.1)]
    chex.assert_trees_all_equal(keys[0], keys[1], keys[2])


def test_reference_eval_key_shared_across_sources_and_scales():
    baseline = jax.random.key(12)
    keys = [
        derive_reference_eval_key(baseline, generation=7, reference_index=2)
        for _source in range(4)
        for _scale in (1.0, 0.5, 0.1)
    ]
    chex.assert_trees_all_equal(*keys)


def test_reference_eval_key_differs_across_reference_indices():
    baseline = jax.random.key(12)
    keys = [derive_reference_eval_key(baseline, 7, index) for index in range(4)]
    encoded = [tuple(map(int, jax.random.key_data(key))) for key in keys]
    assert len(set(encoded)) == 4


def test_common_randomness_does_not_advance_baseline_key():
    baseline = jax.random.key(13)
    before = baseline
    derive_source_update_key(baseline, 8)
    derive_reference_eval_key(baseline, 8, 0)
    chex.assert_trees_all_equal(baseline, before)


def _matrix_fixture():
    batch = lambda value: jnp.full((2,), value)
    sources = (
        controls.SourceControlBatch("EA", "ea_0", batch(0.0), evolutionary_fitness=4.0),
        controls.SourceControlBatch("RL-own", "rl", batch(1.0)),
        controls.SourceControlBatch("Replay-random", "replay_0", batch(2.0)),
        controls.SourceControlBatch("Replay-random", "replay_1", batch(3.0)),
    )
    references = tuple(batch(value) for value in (0.5, 1.5, 2.5, 3.5))
    params = {"w": jnp.array([1.0])}
    optimizer = optax.sgd(0.1)
    return sources, references, params, optimizer


def _matrix_result(sources=None, loss_fn=_quadratic):
    default_sources, references, params, optimizer = _matrix_fixture()
    return evaluate_frozen_source_controls(
        checkpoint=5,
        generation=5,
        sources=default_sources if sources is None else sources,
        reference_batches=references,
        reference_fingerprints=("r0", "r1", "r2", "r3"),
        loss_fn=loss_fn,
        optimizer=optimizer,
        params=params,
        opt_state=optimizer.init(params),
        baseline_key=jax.random.key(21),
        step_scales=(1.0, 0.5, 0.1),
        numerical_tolerance=1e-6,
        temporal_alignment="exact",
    )


def test_four_source_evaluation_matrix_and_reference_self_diagonal():
    result = _matrix_result()
    formal = result.records
    assert len([r for r in formal if r.source_type == "EA"]) == 12
    assert len([r for r in formal if r.source_type == "RL-own"]) == 12
    assert len([r for r in formal if r.source_type == "Replay-random"]) == 24
    self_records = [r for r in formal if r.source_type == "Reference-self"]
    assert len(self_records) == 12
    assert all(r.source_id.endswith(r.reference_id.split("_")[-1]) for r in self_records)
    assert {r.step_scale for r in formal} == {1.0, 0.5, 0.1}


def test_same_reference_lbefore_equal_across_sources_and_step_scales():
    result = _matrix_result()
    for reference_id in {record.reference_id for record in result.records}:
        values = [
            record.loss_ref_before
            for record in result.records
            if record.reference_id == reference_id
        ]
        assert max(values) - min(values) <= 1e-6


def test_zero_step_uses_same_path_and_is_hard_gate():
    result = _matrix_result()
    assert len(result.zero_step_results) == 20
    assert result.valid is True
    assert all(abs(record.utility) <= record.numerical_tolerance for record in result.zero_step_results)


def test_source_reordering_does_not_change_results():
    sources, _, _, _ = _matrix_fixture()
    forward = _matrix_result(sources).records
    reverse = _matrix_result(tuple(reversed(sources))).records
    normalize = lambda records: sorted(
        (r.source_id, r.reference_id, r.step_scale, r.utility) for r in records
    )
    np.testing.assert_allclose(
        [row[-1] for row in normalize(forward)],
        [row[-1] for row in normalize(reverse)],
    )


def test_non_finite_still_invalidates_checkpoint():
    def non_finite(params, batch, key):
        del params, batch, key
        return jnp.array(jnp.nan)

    result = _matrix_result(loss_fn=non_finite)
    assert result.valid is False
    assert "non_finite_utility" in result.reasons


def test_matrix_batch_size_mismatch_rejects_checkpoint():
    sources, references, params, optimizer = _matrix_fixture()
    mismatched = replace(sources[0], batch=jnp.zeros((3,)))
    result = evaluate_frozen_source_controls(
        checkpoint=5,
        generation=5,
        sources=(mismatched, *sources[1:]),
        reference_batches=references,
        reference_fingerprints=("r0", "r1", "r2", "r3"),
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=optimizer.init(params),
        baseline_key=jax.random.key(21),
        step_scales=(1.0, 0.5, 0.1),
        numerical_tolerance=1e-6,
        temporal_alignment="exact",
    )
    assert result.valid is False
    assert result.reasons == ("batch_size_mismatch",)
    assert result.records == ()


def test_hard_zero_step_failure_still_invalidates_checkpoint():
    base = optax.sgd(0.1)
    optimizer = optax.GradientTransformation(
        base.init,
        lambda updates, state, params=None: (
            jtu.tree_map(lambda value: jnp.full_like(value, jnp.inf), updates),
            state,
        ),
    )
    sources, references, params, _ = _matrix_fixture()
    result = evaluate_frozen_source_controls(
        checkpoint=5,
        generation=5,
        sources=sources,
        reference_batches=references,
        reference_fingerprints=("r0", "r1", "r2", "r3"),
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=optimizer.init(params),
        baseline_key=jax.random.key(21),
        step_scales=(1.0, 0.5, 0.1),
        numerical_tolerance=1e-6,
        temporal_alignment="exact",
    )
    assert result.valid is False
    assert "zero_step_failure" in result.reasons


def test_reference_self_negative_is_warning_only():
    sources, references, params, _ = _matrix_fixture()
    optimizer = optax.sgd(10.0)
    result = evaluate_frozen_source_controls(
        checkpoint=5,
        generation=5,
        sources=sources,
        reference_batches=references,
        reference_fingerprints=("r0", "r1", "r2", "r3"),
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=optimizer.init(params),
        baseline_key=jax.random.key(21),
        step_scales=(1.0, 0.5, 0.1),
        numerical_tolerance=1e-6,
        temporal_alignment="exact",
    )
    assert result.valid is True
    assert "reference_self_negative" in result.warnings


def test_frozen_state_mutation_still_invalidates_checkpoint():
    diagnostic = CheckpointDiagnostic.from_records(
        checkpoint=5,
        temporal_alignment="exact",
        records=(),
        reasons=("frozen_state_mutation",),
    )
    assert diagnostic.valid is False
    assert diagnostic.measurement_eligible is True
    assert diagnostic.comparison_eligible is False
