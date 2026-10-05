from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from evorl.algorithms.erl.erl_td3.utility_source_controls import (
    CheckpointDiagnostic,
    SourceUtilityRecord,
    aggregate_reference_utilities,
    build_rl_own_source_batch,
    scaled_virtual_critic_utility,
)
from evorl.sample_batch import SampleBatch


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
        key=jax.random.PRNGKey(0),
        step_scale=1.0,
    )
    half = scaled_virtual_critic_utility(
        loss_fn=_quadratic,
        optimizer=optimizer,
        params=params,
        opt_state=opt_state,
        source_batch=jnp.array([0.0]),
        reference_batch=jnp.array([0.0]),
        key=jax.random.PRNGKey(0),
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
        key=jax.random.PRNGKey(1),
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
            key=jax.random.PRNGKey(2),
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
