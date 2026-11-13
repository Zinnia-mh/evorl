import jax
import jax.numpy as jnp
import numpy as np

from evorl.algorithms.erl.erl_td3.source_replay import (
    build_weighted_source_distribution, init_source_score_table,
)
from evorl.algorithms.erl.erl_td3.erl_utility_replay import weighted_replay_sample
from evorl.types import State
from evorl.algorithms.erl.erl_td3.d0_placebo import permuted_ranks


def test_placebo_preserves_distribution_strength_and_rl_indices():
    table = init_source_score_table(7)._replace(
        source_token=jnp.arange(7),
        measured_generation=jnp.array([20, 20, 20, 20, 30, 30, 30]),
        resident_count=jnp.array([4, 4, 4, 5, 4, 4, 2]),
        weighted_eligible=jnp.array([True] * 6 + [False]),
        rank_normalized_score=jnp.array([0., .25, .5, .75, 0., 1., .5]),
    )
    effective = permuted_ranks(table, seed=2081)
    ranks = np.asarray(effective.rank_normalized_score)
    original = np.asarray(table.rank_normalized_score)
    np.testing.assert_array_equal(ranks, permuted_ranks(table, seed=2081).rank_normalized_score)
    assert np.any(ranks != original)
    for group in ([0, 1, 2], [3], [4, 5], [6]):
        np.testing.assert_array_equal(sorted(ranks[group]), sorted(original[group]))
    for field in table._fields:
        if field != 'rank_normalized_score':
            np.testing.assert_array_equal(getattr(table, field), getattr(effective, field))

    tokens = jnp.concatenate([jnp.repeat(table.source_token, table.resident_count), jnp.full(4, -1)])
    ea = tokens >= 0
    replay = State(buffer_size=jnp.asarray(len(tokens)), origin_type=jnp.where(ea, 0, 1), data=jnp.arange(len(tokens)))
    d0, placebo = [build_weighted_source_distribution(t, tokens, ea, .5, .5) for t in (table, effective)]
    for field in ('ess_fraction', 'eligible_baseline_mass'):
        np.testing.assert_allclose(getattr(d0, field), getattr(placebo, field), rtol=1e-6)
    np.testing.assert_allclose(
        np.sum(abs(d0.source_probability - d0.baseline_source_probability)),
        np.sum(abs(placebo.source_probability - placebo.baseline_source_probability)),
        rtol=1e-6,
    )
    draws = [weighted_replay_sample(replay, jax.random.PRNGKey(71), jax.random.PRNGKey(81), d, True, 256) for d in (d0, placebo)]
    native, shuffled = draws
    np.testing.assert_array_equal(native.baseline_indices, shuffled.baseline_indices)
    np.testing.assert_array_equal(native.ea_slot_mask, shuffled.ea_slot_mask)
    np.testing.assert_array_equal(native.next_intervention_key, shuffled.next_intervention_key)
    np.testing.assert_array_equal(shuffled.final_indices[shuffled.rl_slot_mask], native.final_indices[native.rl_slot_mask])
    np.testing.assert_array_equal(shuffled.batch, shuffled.final_indices)
    assert np.any(native.final_indices != shuffled.final_indices)

