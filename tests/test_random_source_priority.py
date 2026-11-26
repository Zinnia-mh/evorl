"""Minimal R source-selection/lifecycle and real corrected-overlay tests."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp

def module():
    try:return importlib.import_module("scripts.random_source_priority")
    except ModuleNotFoundError as e:
        if e.name=="scripts.random_source_priority":pytest.fail("R implementation missing")
        raise

def test_same_source_selection_as_P_and_32_priorities_per10_generations():
    m=module()
    from evorl.algorithms.erl.erl_td3.budgeted_utility_scoring import DistributedScoringMixin,score_quota
    baseline=jax.random.PRNGKey(42);counts=[]
    for g in range(11,21):
        counts.append(score_quota(g,32,10,10))
        p=SimpleNamespace(_diagnostic_context=SimpleNamespace(key=baseline,generation=g),
            config=SimpleNamespace(pop_size=32,diagnostic_interval=10,warmup_iters=10))
        expected=DistributedScoringMixin._select_diagnostic_sources(p,tuple(range(32)))
        assert tuple(m.selected_source_ids(baseline,g,32,10,10))==expected
    assert counts==[3,3,3,3,4,3,3,3,3,4] and sum(counts)==32

def prepared():
    m=module()
    from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table,apply_checkpoint_score_update,compute_source_eligibility
    update=m.synthetic_update(11,[1,8,16],[True,True,True],0,32)
    table=apply_checkpoint_score_update(init_source_score_table(8),update).table
    table=table._replace(resident_count=table.resident_count.at[:3].set(16))
    table=compute_source_eligibility(table,11,30,4).table
    return m,table

def test_random_rank_grid_persistence_and_residency_freshness_eligibility():
    m,t=prepared()
    from scripts.run_s_utility_extension import permute_cohort_ranks
    from evorl.algorithms.erl.erl_td3.source_replay import compute_source_eligibility
    np.testing.assert_allclose(t.rank_normalized_score[:3],[0.,.5,1.])
    first=permute_cohort_ranks(t,3083);second=permute_cohort_ranks(t,3083)
    np.testing.assert_array_equal(first.rank_normalized_score,second.rank_normalized_score)
    assert sorted(np.asarray(first.rank_normalized_score[:3]))==[0.,.5,1.]
    assert int(compute_source_eligibility(t,41,30,4).eligible_count)==3
    assert int(compute_source_eligibility(t,42,30,4).eligible_count)==0
    reduced=t._replace(resident_count=t.resident_count.at[0].set(3))
    assert int(compute_source_eligibility(reduced,11,30,4).eligible_count)==2
    update=m.synthetic_update(12,[0,3,7],[True,False,True],1,32)
    np.testing.assert_allclose(update.rank_normalized_score,[0.,0.,1.])
    assert not bool(update.validity[1])

def test_actual_corrected_sampling_changes_EA_indices_and_preserves_RL_slots():
    m,table=prepared()
    from scripts.run_s_utility_extension import permute_cohort_ranks
    from evorl.algorithms.erl.erl_td3.source_replay import build_weighted_source_distribution,apply_weighted_ea_overlay
    tokens=jnp.concatenate([jnp.repeat(table.source_token[:3],16),jnp.full(16,-1)])
    origin=jnp.concatenate([jnp.zeros(48,dtype=jnp.int8),jnp.ones(16,dtype=jnp.int8)])
    dist=build_weighted_source_distribution(permute_cohort_ranks(table,3083),tokens,origin==0,.5,.5)
    baseline=jnp.arange(64,dtype=jnp.int32)
    sample=apply_weighted_ea_overlay(baseline,origin,dist,jax.random.PRNGKey(9))
    final=np.asarray(sample.final_indices)
    assert np.any(final[:48]!=np.arange(48))
    np.testing.assert_array_equal(final[48:],np.arange(48,64))
    np.testing.assert_array_equal(np.asarray(origin)[final],np.asarray(origin))
    assert int(dist.eligible_source_count)==3
