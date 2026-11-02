import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.algorithms.erl.erl_td3.source_replay import (init_source_score_table,update_resident_counts,compute_source_eligibility,build_weighted_source_distribution,sample_weighted_ea_indices)
from evorl.algorithms.erl.erl_td3.source_residency_batch import recount_resident_sources

@pytest.mark.parametrize('seed',range(8))
def test_batch_recount_exactly_matches_sequential_slot_reclamation(seed):
    rng=np.random.default_rng(seed);capacity=64
    table=init_source_score_table(capacity)._replace(source_token=jnp.r_[jnp.arange(20,dtype=jnp.int32),jnp.full((44,),-1,jnp.int32)],raw_score=jnp.arange(capacity,dtype=jnp.float32),rank_normalized_score=jnp.arange(capacity,dtype=jnp.float32)/64,validity=jnp.ones(capacity,bool),measured_generation=jnp.ones(capacity,jnp.int32),version=jnp.zeros(capacity,jnp.int32),score_kind=jnp.ones(capacity,jnp.int8),current_version=jnp.int32(0))
    tokens=jnp.array(rng.integers(-1,32,size=capacity),jnp.int32)
    mask=jnp.array(rng.random(capacity)>.25)
    reset=table._replace(resident_count=jnp.zeros(capacity,jnp.int32))
    old=update_resident_counts(reset,jnp.full((capacity,),-1,jnp.int32),jnp.arange(capacity,dtype=jnp.int32),tokens,mask)
    new=recount_resident_sources(table,tokens,mask)
    for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(new)):np.testing.assert_array_equal(a,b)
    for w in [3,10,30]:
        a=compute_source_eligibility(old.table,10,w,4);b=compute_source_eligibility(new.table,10,w,4)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
        da=build_weighted_source_distribution(a.table,old.resident_source_tokens,mask,.5,.5)
        db=build_weighted_source_distribution(b.table,new.resident_source_tokens,mask,.5,.5)
        for x,y in zip(jax.tree.leaves(da),jax.tree.leaves(db)):np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
        np.testing.assert_array_equal(sample_weighted_ea_indices(da,jax.random.PRNGKey(seed),256).sampled_indices,sample_weighted_ea_indices(db,jax.random.PRNGKey(seed),256).sampled_indices)

def test_capacity_failure_rolls_back_like_reference():
    table=init_source_score_table(2);tokens=jnp.array([0,1,2]);mask=jnp.ones(3,bool)
    result=recount_resident_sources(table,tokens,mask)
    assert int(result.status)==1
    for a,b in zip(jax.tree.leaves(result.table),jax.tree.leaves(table)):np.testing.assert_array_equal(a,b)

def test_ring_overwrite_mixed_and_repeated_writes_match_actual_residency():
    from evorl.replay_buffers.source_aware_replay_buffer import SourceAwareReplayBuffer,ReplayProvenance
    rb=SourceAwareReplayBuffer(capacity=16,min_sample_timesteps=1,sample_batch_size=4)
    state=rb.init(dict(value=jnp.float32(0)));table=init_source_score_table(16)
    for generation in range(8):
        n=24 if generation%2 else 7
        origin=(jnp.arange(n)%3==0).astype(jnp.int8)
        tokens=jnp.where(origin==0,generation*4+jnp.arange(n)%4,-1).astype(jnp.int32)
        provenance=ReplayProvenance(origin_type=origin,source_token=tokens,source_generation=jnp.where(origin==0,generation,-1),score_version=jnp.full(n,-1,jnp.int32))
        state=rb.add(state,dict(value=jnp.arange(n,dtype=jnp.float32)),provenance,mask=jnp.arange(n)%5!=0)
        mask=(jnp.arange(16)<state.buffer_size)&(state.origin_type==0)
        actual=recount_resident_sources(table,state.source_token,mask)
        expected=update_resident_counts(table._replace(resident_count=jnp.zeros(16,jnp.int32)),jnp.full(16,-1,jnp.int32),jnp.arange(16),state.source_token,mask)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)
        ids=np.asarray(actual.table.source_token);counts=np.asarray(actual.table.resident_count)
        for token,count in zip(ids,counts):
            if count:assert count==np.sum(np.asarray(mask)&(np.asarray(state.source_token)==token))
        table=actual.table

@pytest.mark.parametrize('empty',[False,True])
def test_recount_does_not_require_registry_capacity_equal_to_replay(empty):
    table=init_source_score_table(16)
    tokens=jnp.arange(128,dtype=jnp.int32)%8
    mask=jnp.zeros(128,bool) if empty else jnp.arange(128)%3!=0
    before=update_resident_counts(table,jnp.full(128,-1,jnp.int32),jnp.arange(128),tokens,mask)
    after=recount_resident_sources(table,tokens,mask)
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(after)):np.testing.assert_array_equal(a,b)

def test_refresh_never_calls_transition_by_registry_scan(monkeypatch):
    from types import SimpleNamespace
    import evorl.algorithms.erl.erl_td3.erl_utility_replay as pilot
    def forbidden(*args,**kwargs):raise AssertionError('quadratic recount path called')
    monkeypatch.setattr(pilot,'update_resident_counts',forbidden)
    state=pilot.init_utility_replay_state(64,jax.random.PRNGKey(5))
    replay=SimpleNamespace(origin_type=jnp.zeros(64,jnp.int8),source_token=jnp.arange(64,dtype=jnp.int32)//8,buffer_size=jnp.int32(64))
    after=pilot.refresh_replay_distribution(state,replay,current_generation=20,freshness_window=30,min_resident=4,alpha=.5,replay_temperature=.5)
    assert after.score_table.resident_count.shape==(64,) and int(after.score_table.resident_count.sum())==64
