"""Small real ring writes: scored identities must survive physical reordering."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from evorl.types import State,PyTreeDict
from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.replay_buffers.source_aware_replay_buffer import SourceAwareReplayBuffer,ReplayProvenance
from evorl.algorithms.erl.erl_td3.erl_utility_replay import init_utility_replay_state,refresh_replay_distribution,weighted_replay_sample,alpha_zero_sample
from evorl.algorithms.erl.erl_td3.source_replay import SCORE_KIND_UTILITY
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

def provenance(tokens,generation,version):
    tokens=jnp.asarray(tokens,jnp.int32);ea=tokens>=0
    return ReplayProvenance(origin_type=jnp.where(ea,0,1).astype(jnp.int8),source_token=tokens,source_generation=jnp.where(ea,generation,-1).astype(jnp.int32),score_version=jnp.where(ea,version,-1).astype(jnp.int32))

def wrapped():
    rb=SourceAwareReplayBuffer(capacity=16,min_sample_timesteps=1,sample_batch_size=256)
    replay=rb.add(rb.init(jnp.int32(0)),jnp.arange(16,dtype=jnp.int32),provenance([100,200]*8,30,7))
    utility=init_utility_replay_state(16,jax.random.PRNGKey(2027),score_table_capacity=3)
    table=utility.score_table._replace(source_token=jnp.array([100,200,400],jnp.int32),resident_count=jnp.array([8,8,0],jnp.int32),raw_score=jnp.array([1.,-1.,99.]),rank_normalized_score=jnp.array([1.,0.,.5]),measured_generation=jnp.array([30,30,1],jnp.int32),score_kind=jnp.full(3,SCORE_KIND_UTILITY,jnp.int8),validity=jnp.ones(3,bool),version=jnp.full(3,7,jnp.int32),current_version=jnp.int32(7))
    utility=utility._replace(score_table=table)
    replay=rb.add(replay,jnp.arange(1000,1004,dtype=jnp.int32),provenance([300,-1,300,-1],31,-1))
    return rb,replay,utility

def refresh(utility,replay,generation=31):
    return refresh_replay_distribution(utility,replay,current_generation=generation,freshness_window=30,min_resident=4,alpha=.5,replay_temperature=.5)

def assert_actual_counts(table,replay):
    actual=np.asarray(replay.source_token);origin=np.asarray(replay.origin_type)
    resident=np.arange(actual.size)<int(replay.buffer_size)
    for token,count in zip(np.asarray(table.source_token),np.asarray(table.resident_count)):
        assert count==np.count_nonzero(resident&(origin==0)&(actual==token)) if token>=0 else count==0

def test_first_ring_wrap_preserves_resident_scores_and_payload_sidecars():
    rb,replay,before=wrapped();after=refresh(before,replay)
    assert int(replay.current_index)==4 and int(replay.buffer_size)==16
    np.testing.assert_array_equal(replay.data,np.r_[np.arange(1000,1004),np.arange(4,16)])
    np.testing.assert_array_equal(replay.source_token,np.r_[[300,-1,300,-1],[100,200]*6])
    np.testing.assert_array_equal(replay.source_generation,np.r_[[31,-1,31,-1],[30]*12])
    np.testing.assert_array_equal(replay.score_version,np.r_[[-1]*4,[7]*12])
    np.testing.assert_array_equal(replay.origin_type,np.r_[[0,1,0,1],[0]*12])
    assert bool(after.distribution_prepared) and int(after.distribution.eligible_source_count)==2
    assert_actual_counts(after.score_table,replay)
    for token in [100,200]:
        old=int(np.flatnonzero(np.asarray(before.score_table.source_token)==token)[0]);new=int(np.flatnonzero(np.asarray(after.score_table.source_token)==token)[0])
        assert new==old
        for field in ['raw_score','rank_normalized_score','measured_generation','score_kind','validity','version']:
            np.testing.assert_array_equal(getattr(after.score_table,field)[new],getattr(before.score_table,field)[old])
    assert int(after.score_table.source_token[2])==300
    assert not bool(after.score_table.validity[2]) and int(after.score_table.version[2])==-1

@pytest.mark.parametrize('offset',[1,5,12])
def test_physical_ring_order_does_not_erase_scores_or_change_source_probabilities(offset):
    _,replay,utility=wrapped();expected=refresh(utility,replay)
    rotated=jax.tree.map(lambda x:jnp.roll(x,offset,axis=0) if hasattr(x,'shape') and x.shape==(16,) else x,replay)
    actual=refresh(utility,rotated)
    assert bool(actual.distribution_prepared)
    assert_actual_counts(actual.score_table,rotated)
    for a,b in zip(jax.tree.leaves(actual.score_table),jax.tree.leaves(expected.score_table)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(actual.distribution.source_probability,expected.distribution.source_probability)
    np.testing.assert_array_equal(actual.distribution.transition_probability,jnp.roll(expected.distribution.transition_probability,offset))

def test_genuine_expiry_residency_threshold_and_full_eviction_remain_neutral():
    rb,replay,utility=wrapped();active=refresh(utility,replay)
    assert bool(active.distribution_prepared)
    assert bool(refresh(active,replay,60).distribution_prepared)
    expired=refresh(active,replay,61)
    assert not bool(expired.distribution_prepared)
    np.testing.assert_array_equal(expired.distribution.source_probability,expired.distribution.baseline_source_probability)
    # Uniform transitions give unequal source mass when counts are 6,6,2.
    # Source ESS can therefore be <1 while the intervention L1 is exactly zero.
    p=np.asarray(expired.distribution.source_probability,dtype=np.float64)
    assert 1/(len(p)*np.sum((p/p.sum())**2))<1
    replay=rb.add(replay,jnp.arange(2000,2006,dtype=jnp.int32),provenance([-1]*6,31,-1))
    thin=refresh(active,replay);assert_actual_counts(thin.score_table,replay)
    assert not bool(thin.distribution_prepared)
    np.testing.assert_array_equal(thin.score_table.resident_count[:2],[3,3])
    replay=rb.add(replay,jnp.arange(3000,3006,dtype=jnp.int32),provenance([-1]*6,31,-1))
    evicted=refresh(thin,replay);assert_actual_counts(evicted.score_table,replay)
    assert not bool(evicted.distribution_prepared)
    np.testing.assert_array_equal(evicted.score_table.resident_count[:2],[0,0])

def test_post_wrap_real_sampling_changes_only_ea_indices_and_alpha_zero_is_baseline():
    _,replay,utility=wrapped();utility=refresh(utility,replay)
    assert bool(utility.distribution_prepared)
    assert np.abs(np.asarray(utility.distribution.source_probability-utility.distribution.baseline_source_probability)).sum()>0
    key=jax.random.PRNGKey(17);intervention=utility.intervention_key
    draw=weighted_replay_sample(replay,key,intervention,utility.distribution,utility.distribution_prepared,256)
    repeated=weighted_replay_sample(replay,key,intervention,utility.distribution,utility.distribution_prepared,256)
    for x,y in zip(jax.tree.leaves(draw),jax.tree.leaves(repeated)):np.testing.assert_array_equal(x,y)
    ea=np.asarray(draw.ea_slot_mask);rl=np.asarray(draw.rl_slot_mask);base=np.asarray(draw.baseline_indices);final=np.asarray(draw.final_indices)
    assert ea.any() and rl.any() and np.any(base[ea]!=final[ea])
    np.testing.assert_array_equal(base[rl],final[rl])
    np.testing.assert_array_equal(replay.origin_type[base],replay.origin_type[final])
    np.testing.assert_array_equal(draw.batch,replay.data[draw.final_indices])
    neutral=alpha_zero_sample(replay,key,intervention,256)
    original=ReplayBuffer(capacity=16,min_sample_timesteps=1,sample_batch_size=256)
    np.testing.assert_array_equal(neutral.batch,original.sample(replay,key))
    np.testing.assert_array_equal(neutral.baseline_indices,neutral.final_indices)
    np.testing.assert_array_equal(neutral.intervention_key,intervention)

def test_checkpoint_restores_wrapped_replay_scores_versions_and_identical_sampling(tmp_path):
    _,replay,utility=wrapped();utility=refresh(utility,replay)
    assert bool(utility.distribution_prepared)
    state=State(key=jax.random.PRNGKey(2),metrics=PyTreeDict(iterations=jnp.uint32(31)),replay_buffer_state=replay,utility_replay_state=utility)
    config=OmegaConf.create(dict(seed=2027,replay_buffer_capacity=16,freshness_window=30,min_resident=4,alpha=.5))
    with PilotCheckpointStore(tmp_path/'checkpoints',config) as store:store.save(state,dict(generation=31),force=True)
    with PilotCheckpointStore(tmp_path/'checkpoints',config,read_only=True) as store:restored,runtime=store.restore(31,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),state))
    assert runtime==dict(generation=31) and initialization_fingerprint(restored)==initialization_fingerprint(state)
    assert_actual_counts(restored.utility_replay_state.score_table,restored.replay_buffer_state)
    key=jax.random.PRNGKey(9)
    first=weighted_replay_sample(replay,key,utility.intervention_key,utility.distribution,utility.distribution_prepared,256)
    u=restored.utility_replay_state;second=weighted_replay_sample(restored.replay_buffer_state,key,u.intervention_key,u.distribution,u.distribution_prepared,256)
    np.testing.assert_array_equal(first.final_indices,second.final_indices)
