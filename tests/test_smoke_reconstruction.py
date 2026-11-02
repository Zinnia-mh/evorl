import jax,jax.numpy as jnp,numpy as np,pytest
from evorl.algorithms.erl.erl_td3.erl_utility_replay import UtilityReplayBuffer,init_utility_replay_state,refresh_replay_distribution,weighted_replay_draws,alpha_zero_sample
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import reconstruct_replay_draws

def fixture():
    rb=UtilityReplayBuffer(capacity=32,min_sample_timesteps=1,sample_batch_size=128)
    state=rb.init(dict(obs=jnp.float32(0))).replace(data=dict(obs=jnp.arange(32,dtype=jnp.float32)),buffer_size=jnp.int32(32),origin_type=jnp.arange(32,dtype=jnp.int8)%2,source_token=jnp.where(jnp.arange(32)%2==0,jnp.arange(32)//16,-1).astype(jnp.int32))
    util=init_utility_replay_state(32,jax.random.PRNGKey(67));table=util.score_table
    table=table._replace(source_token=table.source_token.at[:2].set(jnp.arange(2)),raw_score=table.raw_score.at[:2].set(jnp.array([-1.,1.])),rank_normalized_score=table.rank_normalized_score.at[:2].set(jnp.array([0.,1.])),validity=table.validity.at[:2].set(True),measured_generation=table.measured_generation.at[:2].set(1),version=table.version.at[:2].set(0),score_kind=table.score_kind.at[:2].set(3),current_version=jnp.int32(0))
    util=refresh_replay_distribution(util._replace(score_table=table),state,current_generation=1,freshness_window=30,min_resident=4,alpha=.5,replay_temperature=.5)
    assert bool(util.distribution_prepared) and float(jnp.sum(jnp.abs(util.distribution.source_probability-util.distribution.baseline_source_probability)))>0
    return state,util

@pytest.mark.parametrize('mode',['uniform','neutral','weighted'])
def test_reconstructed_indices_equal_direct_sampler_calls(mode):
    state,util=fixture();key=jax.random.PRNGKey(9)
    if mode=='neutral':util=util._replace(distribution_prepared=jnp.bool_(False))
    actual=reconstruct_replay_draws(state,key,util,num_updates=3,num_draws=2,batch_size=128,uniform=mode=='uniform')
    expected=[];current=util;loop_key=key
    for _ in range(3):
        loop_key,rb_key,_=jax.random.split(loop_key,3);rb_keys=jax.random.split(rb_key,2)
        if mode=='uniform':
            sampled=jax.vmap(lambda k:alpha_zero_sample(state,k,current.intervention_key,128))(rb_keys)
            expected.append((sampled.baseline_indices,sampled.final_indices))
        else:
            sampled=weighted_replay_draws(state,rb_keys,current,128);current=sampled.utility_replay_state
            expected.append((sampled.baseline_indices,sampled.final_indices))
    np.testing.assert_array_equal(actual.baseline_indices,jnp.stack([x[0] for x in expected]));np.testing.assert_array_equal(actual.final_indices,jnp.stack([x[1] for x in expected]))
    np.testing.assert_array_equal(actual.terminal_learning_key,loop_key);np.testing.assert_array_equal(actual.terminal_intervention_key,current.intervention_key)
    assert np.all(np.asarray(actual.baseline_indices)[np.asarray(actual.rl_slot_mask)]==np.asarray(actual.final_indices)[np.asarray(actual.rl_slot_mask)])
    if mode=='weighted':assert np.any(actual.baseline_indices!=actual.final_indices)
    else:np.testing.assert_array_equal(actual.baseline_indices,actual.final_indices)

def test_zero_updates_have_empty_trace_and_do_not_advance_keys():
    state,util=fixture();key=jax.random.PRNGKey(9)
    trace=reconstruct_replay_draws(state,key,util,num_updates=0,num_draws=2,batch_size=128,uniform=False)
    assert trace.baseline_indices.shape==(0,2,128)
    np.testing.assert_array_equal(trace.terminal_learning_key,key);np.testing.assert_array_equal(trace.terminal_intervention_key,util.intervention_key)

def test_initialization_fingerprint_binds_actual_keys_parameters_and_shapes():
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
    state=dict(key=jax.random.PRNGKey(1),params=jnp.array([1.,2.],jnp.float32))
    original=initialization_fingerprint(state)
    assert original==initialization_fingerprint(jax.tree.map(lambda x:np.array(x),state))
    assert original!=initialization_fingerprint(dict(state,key=jax.random.PRNGKey(2)))
    assert original!=initialization_fingerprint(dict(state,params=jnp.array([1.,3.],jnp.float32)))
    assert original!=initialization_fingerprint(dict(state,params=state['params'].reshape(1,2)))
    assert len(initialization_fingerprint(dict(key=jax.random.key(1))))==64
