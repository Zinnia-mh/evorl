import json
import jax,jax.numpy as jnp
import pytest
from evorl.types import State,PyTreeDict
from evorl.algorithms.erl.erl_td3.initialization_evidence import capture_evidence,compare_evidence
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

def state():
    return State(key=jax.random.PRNGKey(1),agent_state=PyTreeDict(params=PyTreeDict(actor_params=jnp.array([1.,2.]),critic_params=jnp.array([3.]),target_actor_params=jnp.array([4.]),target_critic_params=jnp.array([5.]))),opt_state=PyTreeDict(actor=jnp.zeros(2),critic=jnp.zeros(2)),ec_opt_state=PyTreeDict(pop=jnp.ones(3),key=jax.random.PRNGKey(2)),replay_buffer_state=PyTreeDict(data=PyTreeDict(obs=jnp.arange(8.)),buffer_size=jnp.int32(3),current_index=jnp.int32(3),origin_type=jnp.array([0,0,0,-1,-1,-1,-1,-1]),source_token=jnp.arange(8)),utility_replay_state=PyTreeDict(intervention_key=jax.random.PRNGKey(3),score_table=PyTreeDict(version=jnp.int32(-1))))

def test_capture_keeps_full_hash_and_splits_resident_from_unused(tmp_path):
    s=state();result=capture_evidence(s,tmp_path/'a',metadata={'stage':'test'})
    assert result['legacy_sha256']==initialization_fingerprint(s)
    rows={r['path']:r for r in result['leaves']}
    assert rows['replay_buffer_state.data.obs']['resident']['shape']==[3]
    assert rows['replay_buffer_state.data.obs']['unused']['shape']==[5]
    assert rows['agent_state.params.target_critic_params']['group']=='target_critic'
    assert rows['utility_replay_state.intervention_key']['group']=='intervention_prng'
    with pytest.raises(FileExistsError):capture_evidence(s,tmp_path/'a',metadata={})

def test_compare_identifies_real_parameter_change_and_unchanged_replay(tmp_path):
    a=state();b=a.replace(agent_state=a.agent_state.replace(params=a.agent_state.params.replace(actor_params=jnp.array([1.,3.]))))
    capture_evidence(a,tmp_path/'a',metadata={});capture_evidence(b,tmp_path/'b',metadata={})
    result=compare_evidence(tmp_path/'a',tmp_path/'b')
    assert result['different_leaves']==1
    assert result['differences'][0]['path']=='agent_state.params.actor_params'
    assert result['differences'][0]['classification']=='numeric_value_difference'

def test_unused_region_is_reported_not_ignored(tmp_path):
    a=state();b=a.replace(replay_buffer_state=a.replay_buffer_state.replace(data=PyTreeDict(obs=jnp.arange(8.).at[7].set(99))))
    capture_evidence(a,tmp_path/'a',metadata={});capture_evidence(b,tmp_path/'b',metadata={})
    result=compare_evidence(tmp_path/'a',tmp_path/'b')
    assert result['differences'][0]['resident_equal'] is True and result['differences'][0]['unused_equal'] is False

def test_signed_zero_is_classified_without_relaxing_byte_equality(tmp_path):
    a=state();b=a.replace(opt_state=PyTreeDict(actor=jnp.array([-0.,0.]),critic=jnp.zeros(2)))
    capture_evidence(a,tmp_path/'a',metadata={});capture_evidence(b,tmp_path/'b',metadata={})
    result=compare_evidence(tmp_path/'a',tmp_path/'b')
    assert result['different_leaves']==1 and result['differences'][0]['classification']=='byte_representation_difference'
