import copy
import jax,jax.numpy as jnp,numpy as np,optax,pytest
from omegaconf import OmegaConf
from evorl.types import State,PyTreeDict
from evorl.algorithms.erl.erl_td3.erl_utility_replay import init_utility_replay_state
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore

def config():
    return OmegaConf.create(dict(seed=9,replay_buffer_capacity=8,alpha=.5,freshness_window=30,min_resident=4,eval_interval=2,eval_episodes=2,total_episodes=70,warmup_iters=10,optimizer=dict(lr=.0003),agent_network=dict(actor_hidden_layer_sizes=[8,8])))

def state():
    u=init_utility_replay_state(8,jax.random.PRNGKey(7))
    u=u._replace(score_table=u.score_table._replace(current_version=jnp.int32(3),source_token=jnp.arange(8,dtype=jnp.int32),measured_generation=jnp.full(8,12,jnp.int32),version=jnp.full(8,3,jnp.int32),resident_count=jnp.ones(8,jnp.int32),validity=jnp.ones(8,bool)),distribution_prepared=jnp.bool_(True))
    params=PyTreeDict(actor_params=jnp.array([1.,2.]),critic_params=jnp.array([3.,4.]),target_actor_params=jnp.array([5.,6.]),target_critic_params=jnp.array([7.,8.]))
    return State(key=jax.random.PRNGKey(4),metrics=PyTreeDict(iterations=jnp.uint32(13)),agent_state=PyTreeDict(params=params),opt_state=PyTreeDict(actor=optax.adam(.0003).init(params.actor_params),critic=optax.adam(.0003).init(params.critic_params)),ec_opt_state=PyTreeDict(pop=jnp.arange(12.).reshape(4,3),key=jax.random.PRNGKey(5)),replay_buffer_state=PyTreeDict(data=PyTreeDict(obs=jnp.arange(16.).reshape(8,2)),source_token=jnp.arange(8,dtype=jnp.int32),origin_type=jnp.array([0,1]*4),score_version=jnp.arange(8,dtype=jnp.int32),source_generation=jnp.arange(8,dtype=jnp.int32),buffer_size=jnp.int32(8),current_index=jnp.int32(3)),utility_replay_state=u)

def test_native_composite_full_state_and_runtime_survive_close_and_restore(tmp_path):
    s=state();runtime=dict(evaluations=[dict(generation=12,returns=[1.25])],audit=dict(steps=3))
    with PilotCheckpointStore(tmp_path/'checkpoints',config(),interval=2,keep=3) as store:
        path=store.save(s,runtime,force=True)
    with PilotCheckpointStore(tmp_path/'checkpoints',config(),read_only=True) as store:
        restored,progress=store.restore(13,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),s))
    assert initialization_fingerprint(restored)==initialization_fingerprint(s)
    assert progress==runtime and path.name=='13'
    assert int(restored.utility_replay_state.score_table.current_version)==3
    np.testing.assert_array_equal(restored.key,s.key)
    np.testing.assert_array_equal(restored.utility_replay_state.intervention_key,s.utility_replay_state.intervention_key)

@pytest.mark.parametrize('field,value',[('seed',10),('replay_buffer_capacity',16),('eval_interval',10),('warmup_iters',0),('freshness_window',3),('total_episodes',100),('optimizer.lr',.003),('agent_network.actor_hidden_layer_sizes',[16,16])])
def test_resume_rejects_incompatible_training_or_evaluation_config(tmp_path,field,value):
    s=state();c=config()
    with PilotCheckpointStore(tmp_path/'checkpoints',c) as store:store.save(s,{},force=True)
    OmegaConf.update(c,field,value)
    with PilotCheckpointStore(tmp_path/'checkpoints',c,read_only=True) as store:
        with pytest.raises(ValueError,match='compatib'):store.restore(13,s)

def test_existing_checkpoint_cannot_be_overwritten_and_keep_at_least_two(tmp_path):
    s=state()
    with pytest.raises(ValueError):PilotCheckpointStore(tmp_path/'bad',config(),keep=1)
    with PilotCheckpointStore(tmp_path/'checkpoints',config(),keep=2) as store:
        path=store.save(s,{},force=True)
        with pytest.raises(FileExistsError):store.save(s,{},force=True)
        store.save(s.replace(metrics=PyTreeDict(iterations=jnp.uint32(14))),{},force=True)
        assert path.exists()
        store.save(s.replace(metrics=PyTreeDict(iterations=jnp.uint32(15))),{},force=True)
        assert set(store.steps())=={14,15}

def test_incomplete_or_corrupted_checkpoint_is_rejected(tmp_path):
    s=state()
    with PilotCheckpointStore(tmp_path/'checkpoints',config()) as store:
        path=store.save(s,{},force=True)
        (path/'runtime'/'unexpected.txt').write_text('corruption')
        with pytest.raises(ValueError,match='checksum'):store.restore(13,s)
        with pytest.raises(FileNotFoundError):store.restore(14,s)

def test_zero_size_state_fields_are_restored_not_removed(tmp_path):
    s=state().replace(empty_state_array=jnp.empty((0,2),dtype=jnp.float32))
    with PilotCheckpointStore(tmp_path/'checkpoints',config()) as store:
        store.save(s,{},force=True)
        restored,_=store.restore(13,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),s))
    assert restored.empty_state_array.shape==(0,2)
    assert initialization_fingerprint(restored)==initialization_fingerprint(s)

def test_dynamic_distribution_shape_is_restored_from_saved_state_not_initial_template(tmp_path):
    s=state()
    initial=s
    s=s.replace(utility_replay_state=s.utility_replay_state._replace(
        distribution=s.utility_replay_state.distribution._replace(
            source_probability=jnp.full(8,1/8))))
    with PilotCheckpointStore(tmp_path/'checkpoints',config()) as store:
        store.save(s,{},force=True)
        restored,_=store.restore(13,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),initial))
    assert initialization_fingerprint(restored)==initialization_fingerprint(s)
