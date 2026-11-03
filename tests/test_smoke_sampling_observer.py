import chex,jax,jax.numpy as jnp,numpy as np,pytest
from hydra import initialize,compose
from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import SmokeWorkflow,DrawRecorder,reconstruct_replay_draws
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import create_dummy_td3_trainmetric

def test_recorder_reads_actual_slots_and_rejects_rl_changes():
    r=DrawRecorder();r.record(jnp.array([0,1]),jnp.array([2,1]),jnp.array([True,False]),jnp.array([False,True]),jnp.array([True,True]))
    assert r.summary()['changed_ea']==1 and r.summary()['rl_changed']==0
    r.record(jnp.array([0,1]),jnp.array([2,0]),jnp.array([True,False]),jnp.array([False,True]),jnp.array([True,True]))
    assert r.summary()['rl_changed']==1

class DirectTrace(ERLUtilityReplayWorkflow):
    """Test-only callback. Its optimizer outputs never enter continued training."""
    def _observe_weighted_replay_draws(self,sampled,replay):
        jax.debug.callback(self.captured.append,sampled.final_indices,ordered=True)

@pytest.mark.parametrize('mode',['uniform','neutral','weighted'])
def test_actual_jitted_observer_preserves_all_state_and_proves_indices(mode):
    horizon=4 if mode=='weighted' else 1
    with initialize(config_path='../configs',version_base=None):
        c=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer',f'env.max_episode_steps={horizon}','pop_size=4','num_elites=1','batch_size=4','replay_buffer_capacity=128','warmup_iters=0','actor_update_interval=1','rl_updates_frac=0.25','source_batch_size=1','diagnostic_interval=10','num_eval_envs=1','eval_episodes=1','agent_network.critic_hidden_layer_sizes=[8,8]','agent_network.actor_hidden_layer_sizes=[8,8]'])
    if mode!='uniform':c.arm='utility';c.score_kind='utility';c.alpha=.5;c.replay_temperature=.5;c.freshness_window=30;c.min_resident=4;c.diagnostic_enabled=True
    base=type('AuditOff',(ERLUtilityReplayWorkflow,),{}).build_from_config(c,enable_jit=True)
    observed=type('AuditOn',(SmokeWorkflow,),{}).build_from_config(c,enable_jit=True)
    key=jax.random.PRNGKey(9);s1=base.setup(key);s2=observed.setup(key);chex.assert_trees_all_equal(s1,s2)
    captured=[];bound=observed._weighted_rl_update
    def remember(*values):captured.append(values);return bound(*values)
    if mode!='uniform':observed._weighted_rl_update=remember
    rows=[];checkpoint_inputs=None
    for generation in range(1,3 if mode=='uniform' else 12):
        a,s1=base.step(s1);b,s2=observed.step(s2)
        chex.assert_trees_all_equal(s1,s2);chex.assert_trees_all_equal(a.num_updates_per_iter,b.num_updates_per_iter)
        assert int(s2.metrics.iterations)==generation
        row=observed._latest_smoke_audit;rows.append(row)
        assert 0<=row['ess_fraction']==row['ess_fraction_audit64']<=1
        assert row['ess_audit_version']=='normalized-variance-float64-v1'
        assert row['effective_eligible_mass']==float(c.alpha)*row['eligible_baseline_mass']
        assert row['rl_changed']==0 and row['origin_changed']==0
        assert row['sampling_observation']=='deterministic_reconstruction'
        assert row['baseline_prng_transition_verified'] and row['intervention_terminal_key_verified']
        if generation==10:checkpoint_inputs=captured[-1]
    if mode=='weighted':
        assert checkpoint_inputs is not None and bool(checkpoint_inputs[-1].distribution_prepared)
        assert bool(jnp.any(checkpoint_inputs[-1].score_table.validity))
        assert any(r['probability_l1']>0 and r['changed_ea']>0 for r in rows if r['generation']>=10)
        active=[r for r in rows if r['probability_l1']>0]
        assert all(r['valid_score_source_count']>0 and r['eligible_source_count']>0 and r['distribution_prepared'] for r in active)
    elif mode=='neutral':
        assert checkpoint_inputs is not None and not bool(checkpoint_inputs[-1].distribution_prepared)
        assert all(r['probability_l1']==0 and r['changed_ea']==0 for r in rows)
    if checkpoint_inputs is not None:
        direct=type('DirectSamplingProof',(DirectTrace,),{}).build_from_config(c,enable_jit=True);direct.captured=[]
        out=direct._weighted_rl_update(*checkpoint_inputs);jax.block_until_ready(out);jax.effects_barrier()
        agent,opt,replay,learning_key,updates,utility=checkpoint_inputs
        trace=reconstruct_replay_draws(replay,learning_key,utility,num_updates=int(updates),num_draws=1,batch_size=4,uniform=False)
        np.testing.assert_array_equal(jnp.stack(direct.captured),trace.final_indices)
        chex.assert_trees_all_equal(trace.terminal_intervention_key,out[-1].intervention_key)
        carry=(learning_key,agent,opt,replay,create_dummy_td3_trainmetric(1),utility)
        hlo=observed._rl_sample_and_update_fn.lower(carry,()).compiler_ir(dialect='hlo').as_hlo_text()
        assert 'xla_ffi_python' not in hlo and 'debug_callback' not in hlo
