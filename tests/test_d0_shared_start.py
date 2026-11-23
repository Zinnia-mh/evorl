import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.types import State
from evorl.algorithms.erl.erl_td3.d0_shared_start import (
    component_hashes, seal_parent, restore_parent, save_draw_audit,
)
from evorl.algorithms.erl.erl_td3.erl_utility_replay import init_utility_replay_state
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore


@pytest.fixture
def parent(tmp_path):
    config = OmegaConf.create(dict(seed=2081, alpha=.5))
    state = State(
        agent_state=State(actor=jnp.arange(2), critic=jnp.arange(4).reshape(2,2),
                          target_critic=jnp.arange(4).reshape(2,2)),
        opt_state=State(actor=State(count=jnp.array(144000), moments=jnp.arange(5)),
                        critic=State(count=jnp.array(288000), moments=jnp.arange(5))),
        ec_opt_state=State(pop=jnp.arange(4), key=jax.random.PRNGKey(1)),
        replay_buffer_state=State(data=jnp.arange(4), current_index=jnp.array(0),
                                  source_token=jnp.array([7,7,8,-1])),
        utility_replay_state=init_utility_replay_state(4,jax.random.PRNGKey(2)),
        key=jax.random.PRNGKey(3),
        metrics=State(iterations=jnp.array(19), sampled_episodes=jnp.array(617),
                      sampled_timesteps=jnp.array(617000)),
    )
    directory=tmp_path/'permanent'
    directory.mkdir()
    (directory/'resolved_config.yaml').write_text(OmegaConf.to_yaml(config))
    runtime=dict(run_id='parent-run', evaluations=[], budget=dict(eval_episodes=0, updates=144000),
                 target_generation=120)
    with PilotCheckpointStore(directory/'checkpoints',config,interval=1,keep=2) as store:
        cp=store.save(state,runtime,force=True)
    return config,state,cp,runtime


def test_full_state_fork_and_seal(parent):
    config,state,cp,runtime=parent
    before={p:p.read_bytes() for p in cp.rglob('*') if p.is_file()}
    seal_parent(cp,config,state,runtime)
    a,r1=restore_parent(cp,config,state,'d0')
    b,r2=restore_parent(cp,config,state,'p')
    assert component_hashes(a)==component_hashes(b)==component_hashes(state)
    assert r1['run_id']=='d0' and r2['run_id']=='p'
    assert r1['evaluations']==r2['evaluations']==[]
    assert all(p.read_bytes()==content for p,content in before.items())
    with pytest.raises(ValueError,match='configuration'):
        restore_parent(cp,OmegaConf.create(dict(seed=2081,alpha=.4)),state,'x')
    with pytest.raises(FileExistsError):
        seal_parent(cp,config,state,runtime)


def test_trace_reads_actual_indices(tmp_path):
    replay=State(origin_type=jnp.array([0,1,0]), source_token=jnp.array([19,-1,20]),
                 buffer_size=jnp.array(3))
    trace=(jnp.array([[[0,1]]]),jnp.array([[[2,1]]]),
           jnp.zeros((1,2),dtype=jnp.uint32),jnp.zeros((1,2),dtype=jnp.uint32),
           jnp.zeros((1,2),dtype=jnp.uint32))
    path=tmp_path/'trace.npz'
    save_draw_audit(path,trace,replay,8)
    with np.load(path) as data:
        np.testing.assert_array_equal(data['final_indices'],[[[2,1]]])
        np.testing.assert_array_equal(data['ea_source_tokens'],[20])
        np.testing.assert_array_equal(data['ea_source_counts'],[1])
        np.testing.assert_array_equal(data['ea_per_batch'],[[1]])
        np.testing.assert_array_equal(data['rl_per_batch'],[[1]])



def test_audit_exactly_preserves_real_td3_update_and_resume(tmp_path, monkeypatch):
    from hydra import initialize, compose
    from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
    from evorl.algorithms.erl.erl_td3.source_replay import build_weighted_source_distribution
    from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import restore_template

    def forbidden(*args, **kwargs):
        raise AssertionError('no setup, warmup or rollout in preflight')

    for method in ('setup', 'warmup_step', '_ec_rollout', '_rl_rollout', 'evaluate'):
        monkeypatch.setattr(ERLUtilityReplayWorkflow,method,forbidden)

    with initialize(config_path='../configs',version_base=None):
        config=compose(config_name='config',overrides=[
            'agent=erl/erl-utility-replay','env=brax/swimmer','env.max_episode_steps=4',
            'pop_size=4','num_elites=1','batch_size=4','replay_buffer_capacity=8',
            'actor_update_interval=2','agent_network.actor_hidden_layer_sizes=[8,8]',
            'agent_network.critic_hidden_layer_sizes=[8,8]','arm=utility',
            'score_kind=utility','alpha=0.5','diagnostic_enabled=true'])
    native=type('Native',(ERLUtilityReplayWorkflow,),{}).build_from_config(config,enable_jit=True)
    observed=type('Observed',(AuditedD0,),{}).build_from_config(config,enable_jit=True)
    state=restore_template(native,abstract=False)
    replay=state.replay_buffer_state.replace(
        buffer_size=jnp.asarray(8,dtype=state.replay_buffer_state.buffer_size.dtype),
        origin_type=jnp.array([0,0,0,0,0,0,1,1],dtype=jnp.int8),
        source_token=jnp.array([0,0,0,1,1,1,-1,-1]))
    replay=replay.replace(data=replay.data.replace(
        rewards=jnp.arange(8,dtype=replay.data.rewards.dtype)))
    utility=state.utility_replay_state
    table=utility.score_table._replace(
        source_token=utility.score_table.source_token.at[:2].set(jnp.array([0,1])),
        resident_count=utility.score_table.resident_count.at[:2].set(jnp.array([3,3])),
        weighted_eligible=utility.score_table.weighted_eligible.at[:2].set(True),
        rank_normalized_score=utility.score_table.rank_normalized_score.at[:2].set(jnp.array([0.,1.])))
    utility=utility._replace(score_table=table,distribution_prepared=jnp.asarray(True),
        distribution=build_weighted_source_distribution(table,replay.source_token,replay.origin_type==0,.5,.5))
    state=state.replace(metrics=state.metrics.replace(iterations=jnp.asarray(19,dtype=state.metrics.iterations.dtype)),
                        replay_buffer_state=replay,utility_replay_state=utility)
    from evorl.algorithms.erl.erl_td3.d0_placebo import permuted_ranks
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import reconstruct_replay_draws
    for arm in ('D0','P'):
        t=table if arm=='D0' else permuted_ranks(table._replace(
            measured_generation=table.measured_generation.at[:2].set(20)),seed=2081)
        u=utility._replace(distribution=build_weighted_source_distribution(
            t,replay.source_token,replay.origin_type==0,.5,.5))
        key=state.key
        def update(workflow, start, steps, key):
            return workflow._weighted_rl_update(start.agent_state,start.opt_state,
                start.replay_buffer_state,key,steps,start.utility_replay_state)
        start=state.replace(utility_replay_state=u)
        plain=update(native,start,2,key)
        tapped=update(observed,start,2,key)
        assert jax.tree.structure(plain)==jax.tree.structure(tapped)
        for x,y in zip(jax.tree.leaves(plain),jax.tree.leaves(tapped),strict=True):
            np.testing.assert_array_equal(x,y)
        trace=observed.learner_trace
        expected=reconstruct_replay_draws(replay,key,u,num_updates=2,num_draws=2,batch_size=4,uniform=False)
        np.testing.assert_array_equal(trace[0],expected.baseline_indices)
        np.testing.assert_array_equal(trace[1],expected.final_indices)
        # Validate exact payload at the critic boundary using the same TD3 update.
        batches=jax.tree.map(lambda data:data[trace[1][0]].reshape(2,1,4,*data.shape[1:]),replay.data)
        (agent,opt),_=native._rl_update_fn(start.agent_state,start.opt_state,batches,trace[3][0])
        first=update(observed,start,1,key)
        for x,y in zip(jax.tree.leaves((agent,opt)),jax.tree.leaves(first[1:3]),strict=True):
            np.testing.assert_array_equal(x,y)
        mid=start.replace(agent_state=agent,opt_state=opt,utility_replay_state=first[3],
                          key=trace[2][1],metrics=start.metrics.replace(iterations=jnp.asarray(20,dtype=start.metrics.iterations.dtype)))
        with PilotCheckpointStore(tmp_path/arm/'checkpoints',config,interval=1,keep=2) as store:
            store.save(mid,dict(run_id=arm),force=True)
        with PilotCheckpointStore(tmp_path/arm/'checkpoints',config,read_only=True) as store:
            restored,runtime=store.restore(20,restore_template(observed))
        assert runtime['run_id']==arm and component_hashes(restored)==component_hashes(mid)
        tail=update(observed,restored,1,restored.key)
        np.testing.assert_array_equal(observed.learner_trace[1][0],trace[1][1])
        for x,y in zip(jax.tree.leaves(tail),jax.tree.leaves(plain),strict=True):
            np.testing.assert_array_equal(x,y)


def test_permanent_parent_is_outside_child_rotation(parent,tmp_path):
    config,state,cp,runtime=parent
    seal_parent(cp,config,state,runtime)
    with PilotCheckpointStore(tmp_path/'D0'/'checkpoints',config,interval=1,keep=2) as child:
        for g in (20,21,22):
            child.save(state.replace(metrics=state.metrics.replace(iterations=jnp.asarray(g))),
                       dict(run_id='d0'),force=True)
        assert child.steps()==[21,22]
    restored,_=restore_parent(cp,config,state,'p')
    assert component_hashes(restored)==component_hashes(state)


def test_runner_forks_and_resumes_without_repeating_operations(parent,tmp_path,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from evorl.algorithms.erl.erl_td3 import utility_replay_training as runner
    config,state,cp,runtime=parent
    # Rebuild the synthetic archive under the strict runner contract.
    config=OmegaConf.merge(config,dict(arm='utility',score_kind='utility',
        diagnostic_enabled=True,replay_temperature=.5,freshness_window=30,min_resident=4,
        workflow_cls=runner.WORKFLOW, save_replay_buffer=True, checkpoint=dict(enable=False),
        enable_jit=True,pop_size=32,episodes_for_fitness=1,num_rl_agents=1,
        rollout_episodes=1,batch_size=4,actor_update_interval=2,eval_interval=10,
        total_episodes=683,eval_episodes=128,num_eval_envs=128))
    permanent=tmp_path/'runner-parent'
    permanent.mkdir()
    OmegaConf.save(config,permanent/'resolved_config.yaml')
    budget=runner.BudgetAudit(config,initial_episodes=320,initial_iteration=10,
                              initial_timesteps=320000)
    for _ in range(9):
        budget.observe_step(episodes=33,timesteps=33000,rl_updates=16000,
                            batch_size=4,ea_ask=1,ea_tell=1)
    runtime=dict(run_id='runner-parent',initial=dict(generation=10,episodes=320,
                 timesteps=320000,actor_optimizer_count=0,critic_optimizer_count=0),
                 evaluations=[],target_generation=21,budget=runner._budget(budget))
    with PilotCheckpointStore(permanent/'checkpoints',config,interval=1) as store:
        cp=store.save(state,runtime,force=True)
    seal_parent(cp,config,state,runtime)
    monkeypatch.setattr(runner,'validate_arm_manifest',lambda c: None)
    monkeypatch.setattr(runner,'restore_template',lambda workflow: state)

    class StaticWorkflow:
        @classmethod
        def build_from_config(cls, config, **kwargs):
            wf=cls()
            wf.config=config
            wf.recorder=SimpleNamespace(init=lambda: None)
            return wf

        def init(self, key):
            pytest.fail('fork/resume called initialization')

        def close(self):
            pass

        def step(self, old):
            self._budget_audit.observe_step(episodes=33,timesteps=33000,rl_updates=16000,
                        batch_size=4,ea_ask=1,ea_tell=1)
            metrics=old.metrics.replace(iterations=old.metrics.iterations+1,
                    sampled_episodes=old.metrics.sampled_episodes+33,
                    sampled_timesteps=old.metrics.sampled_timesteps+33000)
            opt=State(actor=old.opt_state.actor.replace(count=old.opt_state.actor.count+16000),
                      critic=old.opt_state.critic.replace(count=old.opt_state.critic.count+32000))
            return State(),old.replace(metrics=metrics,opt_state=opt,key=jax.random.split(old.key)[0])

        def evaluate(self, old):
            self._budget_audit.observe_eval(int(old.metrics.iterations),episodes=128)
            return State(rl_episode_returns=jnp.array([0.]),rl_episode_lengths=jnp.array([1000.])),old

    # Empty synthetic metrics exposes the same serialization API as real metrics.
    monkeypatch.setattr(State,'to_local_dict',lambda self: dict(self),raising=False)

    class Fork:
        workflow_type=StaticWorkflow
        parent=cp

        def __init__(self, arm):
            self.identity=dict(arm=arm)

        def configure(self, workflow, segment):
            pass

        def fork_state(self, workflow, run_id):
            return restore_parent(cp,workflow.config,state,run_id)

        def before_generation(self, old):
            assert int(old.metrics.iterations) in (19,20)

        def record(self, *args):
            pass

    reports={}
    for arm in ('D0','P'):
        _,reports[arm]=runner.run_training(config,tmp_path/arm,ablation=Fork(arm),
                      fork=True,stop_after_generation=20,checkpoint_interval=1)
    first=reports['D0']
    assert first['restored_state_sha256']==reports['P']['restored_state_sha256']
    assert not list((tmp_path/'D0').glob('segments/*/checkpoints/19'))
    with pytest.raises(ValueError,match='another run directory'):
        runner.run_training(config,tmp_path/'P',ablation=Fork('P'),checkpoint_interval=1,
                            resume_checkpoint=first['last_checkpoint'])
    for arm in ('D0','P'):
        first=reports[arm]
        _,end=runner.run_training(config,tmp_path/arm,ablation=Fork(arm),checkpoint_interval=1,
                                  resume_checkpoint=first['last_checkpoint'])
        assert end['restored_state_sha256']==first['state_sha256']
        assert [r['generation'] for r in end['evaluations']]==[20,21]
        assert end['budget']['ea_ask']==end['budget']['ea_tell']==11
        assert end['final']['actor_optimizer_count']==176000
        assert end['final']['critic_optimizer_count']==352000
    assert cp.is_dir()


def test_production_step_passes_only_placebo_ranks_to_learner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evorl.algorithms.erl.erl_td3 import erl_utility_replay as production
    from evorl.algorithms.erl.erl_td3.d0_placebo import Ablation
    from evorl.algorithms.erl.erl_td3.source_replay import SCORE_KIND_UTILITY

    # Two interchangeable cohorts plus a smaller resident source and an expired source.
    counts = np.array([4,4,4,4,4,3,4])
    measured = np.array([20,20,20,10,10,20,0])
    ranks = np.array([0.,.5,1.,.2,.8,.9,.7],np.float32)
    tokens = jnp.asarray(np.repeat(np.arange(len(counts)),counts),dtype=jnp.int32)
    replay = State(origin_type=jnp.zeros(tokens.shape,dtype=jnp.int8),
                   source_token=tokens,buffer_size=jnp.asarray(len(tokens)))
    utility = init_utility_replay_state(len(tokens),jax.random.PRNGKey(3),len(counts))
    table = utility.score_table._replace(
        source_token=jnp.arange(len(counts),dtype=jnp.int32),
        measured_generation=jnp.asarray(measured),
        rank_normalized_score=jnp.asarray(ranks),
        raw_score=jnp.asarray(ranks+100),
        score_kind=jnp.full((len(counts),),SCORE_KIND_UTILITY,dtype=jnp.int8),
        version=jnp.zeros(len(counts),dtype=jnp.int32),
        validity=jnp.ones(len(counts),dtype=bool))
    state = State(key=jax.random.PRNGKey(2081), agent_state=State(),opt_state=State(),
                  ec_opt_state=State(),replay_buffer_state=replay,
                  utility_replay_state=utility._replace(score_table=table),
                  metrics=State(iterations=jnp.asarray(30)))
    config = SimpleNamespace(alpha=.5,seed=2081,pop_size=7,episodes_for_fitness=1,
                             freshness_window=30,min_resident=4,replay_temperature=.5,
                             rl_updates_mode='iter',rl_updates_frac=1.,actor_update_interval=2)
    monkeypatch.setattr(production,'erl_replace_td3_actor_params',lambda agent,params:agent)
    metrics=SimpleNamespace(episode_lengths=jnp.ones((7,1)),episode_returns=jnp.arange(7.)[:,None])

    class ReachedLearner(Exception):
        pass

    def intercept(agent,opt,rb,key,updates,next_utility):
        captured.append((rb,key,int(updates),next_utility))
        raise ReachedLearner

    results={}
    captured=[]
    for arm in ('D0','P'):
        wf=SimpleNamespace(config=config,ec_optimizer=SimpleNamespace(ask=lambda ec:(None,ec)),
            _ec_rollout=lambda *x:(metrics,None,replay),
            _rl_rollout=lambda *x:(None,None,replay),
            _weighted_rl_update=intercept,permutation={},_latest_score_lifecycle_audit={'ordering':[]})
        Ablation(arm,2081,'fixture').configure(wf,tmp_path)
        with pytest.raises(ReachedLearner):
            production.ERLUtilityReplayWorkflow._weighted_step(wf,state)
        results[arm]=(captured[-1],wf.permutation)

    (rb,d_key,d_updates,d),_=results['D0']
    (_,p_key,p_updates,p),mapping=results['P']
    assert rb is replay and d_updates==p_updates
    np.testing.assert_array_equal(d_key,jax.random.split(state.key,4)[3])
    np.testing.assert_array_equal(p_key,d_key)
    np.testing.assert_array_equal(d.intervention_key,p.intervention_key)
    np.testing.assert_array_equal(d.score_table.weighted_eligible,[1,1,1,1,1,0,0])
    for x,y in zip(jax.tree.leaves(d.score_table),jax.tree.leaves(p.score_table),strict=True):
        np.testing.assert_array_equal(x,y)
    effective=ranks.copy()
    for token,donor,gen,resident,old,new in mapping['eligible_mapping']:
        assert measured[token]==measured[donor]==gen
        assert counts[token]==counts[donor]==resident
        assert (old,new)==(ranks[token],ranks[donor])
        effective[token]=new
    assert mapping['changed_ranks']>0
    for dist,rank in ((d.distribution,ranks),(p.distribution,effective)):
        mask=np.arange(len(counts))<5
        base=counts/counts.sum()
        logits=np.exp(rank[mask]/.5)
        expected=base.copy()
        expected[mask]=.5*(base[mask]+base[mask].sum()*logits/logits.sum())
        np.testing.assert_allclose(dist.source_probability,expected,rtol=1e-6)
        np.testing.assert_allclose(dist.transition_probability,
                                   np.repeat(expected/counts,counts),rtol=1e-6)
    np.testing.assert_array_equal(np.sort(d.distribution.weighted_source_probability),
                                  np.sort(p.distribution.weighted_source_probability))
    assert np.any(np.asarray(d.distribution.source_probability)!=np.asarray(p.distribution.source_probability))


def test_resume_requires_actual_audit_bound_to_restored_state(parent,tmp_path):
    import json
    from evorl.algorithms.erl.erl_td3.d0_shared_start import SharedStartAblation, sha
    config,state,cp,runtime=parent
    seal_parent(cp,config,state,runtime)
    state,runtime=restore_parent(cp,config,state,'child')
    child=state.replace(metrics=state.metrics.replace(iterations=jnp.asarray(20)))
    segment=tmp_path/'P'/'segments'/'one'
    with PilotCheckpointStore(segment/'checkpoints',config,interval=1) as store:
        checkpoint=store.save(child,runtime,force=True)
    with PilotCheckpointStore(checkpoint.parent,config,read_only=True) as store:
        restored,_=store.restore(20,child)
    audit=SharedStartAblation('P',2081,cp)
    with pytest.raises(ValueError,match='lacks actual'):
        audit.validate_resume(checkpoint,restored)
    artifact=segment/'critic-draws-20.npz'
    replay=State(origin_type=jnp.array([0,1]),source_token=jnp.array([12,-1]),
                 buffer_size=jnp.asarray(2))
    indices=jnp.array([[[0,1]]])
    keys=jnp.zeros((1,2),dtype=jnp.uint32)
    save_draw_audit(artifact,(indices,indices,keys,keys,keys),replay,2)
    receipt=segment/'critic-draws-20.json'
    receipt.write_text(json.dumps(dict(artifact_sha256=sha(artifact),state=component_hashes(child))))
    audit.validate_resume(checkpoint,restored)
    with pytest.raises(ValueError,match='does not belong'):
        audit.validate_resume(checkpoint,restored.replace(key=jax.random.PRNGKey(99)))
    with artifact.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        audit.validate_resume(checkpoint,restored)



def test_cross_process_parent_restore_replays_first_draw(parent):
    import json
    import pickle
    import subprocess
    import sys
    from evorl.algorithms.erl.erl_td3.source_replay import draw_baseline_indices

    config,state,checkpoint,runtime=parent
    seal_parent(checkpoint,config,state,runtime)
    args=(str(checkpoint),OmegaConf.to_container(config),jax.tree.map(np.asarray,state))
    code = """
import json, pickle, sys
import jax
import numpy as np
from omegaconf import OmegaConf
from evorl.types import State
from evorl.algorithms.erl.erl_td3.d0_shared_start import component_hashes, restore_parent
from evorl.algorithms.erl.erl_td3.source_replay import draw_baseline_indices
cp,config,template=pickle.load(sys.stdin.buffer)
assert jax.default_backend()=='cpu'
state,runtime=restore_parent(cp,OmegaConf.create(config),template,'separate-process')
next_key,draw_key=jax.random.split(state.key)
draw=draw_baseline_indices(State(buffer_size=np.int32(4)),draw_key,8)
print(json.dumps(dict(state=component_hashes(state),key=next_key.tolist(),draw=draw.tolist(),
                      run_id=runtime['run_id'])))
"""
    process=subprocess.run([sys.executable,'-c',code],input=pickle.dumps(args),
                           capture_output=True,check=True)
    result=json.loads(process.stdout.splitlines()[-1])
    next_key,draw_key=jax.random.split(state.key)
    draw=draw_baseline_indices(State(buffer_size=jnp.asarray(4)),draw_key,8)
    assert result==dict(state=component_hashes(state),key=next_key.tolist(),
                        draw=draw.tolist(),run_id='separate-process')

