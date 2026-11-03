import json
from pathlib import Path
import jax,numpy as np,pytest
from hydra import initialize,compose
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest,validate_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training,prepare_protocol
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

def configs(tmp_path):
    with initialize(config_path='../configs',version_base=None):
        base=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer','env.max_episode_steps=4','pop_size=4','num_elites=1','batch_size=4','replay_buffer_capacity=128','warmup_iters=10','actor_update_interval=2','rl_updates_frac=0.25','source_batch_size=4','diagnostic_interval=2','num_eval_envs=2','eval_episodes=2','eval_interval=2','total_episodes=70','agent_network.critic_hidden_layer_sizes=[8,8]','agent_network.actor_hidden_layer_sizes=[8,8]','seed=9'])
    result=[]
    for name in ['uniform','fitness','td','utility']:
        c=OmegaConf.create(OmegaConf.to_container(base,resolve=True));c.arm=name;c.score_kind=name;c.alpha=0. if name=='uniform' else .5;c.diagnostic_enabled=name!='uniform';c.arm_manifest_path=str(tmp_path/'arms.json');result.append(c)
    prepare_arm_manifest(result,tmp_path/'arms.json');return {str(c.arm):c for c in result}

@pytest.mark.parametrize('arm',['uniform','utility'])
def test_six_generations_equal_three_then_explicit_resume_without_warmup(tmp_path,monkeypatch,arm):
    c=configs(tmp_path)[arm]
    full,full_report=run_training(c,tmp_path/'continuous',checkpoint_interval=2)
    paused,first=run_training(c,tmp_path/'split',checkpoint_interval=2,stop_after_generation=13)
    assert first['status']=='PAUSED' and int(paused.metrics.iterations)==13
    assert first['initial']['generation']==10 and [r['generation'] for r in first['evaluations']]==[12]
    paused_hash=initialization_fingerprint(paused)
    # A crash can leave an uncommitted log tail after its last checkpoint.
    # Keep it for provenance; it must not become a duplicated canonical record.
    old_events=Path(first['segment_directory'])/'events.jsonl'
    with old_events.open('a') as stream:stream.write('{"type":"uncommitted_test_tail","generation":14}\n')
    old_log=old_events.read_bytes()
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
    def forbidden(*args,**kwargs):raise AssertionError('resume must not call setup/warmup')
    monkeypatch.setattr(ERLUtilityReplayWorkflow,'setup',forbidden)
    monkeypatch.setattr(ERLUtilityReplayWorkflow,'warmup_step',forbidden)
    resumed,second=run_training(c,tmp_path/'split',resume_checkpoint=first['last_checkpoint'],checkpoint_interval=2)
    assert second['restored_state_sha256']==paused_hash
    assert second['status']=='COMPLETE' and int(resumed.metrics.iterations)==16
    assert second['budget']==full_report['budget']
    assert second['budget']['environment_episodes']==30 and second['budget']['environment_timesteps']==120
    assert second['budget']['ea_ask']==second['budget']['ea_tell']==6
    assert [r['generation'] for r in second['evaluations']]==[12,14,16]
    assert second['budget']['evaluation_episodes']==6
    assert second['final']==full_report['final']
    assert Path(first['segment_directory'],'events.jsonl').exists()
    assert Path(second['segment_directory'],'evaluations.csv').exists()
    assert old_events.read_bytes()==old_log
    # Serialization must be exact; cross-run Brax floats need only agree
    # numerically. Integer provenance, counters and both PRNG streams are exact.
    for x,y in zip(jax.tree.leaves(full),jax.tree.leaves(resumed)):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype
        if np.issubdtype(x.dtype,np.inexact):np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
        else:np.testing.assert_array_equal(x,y)
    if arm=='utility':
        assert bool(paused.utility_replay_state.distribution_prepared)
        assert bool(np.any(np.asarray(paused.utility_replay_state.score_table.validity)))
    with pytest.raises(ValueError,match='complete'):run_training(c,tmp_path/'split',resume_checkpoint=first['last_checkpoint'])

def test_protocol_preparation_is_four_arm_immutable_and_keeps_frozen_selection(tmp_path):
    frozen=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729')
    import hashlib
    before=hashlib.sha256((frozen/'checksums.sha256').read_bytes()).hexdigest()
    m=prepare_protocol(frozen,tmp_path/'protocol',seed=1,eval_interval=10,eval_episodes=128)
    assert m['final_generation']==3031 and m['effective_total_episodes']==100013
    for name in m['configs'].values():
        c=OmegaConf.load(tmp_path/'protocol'/name);validate_arm_manifest(c)
        assert c.eval_interval==10 and c.eval_episodes==128 and c.rl_updates_frac==1.
        assert c.save_replay_buffer and c.replay_buffer_capacity==1000000
    assert hashlib.sha256((frozen/'checksums.sha256').read_bytes()).hexdigest()==before
    with pytest.raises(FileExistsError):prepare_protocol(frozen,tmp_path/'protocol')
