"""Registry tests use synthetic reports only, never training or real baselines."""
import importlib.util,json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/utility_replay_baseline_registry.py'
spec=importlib.util.spec_from_file_location('baseline_registry',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def config(seed=1):
    return dict(seed=seed,env=dict(env_name='swimmer',max_episode_steps=1000),total_episodes=100000,warmup_iters=10,pop_size=32,episodes_for_fitness=1,num_rl_agents=1,rollout_episodes=1,eval_interval=10,eval_episodes=128,num_eval_envs=128,rl_updates_frac=1.,rl_updates_mode='iter',actor_update_interval=2,batch_size=256,agent_network={'hidden':[256,256]},alpha=.5,arm='utility',score_kind='utility')

def report(c):
    p=r.protocol(c);rows=[dict(generation=g,environment_timesteps=(320+(g-10)*33)*1000,return_mean=float(g)) for g in p['evaluation_generations']]
    return dict(status='COMPLETE',initial=dict(generation=10,episodes=320,timesteps=320000,actor_optimizer_count=0,critic_optimizer_count=0),final=dict(generation=3031,episodes=100013,timesteps=100013000,actor_optimizer_count=48336000,critic_optimizer_count=96672000),budget=dict(arm=c['arm'],score_kind=c['score_kind'],environment_episodes=99693,environment_timesteps=99693000,rl_updates=48336000,ea_ask=3021,ea_tell=3021,batch_size=256,evaluation_checkpoints=p['evaluation_generations'],evaluation_episodes=38784,configured_total_episodes=100000),evaluations=rows)

def test_formal_rounding_and_final_evaluation():
    p=r.protocol(config());assert (p['generation'],p['episodes'],len(p['evaluation_generations']))==(3031,100013,303);assert p['evaluation_generations'][-2:]==[3030,3031]

def test_complete_actual_counters_and_curve_metrics():
    m=r.validate_report(report(config()),config());assert m['final_return']==3031;assert m['late_evaluation_count']==5;assert m['auc']>0

@pytest.mark.parametrize('field',['episodes','critic_optimizer_count','actor_optimizer_count','generation'])
def test_actual_counter_mismatch_rejected(field):
    q=report(config());q['final'][field]+=1
    with pytest.raises(ValueError):r.validate_report(q,config())

@pytest.mark.parametrize('change',['missing','duplicate','nan','budget'])
def test_missing_or_invalid_actual_evaluation_rejected(change):
    q=report(config())
    if change=='missing':q['evaluations'].pop()
    if change=='duplicate':q['evaluations'][1]=q['evaluations'][0]
    if change=='nan':q['evaluations'][0]['return_mean']=float('nan')
    if change=='budget':q['budget']['evaluation_episodes']-=1
    with pytest.raises(ValueError):r.validate_report(q,config())

def test_d_only_parameters_do_not_change_common_contract():
    a=config();b=config();b.update(arm='uniform',score_kind='uniform',alpha=0,diagnostic_enabled=False)
    assert r.common_config(a)==r.common_config(b)

@pytest.mark.parametrize('key,value',[('batch_size',128),('eval_interval',1),('rl_updates_frac',.5),('total_episodes',9890),('agent_network',{'hidden':[64]})])
def test_shared_change_rejected(key,value):
    a=r.contract(config(),{'shared.py':'old'},{'python':'3'});c=config();c[key]=value;b=r.contract(c,{'shared.py':'old'},{'python':'3'})
    with pytest.raises(ValueError,match=key):r.compatible(a,b)

def test_shared_code_or_dependencies_change_rejected():
    a=r.contract(config(),{'shared.py':'old'},{'jax':'1'})
    for b in [r.contract(config(),{'shared.py':'new'},{'jax':'1'}),r.contract(config(),{'shared.py':'old'},{'jax':'2'})]:
        with pytest.raises(ValueError):r.compatible(a,b)

def test_atomic_frozen_entry_refuses_overwrite(tmp_path):
    r.freeze_json(tmp_path/'entry.json',{'status':'frozen_baseline'})
    with pytest.raises(FileExistsError):r.freeze_json(tmp_path/'entry.json',{'status':'pending'})
    assert json.loads((tmp_path/'entry.json').read_text())['status']=='frozen_baseline'

def test_incomplete_run_cannot_freeze():
    q=report(config());q['status']='PAUSED'
    with pytest.raises(ValueError):r.validate_report(q,config())

def test_file_checksum_tampering_rejected(tmp_path):
    (tmp_path/'curve.csv').write_text('original');r.seal(tmp_path);r.verify_seal(tmp_path);(tmp_path/'curve.csv').write_text('edited')
    with pytest.raises(ValueError):r.verify_seal(tmp_path)

def test_seed_statistics_use_independent_seeds():
    rows=[dict(seed=i,final_return=float(i),auc=float(i*10),late_mean=float(i)) for i in [1,2,3]]
    s=r.aggregate(rows);assert s['n_seeds']==3;assert s['final_return']['sample_sd']==1
    with pytest.raises(ValueError):r.aggregate(rows+[rows[0]])

def test_incompatible_seed_pair_rejected():
    with pytest.raises(ValueError):r.compatible(r.contract(config(1),{},{}),r.contract(config(2),{},{}))

def completed_artifacts(tmp_path,arm='A',seed=1):
    from omegaconf import OmegaConf
    c=config(seed);c.update(arm=r.ARMS[arm],score_kind=r.ARMS[arm]);run=tmp_path/'runs'/arm/f'seed_{seed}';run.mkdir(parents=True)
    cfg=tmp_path/f'config_{arm}_{seed}.yaml';OmegaConf.save(OmegaConf.create(c),cfg);OmegaConf.save(OmegaConf.create(c),run/'resolved_config.yaml')
    segment=run/'segments'/'segment';segment.mkdir(parents=True);q=report(c)
    cp=segment/'checkpoints'/'3031';cp.mkdir(parents=True);(cp/'state').write_bytes(b'synthetic test checkpoint')
    q['last_checkpoint']=str(cp);r.freeze_json(segment/'report.json',q)
    r.freeze_json(segment/'checkpoint_checksums'/'3031.json',{'state':r.sha(cp/'state')})
    r.freeze_json(run/'completion.json',dict(segment=str(segment),report_sha256=r.sha(segment/'report.json')))
    r.freeze_json(run/'manifest.json',dict(commit='test',resolved_config_sha256=r.sha(run/'resolved_config.yaml'),checkpoint_policy=dict(interval=50,keep=3)))
    e=dict(status='READY_NOT_RUN',arm=arm,version='D0' if arm=='D' else 'ABC-v1',seed=seed,commit='test',config_path=str(cfg),output_directory=str(run),contract=r.contract(c,{},{}))
    attempt=tmp_path/'attempts'/e['version']/arm/f'seed_{seed}'/'attempt1';attempt.mkdir(parents=True);r.freeze_json(attempt/'result.json',dict(status='COMPLETE',wall_seconds=10.))
    return e,run,segment

def test_completed_archive_copies_and_freezes_actual_artifacts(tmp_path):
    e,run,segment=completed_artifacts(tmp_path);result=r.archive(tmp_path,e);p=Path(result['directory']);r.verify_seal(p)
    assert r.load(p/'entry.json')['status']=='frozen_baseline'
    assert (p/'evaluations.csv').exists() and (p/'checkpoints/3031/state').exists() and (p/'attempt_history/attempt1/result.json').exists()
    with pytest.raises(FileExistsError):r.archive(tmp_path,e)
    (run/'resolved_config.yaml').write_text('tampered source')
    r.verify_seal(p)

def test_archive_rejects_checkpoint_tamper(tmp_path):
    e,run,segment=completed_artifacts(tmp_path);(segment/'checkpoints/3031/state').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checkpoint'):r.archive(tmp_path,e)
    assert not (tmp_path/'registry/baselines/A/seed_1').exists()

def test_three_seed_four_arm_comparison_uses_frozen_curves(tmp_path):
    for arm in 'ABCD':
        for seed in [1,2,3]:
            e,_,_=completed_artifacts(tmp_path,arm,seed);r.archive(tmp_path,e)
    result=r.compare(tmp_path,'D0');m=r.load(Path(result['directory'])/'metrics.json')
    assert all(v['aggregate']['n_seeds']==3 for v in m['arms'].values())

def test_changed_protocol_cannot_compare_frozen_baseline(tmp_path):
    for arm in 'ABCD':
        for seed in [1,2,3]:
            e,_,_=completed_artifacts(tmp_path,arm,seed)
            if arm=='D':e['contract']['config']['batch_size']=128
            r.archive(tmp_path,e)
    with pytest.raises(ValueError,match='batch_size'):r.compare(tmp_path,'D0')

def test_prepare_twelve_configs_uses_real_native_manifest_validation(tmp_path,monkeypatch):
    from omegaconf import OmegaConf
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest,validate_arm_manifest,validate_arm_set
    frozen=tmp_path/'frozen';frozen.mkdir();configs=[]
    for arm,kind in r.ARMS.items():
        c=config();c.update(arm=kind,score_kind=kind,alpha=0 if arm=='A' else .5,diagnostic_enabled=arm!='A',replay_temperature=.5,freshness_window=30,min_resident=4,diagnostic_interval=10,num_elites=4,tournament_size=3,rl_injection_interval=10,replay_buffer_capacity=1000000,arm_manifest_path=str(frozen/'arm_manifest.json'),checkpoint=dict(enable=False))
        c=OmegaConf.create(c);configs.append(c);OmegaConf.save(c,frozen/f'arm_{arm}.yaml')
    prepare_arm_manifest(configs,frozen/'arm_manifest.json');r.seal(frozen)
    monkeypatch.setattr(r,'provenance',lambda:dict(commit='test',shared_code={},environment={}))
    root=tmp_path/'formal';result=r.prepare(frozen,root)
    assert result['manifests']==12 and len(list((root/'manifests').glob('*.json')))==12
    assert not (root/'runs').exists() and not (root/'registry').exists()
    for seed in [1,2,3]:
        cs=[OmegaConf.load(root/'protocols'/f'seed_{seed}'/f'arm_{arm}.yaml') for arm in 'ABCD'];validate_arm_set(cs)
        for c in cs:
            validate_arm_manifest(c);assert c.seed==seed and c.eval_interval==10 and c.eval_episodes==128
    r.verify_seal(root)
    with pytest.raises(FileExistsError):r.prepare(frozen,root)
    evidence=tmp_path/'dev.json';dev_report=tmp_path/'dev-report.json';r.freeze_json(dev_report,{'status':'synthetic test evidence'})
    r.freeze_json(evidence,dict(seed=2027,head='test',status='PASS',report_path=str(dev_report),report_sha256=r.sha(dev_report)))
    r.register_d(root,'D1',root/'protocols',evidence)
    entry=r.get_entry(root,'D',2,'D1');assert entry['version']=='D1' and entry['status']=='READY_NOT_RUN'
    assert '/runs/D1/' in entry['output_directory'] and not (root/'runs').exists()
    with pytest.raises(FileExistsError):r.register_d(root,'D1',root/'protocols',evidence)

def test_missing_seed_cannot_produce_three_seed_statistics():
    with pytest.raises(ValueError):r.aggregate([dict(seed=1,auc=1.,final_return=1.,late_mean=1.)])

def test_checkpoint_sd_not_used_as_cross_seed_variance():
    rows=[dict(seed=s,auc=1.,final_return=3.,late_mean=2.,late_checkpoint_sd=999.) for s in [1,2,3]]
    assert r.aggregate(rows)['final_return']['sample_sd']==0.

def test_register_new_d_rejects_formal_seed_development_evidence(tmp_path):
    p=tmp_path/'dev.json';r.freeze_json(p,dict(seed=1))
    with pytest.raises(ValueError,match='independent'):r.register_d(tmp_path,'D1',tmp_path,p)
