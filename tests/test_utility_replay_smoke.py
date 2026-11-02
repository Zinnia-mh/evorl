import copy,json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.utility_replay_smoke import prepare_smoke,verify_smoke_manifest,validate_smoke_outputs

FROZEN=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729')

def add_provenance(reports):
    for a,report in reports.items():
        kind=dict(A='uniform',B='fitness',C='td',D='utility')[a]
        report.update(seed=1,initialization_sha256='a'*64)
        report['budget'].update(arm=kind,score_kind=kind)
        for row in report['rows']:row.update(baseline_prng_transition_verified=True,intervention_terminal_key_verified=True,sampling_observation='deterministic_reconstruction')

def test_smoke_manifest_uses_one_environment_one_seed_four_arms(tmp_path):
    root=tmp_path/'smoke';m=prepare_smoke(FROZEN,root,seed=1,commit='a'*40)
    assert m['seed']==1 and m['environment']=='swimmer' and set(m['configs'])==set('ABCD')
    configs=[OmegaConf.load(root/name) for name in m['configs'].values()]
    assert {c.total_episodes for c in configs}=={716}
    assert {c.env.max_episode_steps for c in configs}=={1000}
    assert {c.rl_updates_frac for c in configs}=={.001}
    assert configs[0].alpha==0 and {c.alpha for c in configs[1:]}=={.5}
    assert {c.freshness_window for c in configs}=={30}
    assert verify_smoke_manifest(root)==m
    with pytest.raises(FileExistsError):prepare_smoke(FROZEN,root,seed=1,commit='a'*40)
    with pytest.raises(ValueError):prepare_smoke(FROZEN,tmp_path/'bad',seed=104729,commit='a'*40)

def test_smoke_manifest_references_frozen_resolved_config(tmp_path):
    m=prepare_smoke(FROZEN,tmp_path/'smoke',seed=2,commit='a'*40)
    assert m['frozen_directory']==str(FROZEN) and len(m['frozen_checksums_sha256'])==64
    assert m['source_collection_commit'].startswith('8240863') and m['source_analysis_commit'].startswith('dbcb4f9')
    root=tmp_path/'smoke';c=OmegaConf.load(root/'arm_D.yaml');c.alpha=.9;OmegaConf.save(c,root/'arm_D.yaml')
    with pytest.raises(ValueError):verify_smoke_manifest(root)

def test_smoke_output_validator_checks_all_gate_counters(tmp_path):
    root=tmp_path/'smoke';prepare_smoke(FROZEN,root,seed=1,commit='a'*40)
    with pytest.raises(ValueError,match='missing'):validate_smoke_outputs(root)

def test_all_four_prepared_configs_build_through_formal_manifest_checks(tmp_path):
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import SmokeWorkflow
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import validate_arm_manifest
    root=tmp_path/'smoke';manifest=prepare_smoke(FROZEN,root,seed=1,commit='a'*40)
    verify_smoke_manifest(root)
    for label in 'ABCD':
        config=OmegaConf.load(root/manifest['configs'][label]);validate_arm_manifest(config)
        workflow=type('Preflight'+label,(SmokeWorkflow,),{}).build_from_config(config,enable_jit=True)
        assert workflow.replay_buffer.capacity==1000000 and workflow.config.score_kind==config.score_kind

@pytest.mark.parametrize('field,value',[('rl_updates',9),('environment_timesteps',2),('ea_ask',0),('evaluation_episodes',8),('batch_size',1)])
def test_smoke_budget_mismatch_or_missing_audit_is_rejected(field,value):
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke import validate_reports
    base=dict(complete=True,budget=dict(configured_total_episodes=716,environment_episodes=396,environment_timesteps=396000,rl_updates=192,batch_size=256,ea_ask=12,ea_tell=12,evaluation_checkpoints=[20,22],evaluation_episodes=4),initial=dict(episodes=320,timesteps=320000,generation=10),final=dict(episodes=716,timesteps=716000,generation=22),rows=[dict(generation=i,observed_batches=32,ea_slots=8000,rl_slots=192,changed_ea=100,rl_changed=0,origin_changed=0,finite=True,probability_l1=.01 if i>=20 else 0.,eligible_baseline_mass=.05 if i>=20 else 0.,ess_fraction=.99,reference_fingerprints=['ref'] if i==20 else [],freshness_probe_neutral=True,sampling_probe_seconds=.01) for i in range(11,23)])
    reports={a:copy.deepcopy(base) for a in 'ABCD'}
    for a,report in reports.items():
        kind=dict(A='uniform',B='fitness',C='td',D='utility')[a]
        report['budget'].update(arm=kind,score_kind=kind)
        for row in report['rows']:row.update(actor_optimizer_steps=16,critic_optimizer_steps=32,baseline_draw_sha256='a'*64,fresh_fraction=.1,expired_fraction=0.,unscored_fraction=.9)
    reports['A']['rows']=[dict(r,changed_ea=0,probability_l1=0.,eligible_baseline_mass=0.) for r in reports['A']['rows']]
    add_provenance(reports)
    assert validate_reports(reports)['status']=='PASS'
    reports['C']['budget'][field]=value
    with pytest.raises(ValueError):validate_reports(reports)


@pytest.mark.parametrize('failure',['missing_field','rl_change','nonfinite','no_weighting','missing_reference','optimizer','prng_transition'])
def test_smoke_sampling_counter_failures_are_rejected(failure):
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke import validate_reports
    budget=dict(configured_total_episodes=716,environment_episodes=396,environment_timesteps=396000,rl_updates=192,batch_size=256,ea_ask=12,ea_tell=12,evaluation_checkpoints=[20,22],evaluation_episodes=4)
    reports={}
    for a in 'ABCD':
        rows=[dict(generation=g,observed_batches=32,ea_slots=8000,rl_slots=192,changed_ea=0 if a=='A' else 100,rl_changed=0,origin_changed=0,finite=True,probability_l1=0 if a=='A' else .01,eligible_baseline_mass=0 if a=='A' else .1,ess_fraction=.99,reference_fingerprints=['ref'] if g==20 else [],freshness_probe_neutral=True,sampling_probe_seconds=.01,actor_optimizer_steps=16,critic_optimizer_steps=32,baseline_draw_sha256='a'*64,fresh_fraction=.1,expired_fraction=0.,unscored_fraction=.9) for g in range(11,23)]
        kind=dict(A='uniform',B='fitness',C='td',D='utility')[a]
        reports[a]=dict(complete=True,budget=dict(budget,arm=kind,score_kind=kind),initial=dict(episodes=320,timesteps=320000,generation=10),final=dict(episodes=716,timesteps=716000,generation=22),rows=rows)
    add_provenance(reports)
    assert validate_reports(reports)['status']=='PASS'
    rows=reports['D']['rows']
    if failure=='missing_field':del rows[0]['ea_slots']
    elif failure=='rl_change':rows[0]['rl_changed']=1
    elif failure=='nonfinite':rows[0]['ess_fraction']=float('nan')
    elif failure=='no_weighting':
        for row in rows:row['probability_l1']=0
    elif failure=='missing_reference':
        for row in rows:row['reference_fingerprints']=[]
    elif failure=='optimizer':rows[0]['critic_optimizer_steps']=1
    elif failure=='prng_transition':rows[0]['baseline_prng_transition_verified']=False
    with pytest.raises(ValueError):validate_reports(reports)
    reports.pop('C')
    with pytest.raises(ValueError):validate_reports(reports)
