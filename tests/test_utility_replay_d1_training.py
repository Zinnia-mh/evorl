import json,hashlib
from pathlib import Path
import pytest,jax,numpy as np
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3 import utility_replay_training as training
from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

def baseline(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('resume_fixture',Path(__file__).with_name('test_utility_replay_resume.py'));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);configs=fixture.configs
    cs=configs(tmp_path)
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
    for c in cs.values():c.seed=2027;c.arm_manifest_path=str(tmp_path/'baseline_arms.json')
    prepare_arm_manifest(list(cs.values()),tmp_path/'baseline_arms.json')
    c=cs['utility']
    p=tmp_path/'baseline.yaml';OmegaConf.save(c,p,resolve=True);return p

def test_explicit_d1_and_unchanged_d0_component_selection(tmp_path):
    factory=getattr(training,'_runner_components',None)
    assert factory is not None,'persistent runner lacks explicit D1 component selection'
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol,D1TrainingWorkflow,validate_d1_manifest
    from evorl.algorithms.erl.erl_td3.utility_replay_d1 import D1CheckpointStore
    p=baseline(tmp_path);c=OmegaConf.load(p);a,b,v=factory(c)
    assert a is ERLUtilityReplayWorkflow and b is PilotCheckpointStore
    prepare_d1_protocol(p,tmp_path/'d1');d=OmegaConf.load(tmp_path/'d1/resolved_config.yaml');a,b,v=factory(d)
    assert a is D1TrainingWorkflow and b is D1CheckpointStore and v is validate_d1_manifest

def test_d1_protocol_seals_source_config_and_baseline_without_mutation(tmp_path):
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol,validate_d1_manifest
    p=baseline(tmp_path);before=p.read_bytes();prepare_d1_protocol(p,tmp_path/'d1');c=OmegaConf.load(tmp_path/'d1/resolved_config.yaml');validate_d1_manifest(c)
    assert p.read_bytes()==before and c.rho==.3 and c.diagnostic_interval==1
    with pytest.raises(FileExistsError):prepare_d1_protocol(p,tmp_path/'d1')
    c.rho=.4
    with pytest.raises(ValueError):validate_d1_manifest(c)

@pytest.mark.parametrize('field,value',[('seed',1),('half_life',14.),('env.max_episode_steps',5),('total_episodes',71)])
def test_d1_manifest_rejects_unfrozen_config(tmp_path,field,value):
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol,validate_d1_manifest
    p=baseline(tmp_path);prepare_d1_protocol(p,tmp_path/'d1');c=OmegaConf.load(tmp_path/'d1/resolved_config.yaml');OmegaConf.update(c,field,value)
    with pytest.raises(ValueError):validate_d1_manifest(c)

def test_d1_durable_stop_resume_skips_warmup_preserves_evaluation_and_state(tmp_path,monkeypatch):
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol
    p=baseline(tmp_path);prepare_d1_protocol(p,tmp_path/'d1');c=OmegaConf.load(tmp_path/'d1/resolved_config.yaml')
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import D1TrainingWorkflow
    manifest_path=tmp_path/'d1/protocol.json';m=json.loads(manifest_path.read_text());m['index_audit_generations']=[13,16];manifest_path.write_text(json.dumps(m))
    seal=tmp_path/'d1/checksums.sha256';seal.write_text(''.join(hashlib.sha256((tmp_path/'d1'/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['protocol.json','resolved_config.yaml']))
    observe=D1TrainingWorkflow.observe_training_step
    def request_at13(self,before,metrics,state,segment,elapsed):
        observe(self,before,metrics,state,segment,elapsed)
        if int(state.metrics.iterations)==13:
            manifest=json.loads((tmp_path/'run/manifest.json').read_text())
            (tmp_path/'run/stop_request.json').write_text(json.dumps(dict(run_id=manifest['run_id'])))
    monkeypatch.setattr(D1TrainingWorkflow,'observe_training_step',request_at13)
    paused,first=training.run_training(c,tmp_path/'run',checkpoint_interval=50)
    assert first['status']=='PAUSED' and first['final']['generation']==13
    assert not (tmp_path/'run/completion.json').exists()
    original=initialization_fingerprint(paused)
    def forbidden(*args,**kw):raise AssertionError('resume cannot setup/warmup')
    monkeypatch.setattr(ERLUtilityReplayWorkflow,'setup',forbidden);monkeypatch.setattr(ERLUtilityReplayWorkflow,'warmup_step',forbidden)
    state,last=training.run_training(c,tmp_path/'run',resume_checkpoint=first['last_checkpoint'],checkpoint_interval=50)
    assert Path(last['last_checkpoint']).parent==Path(first['last_checkpoint']).parent
    assert not (tmp_path/'run/stop_request.json').exists()
    assert last['restored_state_sha256']==original and last['status']=='COMPLETE'
    assert last['final']==dict(generation=16,episodes=70,timesteps=280,actor_optimizer_count=12,critic_optimizer_count=24)
    assert [r['generation'] for r in last['evaluations']]==[12,14,16]
    assert last['budget']['ea_ask']==last['budget']['ea_tell']==6
    rows=[json.loads(x) for p in (tmp_path/'run/segments').glob('*/d1_metrics.jsonl') for x in p.read_text().splitlines()]
    assert sorted(r['generation'] for r in rows)==list(range(11,17))
    assert all(r['new_source_count']==4 and r['distribution_build_count']==1 for r in rows)
    assert all(r['positive_utility_fraction']+r['negative_utility_fraction']+r['zero_utility_fraction']==1 for r in rows)
    for row in [r for r in rows if r['generation'] in [13,16]]:
        assert row['index_reconstruction']['rl_changed']==row['index_reconstruction']['origin_changed']==0
        assert row['index_reconstruction']['changed_ea']>0
        assert all(x['measured_generation']==row['generation'] for x in row['sources'])
    with pytest.raises(ValueError,match='complete'):training.run_training(c,tmp_path/'run',resume_checkpoint=last['last_checkpoint'])


def test_d1_rejects_checkpoint_policy_override_before_creating_run(tmp_path):
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol
    p=baseline(tmp_path);prepare_d1_protocol(p,tmp_path/'d1');c=OmegaConf.load(tmp_path/'d1/resolved_config.yaml')
    with pytest.raises(ValueError,match='checkpoint policy'):training.run_training(c,tmp_path/'run',checkpoint_interval=2)
    assert not (tmp_path/'run').exists()


def test_d1_stop_request_is_run_bound_and_acknowledged(tmp_path):
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import D1TrainingWorkflow
    poll=getattr(D1TrainingWorkflow,'stop_requested',None)
    assert poll is not None,'D1 lacks graceful Host-boundary stop request'
    assert not poll(None,tmp_path,'run1')
    p=tmp_path/'stop_request.json';p.write_text(json.dumps(dict(run_id='wrong')))
    with pytest.raises(ValueError):poll(None,tmp_path,'run1')
    p.write_text(json.dumps(dict(run_id='run1')))
    assert poll(None,tmp_path,'run1')
    segment=tmp_path/'segment';segment.mkdir();D1TrainingWorkflow.acknowledge_stop(None,tmp_path,segment)
    assert not p.exists() and (segment/'accepted_stop_request.json').exists()
