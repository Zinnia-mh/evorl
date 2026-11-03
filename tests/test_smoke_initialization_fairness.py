import runpy
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3 import utility_replay_smoke as smoke
reports=runpy.run_path(str(Path(__file__).with_name('test_smoke_feedback_validation.py')))['reports']
FROZEN=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729')

def test_independent_postwarmup_state_hashes_are_traceability_not_equality_gate():
    data=reports()
    for i,report in enumerate(data.values()):
        report['initialization_sha256']=str(i)*64
        report['initialization_settings_sha256']='f'*64
        report['fairness_policy']='matched-settings-and-budget-v1'
    assert smoke.validate_reports(data)['status']=='PASS'

@pytest.mark.parametrize('field',['seed','network','environment','mechanism','budget'])
def test_actual_config_fingerprint_covers_shared_initialization_and_training(field):
    c=OmegaConf.load(FROZEN/'arm_A.yaml');changed=OmegaConf.create(OmegaConf.to_container(c,resolve=True))
    if field=='seed':changed.seed+=1
    elif field=='network':changed.agent_network.actor_hidden_layer_sizes=[32,32]
    elif field=='environment':changed.env.max_episode_steps-=1
    elif field=='mechanism':changed.enable_crossover=not changed.enable_crossover
    else:changed.batch_size+=1
    assert smoke.initialization_settings_fingerprint(c)!=smoke.initialization_settings_fingerprint(changed)

def test_only_preregistered_arm_fields_are_excluded_from_shared_fingerprint():
    values=[OmegaConf.load(FROZEN/f'arm_{a}.yaml') for a in 'ABCD']
    assert len({smoke.initialization_settings_fingerprint(c) for c in values})==1

def test_prepare_freezes_fairness_policy_and_actual_shared_fingerprint(tmp_path):
    m=smoke.prepare_smoke(FROZEN,tmp_path/'smoke',seed=1,commit='a'*40)
    assert m['fairness_policy']=='matched-settings-and-budget-v1'
    for name in m['configs'].values():assert smoke.initialization_settings_fingerprint(OmegaConf.load(tmp_path/'smoke'/name))==m['initialization_settings_sha256']

@pytest.mark.parametrize('fault',['settings','missing_settings','malformed_state_hash','prng','utility_valid','utility_prepared','utility_replace','utility_eligible'])
def test_new_fairness_policy_does_not_weaken_core_checks(fault):
    data=reports()
    for report in data.values():report.update(initialization_settings_sha256='f'*64,fairness_policy='matched-settings-and-budget-v1')
    row=next(r for r in data['D']['rows'] if r['generation']==20)
    row.update(valid_score_source_count=4,distribution_prepared=True,eligible_source_count=4,effective_eligible_mass=.05)
    if fault=='settings':data['D']['initialization_settings_sha256']='e'*64
    elif fault=='missing_settings':del data['D']['initialization_settings_sha256']
    elif fault=='malformed_state_hash':data['D']['initialization_sha256']='invalid'
    elif fault=='prng':row['baseline_prng_transition_verified']=False
    elif fault=='utility_valid':row['valid_score_source_count']=0
    elif fault=='utility_prepared':row['distribution_prepared']=False
    elif fault=='utility_replace':row['changed_ea']=0
    else:row['eligible_source_count']=0
    with pytest.raises(ValueError):smoke.validate_reports(data)

def test_single_arm_check_cannot_claim_four_arm_acceptance():
    data=reports()
    for report in data.values():report.update(initialization_settings_sha256='f'*64,fairness_policy='matched-settings-and-budget-v1')
    result=smoke.validate_reports({'A':data['A']},require_all=False)
    assert result['status']=='PASS_SINGLE_ARM' and result['arms']==['A']
    with pytest.raises(ValueError):smoke.validate_reports({'A':data['A']})

def test_saved_actual_configs_are_bound_to_manifest_even_if_output_checksum_is_resealed(tmp_path):
    directory=tmp_path/'smoke';m=smoke.prepare_smoke(FROZEN,directory,seed=1,commit='a'*40)
    data=reports()
    for i,(arm,report) in enumerate(data.items()):
        out=directory/arm;out.mkdir()
        c=OmegaConf.load(directory/m['configs'][arm]);OmegaConf.save(c,out/'resolved_config.yaml')
        report.update(initialization_sha256=str(i)*64,initialization_settings_sha256=m['initialization_settings_sha256'],fairness_policy=m['fairness_policy'])
        smoke.write(out/'report.json',report);smoke.checksums(out,'checksums.sha256')
    assert smoke.validate_smoke_outputs(directory)['status']=='PASS'
    c=OmegaConf.load(directory/'D'/'resolved_config.yaml');c.agent_network.actor_hidden_layer_sizes=[16,16]
    OmegaConf.save(c,directory/'D'/'resolved_config.yaml')
    # Re-signing a run's own output cannot rewrite the preregistered config.
    checksum=directory/'D'/'checksums.sha256';checksum.unlink();smoke.checksums(directory/'D','checksums.sha256')
    with pytest.raises(ValueError,match='actual run config'):smoke.validate_smoke_outputs(directory)
