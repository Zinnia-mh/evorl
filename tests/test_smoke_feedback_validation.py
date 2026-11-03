import copy
import pytest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke import validate_reports

def reports():
    budget=dict(configured_total_episodes=716,environment_episodes=396,environment_timesteps=396000,rl_updates=192,batch_size=256,ea_ask=12,ea_tell=12,evaluation_checkpoints=[20,22],evaluation_episodes=4)
    result={}
    for a in 'ABCD':
        rows=[dict(generation=g,observed_batches=32,ea_slots=8000,rl_slots=192,changed_ea=0 if a=='A' else 100,rl_changed=0,origin_changed=0,finite=True,probability_l1=0 if a=='A' else .01,eligible_baseline_mass=0 if a=='A' else .1,ess_fraction=.99,reference_fingerprints=['ref'] if g==20 else [],freshness_probe_neutral=True,sampling_probe_seconds=.01,actor_optimizer_steps=16,critic_optimizer_steps=32,baseline_draw_sha256='a'*64,fresh_fraction=.1,expired_fraction=0.,unscored_fraction=.9,baseline_prng_transition_verified=True,intervention_terminal_key_verified=True,sampling_observation='deterministic_reconstruction') for g in range(11,23)]
        kind=dict(A='uniform',B='fitness',C='td',D='utility')[a]
        for row in rows:row.update(valid_score_source_count=0 if a=='A' else 4,distribution_prepared=a!='A',eligible_source_count=0 if a=='A' else 4,effective_eligible_mass=0 if a=='A' else .05)
        result[a]=dict(complete=True,seed=1,initialization_sha256='a'*64,initialization_settings_sha256='f'*64,fairness_policy='matched-settings-and-budget-v1',budget=dict(budget,arm=kind,score_kind=kind),initial=dict(episodes=320,timesteps=320000,generation=10),final=dict(episodes=716,timesteps=716000,generation=22),rows=rows)
    return result

def test_legal_learning_feedback_may_change_draw_hash_and_origin_counts():
    data=reports()
    for row in data['D']['rows'][1:]:row.update(baseline_draw_sha256='b'*64,ea_slots=7999,rl_slots=193)
    assert validate_reports(data)['status']=='PASS'

@pytest.mark.parametrize('failure',['initialization','seed','rl_slot','baseline_prng','intervention_key','sampling_count'])
def test_within_run_invariants_and_common_initialization_remain_strict(failure):
    data=reports();assert validate_reports(data)['status']=='PASS'
    if failure=='initialization':data['D']['initialization_settings_sha256']='b'*64
    elif failure=='seed':data['D']['seed']=2
    elif failure=='rl_slot':data['D']['rows'][0]['rl_changed']=1
    elif failure=='baseline_prng':data['D']['rows'][0]['baseline_prng_transition_verified']=False
    elif failure=='intervention_key':data['D']['rows'][0]['intervention_terminal_key_verified']=False
    elif failure=='sampling_count':data['D']['rows'][0]['ea_slots']-=1
    with pytest.raises(ValueError):validate_reports(data)
