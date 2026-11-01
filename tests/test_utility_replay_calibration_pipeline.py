import copy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.calibrate_utility_replay import registration, select, freeze
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_pipeline import simulate, aggregate, CalibrationObserver


def snapshot():
    arrays=dict(origin_type=np.array([0]*256+[1,1],np.int8),source_token=np.array(list(np.repeat(np.arange(4,8),64))+[-1,-1]),
                source_generation=np.array([1]*256+[-1,-1]),score_version=np.array([-1]*258))
    history=[dict(source_token=i,measured_generation=1,observer_score_version=0,valid=True,
                  fitness=float(i),td=float(i),utility=float(i),ranks=dict(fitness=float(i-4)/3,td=float(i-4)/3,utility=float(i-4)/3)) for i in (4,5,6,7)]
    meta=dict(generation=2,capacity=258,buffer_size=258,current_index=0,score_history=history,population_size=4,scored_generations=[1],
              timing={'diagnostic_ms':1.,'sampling_ms':.1},memory_mib=.001,
              boundary='post_step_pre_next_rollout',resident_counts={str(i):64 for i in range(4,8)})
    return meta,arrays


def test_snapshot_to_36_candidates_to_deterministic_selection_and_freeze(tmp_path):
    meta,arrays=snapshot(); before=copy.deepcopy(arrays); reg=registration()
    rows,details=simulate(meta,arrays,reg,draws=64)
    assert len(rows)==36 and len(details)==108
    repeated,_=simulate(meta,arrays,reg,draws=64)
    # Timing is measured anew; selection reproducibility applies to identical frozen metrics.
    for a,b in zip(rows,repeated):
        for k in a.keys()-{'overhead_ms','memory_mib'}: assert a[k]==b[k]
    combined=aggregate([rows],reg)
    result=select(reg,combined)
    p={'source':'synthetic-test','input_sha256':'f'*64}
    assert freeze(reg,combined,tmp_path/'a',p)==freeze(reg,combined,tmp_path/'b',p)
    assert result['selected_id'] in range(36)
    for k in arrays: np.testing.assert_array_equal(arrays[k],before[k])
    from hydra import compose,initialize
    from omegaconf import OmegaConf
    from scripts.collect_utility_replay_calibration import write_launch_configs
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import validate_arm_manifest
    with initialize(config_path='../configs',version_base=None):
        base=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer','seed=1'])
    resolved=freeze(reg,combined,tmp_path/'launch',p)
    write_launch_configs(base,resolved,tmp_path/'launch')
    for arm in 'ABCD': validate_arm_manifest(OmegaConf.load(tmp_path/'launch'/f'arm_{arm}.yaml'))


def test_future_score_and_corrupt_residency_are_rejected():
    meta,arrays=snapshot(); meta['score_history'][0]['measured_generation']=3
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)
    meta,arrays=snapshot(); meta['resident_counts']['4']=3
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)


def test_missing_snapshot_or_outcome_leakage_is_rejected():
    meta,arrays=snapshot(); meta['evaluation_return']=1
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)
    with pytest.raises(ValueError): aggregate([],registration())


def test_invalid_eligible_score_or_rank_is_rejected():
    meta,arrays=snapshot(); meta['score_history'][0]['utility']=float('nan')
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)
    meta,arrays=snapshot(); meta['score_history'].pop()
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)
    meta,arrays=snapshot(); meta['score_history'][0]['final_return']=2
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)
    meta,arrays=snapshot(); meta['score_history'][0]['ranks']['td']=2
    with pytest.raises(ValueError): simulate(meta,arrays,registration(),draws=8)


def test_prepare_freezes_disjoint_seed_config_and_protocol_without_running(tmp_path,monkeypatch):
    from hydra import compose,initialize
    from omegaconf import OmegaConf
    import json
    import scripts.collect_utility_replay_calibration as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'require_clean_tree',lambda:None)
    monkeypatch.setattr(runner,'git_head',lambda:'test-commit')
    with initialize(config_path='../configs',version_base=None):
        config=compose(config_name='config',overrides=[
            'agent=erl/erl-utility-replay','env=brax/swimmer','seed=104729',
            'diagnostic_interval=10','total_episodes=2300'])
    OmegaConf.save(config,tmp_path/'input.yaml',resolve=True)
    (tmp_path/'candidates.json').write_text(json.dumps(registration()))
    registered=runner.prepare(tmp_path/'input.yaml',tmp_path/'candidates.json',[1,2,3],tmp_path/'preflight')
    assert registered['calibration_seed']==104729 and registered['protocol_sha256']
    runner.verify(tmp_path/'preflight','preregistration.sha256')
    assert not (tmp_path/'preflight/snapshots').exists()
    with pytest.raises(ValueError):
        runner.prepare(tmp_path/'input.yaml',tmp_path/'candidates.json',[1,2,104729],tmp_path/'conflict')


def test_analysis_reuses_frozen_measurements_for_identical_selection(tmp_path,monkeypatch):
    import json,hashlib
    from hydra import compose,initialize
    from omegaconf import OmegaConf
    import scripts.collect_utility_replay_calibration as runner
    import scripts.calibrate_utility_replay as selector
    monkeypatch.setattr(selector,'ARTIFACT_ROOT',tmp_path)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'require_clean_tree',lambda:None)
    monkeypatch.setattr(runner,'git_head',lambda:'0'*40)
    with initialize(config_path='../configs',version_base=None):
        collection=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer',
            'seed=104729','diagnostic_interval=10','total_episodes=2300'])
        formal=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer','seed=1'])
    OmegaConf.save(collection,tmp_path/'input.yaml',resolve=True)
    OmegaConf.save(formal,tmp_path/'formal.yaml',resolve=True)
    reg=registration(); (tmp_path/'candidates.json').write_text(json.dumps(reg))
    directory=tmp_path/'corpus'; prereg=runner.prepare(tmp_path/'input.yaml',tmp_path/'candidates.json',[1,2,3],directory)
    (directory/'snapshots').mkdir()
    meta,arrays=snapshot()
    for generation in range(11,71):
        (directory/'snapshots'/f'generation_{generation:06d}.json').write_text(json.dumps(dict(meta,generation=generation)))
        np.savez_compressed(directory/'snapshots'/f'generation_{generation:06d}.npz',**arrays)
    runner.hashes(directory/'snapshots','snapshots.sha256')
    (directory/'collection_counters.json').write_text(json.dumps(dict(episodes=2300,generation=70)))
    rows,_=simulate(meta,arrays,reg,draws=64)
    metrics=directory/'offline_metrics'; metrics.mkdir()
    identity=dict(corpus_sha256=hashlib.sha256((directory/'snapshots/snapshots.sha256').read_bytes()).hexdigest(),
                  protocol_sha256=prereg['protocol_sha256'],commit=prereg['commit'])
    for name,value in [('identity',identity),('aggregated',rows),('measurements',[])]:
        (metrics/f'{name}.json').write_text(json.dumps(value))
    runner.hashes(metrics,'checksums.sha256')
    monkeypatch.setattr(runner,'simulate',lambda *a,**k:pytest.fail('timings must not be remeasured'))
    first=runner.analyze(directory,tmp_path/'selected1',tmp_path/'formal.yaml')
    second=runner.analyze(directory,tmp_path/'selected2',tmp_path/'formal.yaml')
    assert first==second


def test_real_jitted_training_state_and_samples_equal_with_observer_enabled():
    import chex
    import jax
    import jax.numpy as jnp
    from hydra import compose, initialize
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
    with initialize(config_path='../configs',version_base=None):
        config=compose(config_name='config',overrides=[
            'agent=erl/erl-utility-replay','env=brax/swimmer','env.max_episode_steps=1',
            'num_envs=1','pop_size=4','num_elites=1','num_rl_agents=1',
            'batch_size=4','replay_buffer_capacity=64','warmup_iters=0',
            'actor_update_interval=1','rl_updates_frac=0.25','source_batch_size=1',
            'diagnostic_interval=10','num_eval_envs=1','eval_episodes=1',
            'agent_network.critic_hidden_layer_sizes=[8,8]',
            'agent_network.actor_hidden_layer_sizes=[8,8]'])
    baseline=type('CalibrationOff',(ERLUtilityReplayWorkflow,),{}).build_from_config(config,enable_jit=True)
    observed=type('CalibrationOn',(ERLUtilityReplayWorkflow,),{}).build_from_config(config,enable_jit=True)
    key=jax.random.PRNGKey(81)
    _,base=baseline.step(baseline.setup(key))
    _,other=observed.step(observed.setup(key))
    base=base.replace(metrics=base.metrics.replace(iterations=jnp.uint32(9)))
    other=other.replace(metrics=other.metrics.replace(iterations=jnp.uint32(9)))
    observer=CalibrationObserver(); observed.attach_calibration_observer(observer)
    base_metrics,base=baseline.step(base); other_metrics,other=observed.step(other)
    chex.assert_trees_all_equal(base,other)
    chex.assert_trees_all_equal(base_metrics.num_updates_per_iter,other_metrics.num_updates_per_iter)
    sample_key=jax.random.PRNGKey(90)
    chex.assert_trees_all_equal(baseline.replay_buffer.sample(base.replay_buffer_state,sample_key),
                               observed.replay_buffer.sample(other.replay_buffer_state,sample_key))
    assert len(observer.snapshots)==1 and len(observer.records)==4
    assert all(r.eligible for r in observer.records)
    assert len(observer.snapshots[0][0]['score_history'])==4
