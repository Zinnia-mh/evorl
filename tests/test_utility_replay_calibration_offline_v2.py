import copy
import json
from pathlib import Path
import jax
import numpy as np
import pytest

from scripts.calibrate_utility_replay import registration
from evorl.algorithms.erl.erl_td3 import utility_replay_calibration_pipeline as old
from evorl.algorithms.erl.erl_td3.source_replay import sample_weighted_ea_indices
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_offline_v2 import (
    SnapshotSimulator, CandidateStore, simulate_snapshot)


def snapshot():
    tokens=np.repeat(np.arange(4,8,dtype=np.int32),64)
    arrays=dict(origin_type=np.array([0]*256+[1,1],np.int8),source_token=np.r_[tokens,[-1,-1]].astype(np.int32),
        source_generation=np.array([1]*256+[-1,-1],np.int32),score_version=np.full(258,-1,np.int32))
    history=[dict(source_token=i,measured_generation=1,observer_score_version=0,valid=True,
        fitness=float(i),td=float(i),utility=float(i),ranks={k:(i-4)/3 for k in ['fitness','td','utility']}) for i in range(4,8)]
    meta=dict(generation=2,capacity=258,buffer_size=258,current_index=0,score_history=history,
        population_size=4,scored_generations=[1],timing={'diagnostic_ms':1.,'sampling_ms':.1},memory_mib=.001,
        boundary=old.PROTOCOL['boundary'],resident_counts={str(i):64 for i in range(4,8)})
    return meta,arrays


def compare_implementations(meta,arrays,reg,monkeypatch,draws=64):
    reference_distributions=[];reference_samples=[]
    original_build=old.build_weighted_source_distribution
    original_sample=old.sample_weighted_ea_indices
    def build(*args,**kwargs):
        result=original_build(*args,**kwargs)
        reference_distributions.append(jax.device_get(result))
        return result
    def sample(*args,**kwargs):
        result=original_sample(*args,**kwargs)
        reference_samples.append(np.array(result.sampled_indices))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(old,'build_weighted_source_distribution',build)
        patch.setattr(old,'sample_weighted_ea_indices',sample)
        before_rows,before_details=old.simulate(meta,arrays,reg,draws=draws)
    plan=SnapshotSimulator(meta,arrays)
    index=0
    for candidate in reg['candidates']:
        for kind,kind_id in [('fitness',1),('td',2),('utility',3)]:
            _,distribution=plan.distribution(candidate,kind)
            expected=reference_distributions[index]
            for field in ['source_probability','transition_probability','transition_cdf','ess_fraction']:
                assert np.all(np.isfinite(np.asarray(getattr(distribution,field))))
                assert np.all(np.isfinite(getattr(expected,field)))
                np.testing.assert_allclose(np.asarray(getattr(distribution,field)),getattr(expected,field),atol=1e-6,rtol=1e-5)
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(104729),old.SIMULATION_TAG),meta['generation']*3+kind_id)
            for repeat in range(2):
                actual=sample_weighted_ea_indices(distribution,key,draws)
                np.testing.assert_array_equal(actual.sampled_indices,reference_samples[index*2+repeat])
            index+=1
    after_rows,after_details=simulate_snapshot(meta,arrays,reg,draws=draws)
    assert all(r['numerical_ok'] for r in before_rows+after_rows)
    for before,after in zip(before_rows,after_rows):
        for key in before.keys()-{'memory_mib','overhead_ms'}:
            if isinstance(before[key],float): assert after[key]==pytest.approx(before[key],abs=1e-6,rel=1e-5)
            else: assert after[key]==before[key]
    for before,after in zip(before_details,after_details):
        np.testing.assert_array_equal(before['eligible'],after['eligible'])
        assert before['repeat_rate']==after['repeat_rate']


def test_all_36_candidates_preserve_probabilities_samples_and_metrics(monkeypatch):
    meta,arrays=snapshot();before=copy.deepcopy(arrays)
    compare_implementations(meta,arrays,registration(),monkeypatch)
    for key in arrays: np.testing.assert_array_equal(arrays[key],before[key])


@pytest.mark.parametrize('generation',[20,70])
def test_one_real_million_capacity_snapshot_preserves_samples(monkeypatch,generation):
    root=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-option-b-v1-seed104729-head8240863/snapshots')
    meta=json.loads((root/f'generation_{generation:06d}.json').read_text())
    with np.load(root/f'generation_{generation:06d}.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    reg=registration();reg['candidates']=[reg['candidates'][0 if generation==20 else 5]]
    # Production v1 intentionally uses NaN for unmeasured, validity=False
    # scores. The debug hook rejects that sentinel before the mask is applied.
    # Match production for this test; finite probabilities, correctness flags,
    # exact sample indices and the original tolerances are asserted above.
    with jax.debug_nans(False):
        compare_implementations(meta,arrays,reg,monkeypatch)


def test_expired_invalid_and_under_resident_sources_match_reference(monkeypatch):
    meta,arrays=snapshot();counts=[2,16,64,64]
    arrays=dict(origin_type=np.array([0]*sum(counts)+[1,1],np.int8),
        source_token=np.r_[np.repeat(np.arange(4,8),counts),[-1,-1]].astype(np.int32),
        source_generation=np.array([1]*sum(counts)+[-1,-1],np.int32),
        score_version=np.full(sum(counts)+2,-1,np.int32))
    meta.update(generation=15,capacity=len(arrays['source_token']),buffer_size=len(arrays['source_token']),
        resident_counts={str(i):n for i,n in zip(range(4,8),counts)})
    for i,record in enumerate(meta['score_history']):
        record['valid']=i<3;record['ranks']={kind:(i/2 if i<3 else 0.) for kind in ['fitness','td','utility']}
    compare_implementations(meta,arrays,registration(),monkeypatch)


def test_per_candidate_checkpoint_resume_never_remeasures(tmp_path,monkeypatch):
    meta,arrays=snapshot();reg=registration();events=[]
    store=CandidateStore(tmp_path/'work',{'implementation':'test','source_sha256':'a'*64})
    first=simulate_snapshot(meta,arrays,reg,draws=64,checkpoint=store,progress=events.append)
    assert len(events)==36 and len(list((tmp_path/'work').glob('generation_*/candidate_*.json')))==36
    def forbidden(*args,**kwargs): pytest.fail('completed measurements must not be repeated')
    monkeypatch.setattr(SnapshotSimulator,'measure',forbidden)
    assert simulate_snapshot(meta,arrays,reg,draws=64,checkpoint=store)==first
    with pytest.raises(ValueError): CandidateStore(tmp_path/'work',{'implementation':'changed'})


def test_cached_inputs_are_reused_and_corrupt_checkpoint_rejected(tmp_path):
    meta,arrays=snapshot();plan=SnapshotSimulator(meta,arrays);c=registration()['candidates'][0]
    slots=plan.device_slots;mask=plan.device_ea
    plan.distribution(c,'utility');plan.distribution(dict(c,alpha=.5),'utility')
    assert plan.device_slots is slots and plan.device_ea is mask
    store=CandidateStore(tmp_path/'work',{'implementation':'test'})
    store.save(2,0,{'candidate_id':0},[])
    path=tmp_path/'work/generation_000002/candidate_000.json'
    data=json.loads(path.read_text());data['row']['candidate_id']=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError): store.load(2,0)


def test_heavy_jit_cache_is_reused_across_source_count_changes():
    from evorl.algorithms.erl.erl_td3.utility_replay_calibration_offline_v2 import (
        expand_transition_probabilities,transition_cdf,device_checks)
    meta,arrays=snapshot();candidate=registration()['candidates'][0]
    first=SnapshotSimulator(meta,arrays);first.prepare_candidates([candidate]);first.measure(0,candidate,64)
    before=[f._cache_size() for f in [expand_transition_probabilities,transition_cdf,device_checks]]
    meta2=copy.deepcopy(meta);arrays2=copy.deepcopy(arrays)
    mask=arrays2['source_token']==7
    arrays2['origin_type'][mask]=1;arrays2['source_token'][mask]=-1;arrays2['source_generation'][mask]=-1
    meta2['resident_counts'].pop('7');meta2['score_history']=[r for r in meta2['score_history'] if r['source_token']!=7]
    second=SnapshotSimulator(meta2,arrays2);second.prepare_candidates([candidate]);second.measure(0,candidate,64)
    assert len(first.tokens)==4 and len(second.tokens)==3 and first.source_block==second.source_block==4
    assert [f._cache_size() for f in [expand_transition_probabilities,transition_cdf,device_checks]]==before
