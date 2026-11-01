from types import SimpleNamespace
import numpy as np
import pytest
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_capture import capture, seed_preflight, write_snapshot


def replay():
    return SimpleNamespace(origin_type=np.array([0,0,1,0]),source_token=np.array([4,4,-1,5]),
                           source_generation=np.array([1,1,-1,1]),score_version=np.array([0,0,-1,0]),
                           buffer_size=4,current_index=0)


def test_capture_copies_real_ring_counts_without_mutating_state_or_key(tmp_path):
    state=replay(); before={k:v.copy() for k,v in vars(state).items() if isinstance(v,np.ndarray)}
    key=np.array([3,9]); original=key.copy()
    metadata,arrays=capture(state,[],2,population_size=4,timing={'diagnostic_ms':2.,'sampling_ms':1.},memory_mib=3.)
    assert metadata['resident_counts']=={'4':2,'5':1}
    assert metadata['current_index']==0
    arrays['source_token'][0]=99
    for k,v in before.items(): np.testing.assert_array_equal(getattr(state,k),v)
    np.testing.assert_array_equal(key,original)
    assert len(write_snapshot(tmp_path,metadata,arrays))==2
    with pytest.raises(FileExistsError): write_snapshot(tmp_path,metadata,arrays)


def test_missing_runtime_metrics_are_rejected():
    with pytest.raises(ValueError): capture(replay(),[],2,population_size=4,timing={},memory_mib=0)


def test_seed_registration_requires_actual_disjoint_formal_set():
    with pytest.raises(ValueError): seed_preflight([])
    with pytest.raises(ValueError): seed_preflight([1,2,104729])
    assert seed_preflight([1,2,3])['calibration_seed']==104729
