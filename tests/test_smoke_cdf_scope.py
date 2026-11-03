import numpy as np
import pytest
from evorl.algorithms.erl.erl_td3.replay_distribution_audit import audit_distribution

def inputs():
    return [np.array([.5,.5],np.float32),np.array([1000,1000]),np.array([.5,.5,0.,0.],np.float32),np.array([.5,.9999999403953552,.9999998807907104,1.],np.float32),np.array([True,True,False,False]),1.000000238418579]

def test_unused_uniform_cdf_roundoff_is_recorded_without_mutating_any_array():
    args=inputs();before=[x.copy() if isinstance(x,np.ndarray) else x for x in args]
    result=audit_distribution(*args,cdf_in_training_sampling=False)
    assert result['cdf_in_training_sampling'] is False
    assert result['cdf_monotonic_raw'] is False
    assert result['cdf_valid_for_sampling'] is True
    assert result['cdf_negative_delta_count']==1
    assert result['cdf_min_delta_raw']==-5.960464477539063e-8
    assert result['ess_fraction_audit64']==1.
    for a,b in zip(args,before):np.testing.assert_array_equal(a,b)

def test_unused_cdf_with_a_real_shape_error_is_diagnostic_only_and_explicitly_flagged():
    args=inputs();args[3]=np.array([.5,.9,.7,1.],np.float32)
    result=audit_distribution(*args,cdf_in_training_sampling=False)
    assert not result['cdf_valid_for_sampling'] and not result['cdf_in_training_sampling']
    assert result['cdf_max_abs_reference_error']>result['cdf_accumulation_error_bound']

def test_weighted_cdf_roundoff_must_match_monotone_reference_within_dtype_error_bound():
    result=audit_distribution(*inputs(),cdf_in_training_sampling=True)
    assert result['cdf_valid_for_sampling']
    assert not result['cdf_monotonic_raw'] and result['cdf_monotonic_within_bound']
    assert result['cdf_max_abs_reference_error']<=result['cdf_accumulation_error_bound']

@pytest.mark.parametrize('case',['material_drop','wrong_monotone_cdf','float64_drop','nonfinite'])
def test_active_cdf_still_rejects_real_errors(case):
    args=inputs()
    if case=='material_drop':args[3]=np.array([.5,.9,.7,1.],np.float32)
    elif case=='wrong_monotone_cdf':args[3]=np.array([.9,1.,1.,1.],np.float32)
    elif case=='float64_drop':args[3]=np.array([.5,1.-1e-8,1.-2e-8,1.],np.float64)
    else:args[3][1]=np.nan
    with pytest.raises(ValueError):audit_distribution(*args,cdf_in_training_sampling=True)

@pytest.mark.parametrize('bad',['negative_probability','nan_probability','mass','nonresident'])
def test_auxiliary_cdf_scope_does_not_waive_probability_validity(bad):
    args=inputs()
    if bad=='negative_probability':args[0][0]=-.1
    elif bad=='nan_probability':args[0][0]=np.nan
    elif bad=='mass':args[0][0]=.9
    else:args[1][0]=0
    with pytest.raises(ValueError):audit_distribution(*args,cdf_in_training_sampling=False)

def test_actual_monotonic_weighted_cdf_retains_strict_finite_boundary_and_scope_checks():
    args=inputs();args[3]=np.array([.5,1.,1.,1.],np.float32)
    result=audit_distribution(*args,cdf_in_training_sampling=True)
    assert result['cdf_valid_for_sampling'] and result['cdf_in_training_sampling']
    args[3][-1]=.9
    with pytest.raises(ValueError):audit_distribution(*args,cdf_in_training_sampling=True)

@pytest.mark.parametrize('arm',['A','D'])
def test_signed_raw_interval_roundoff_is_not_misidentified_as_sampling_probability(arm):
    import runpy
    from pathlib import Path
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke import validate_reports
    data=runpy.run_path(str(Path(__file__).with_name('test_smoke_feedback_validation.py')))['reports']()
    row=next(r for r in data[arm]['rows'] if r['generation']==20)
    row.update(audit_distribution(*inputs(),cdf_in_training_sampling=arm!='A'))
    row['ess_fraction']=row['ess_fraction_audit64']
    row['cdf_non_ea_interval_mass64']=-1e-8
    assert validate_reports(data)['status']=='PASS'
    if arm=='D':
        row['cdf_non_ea_interval_mass64']=-.1
        with pytest.raises(ValueError):validate_reports(data)
        row['cdf_non_ea_interval_mass64']=0.
        row['cdf_in_training_sampling']=False
        with pytest.raises(ValueError):validate_reports(data)
