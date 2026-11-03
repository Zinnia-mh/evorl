from decimal import Decimal,localcontext
import numpy as np
import pytest
from evorl.algorithms.erl.erl_td3.replay_distribution_audit import audit_distribution

def audit(p,counts=None,cdf=None,transition=None,ea=None,raw=1.):
    p=np.asarray(p,np.float32)
    if counts is None:counts=np.ones(len(p),np.int32)
    if transition is None:transition=p
    if cdf is None:cdf=np.cumsum(transition,dtype=np.float32);cdf[-1]=max(cdf[-1],1.) if np.sum(counts)>0 else 0.
    if ea is None:ea=np.ones(len(transition),bool)
    return audit_distribution(p,counts,np.asarray(transition),np.asarray(cdf),np.asarray(ea),raw)

@pytest.mark.parametrize('n',[3,7,11,32,352,416,448,672,704])
def test_uniform_ess_is_mathematically_bounded_without_clipping(n):
    p=np.full(n,np.float32(1/n));before=p.copy()
    r=audit(p,raw=1.0000003576278687)
    assert r['ess_fraction_audit64']==1. and r['ess_raw_float32']==1.0000003576278687
    assert r['source_count']==n and r['effective_source_count_audit64']==n
    np.testing.assert_array_equal(p,before)

@pytest.mark.parametrize('weights',[[.1,.2,.7],[1.,0.,0.],[1e-30,1e-30,1.],[.25,.25,.5]])
def test_nonuniform_and_extreme_match_decimal_reference(weights):
    p=np.asarray(weights,np.float32);r=audit(p)
    with localcontext() as ctx:
        ctx.prec=80;v=[Decimal.from_float(float(x)) for x in p]
        reference=float(sum(v)**2/(Decimal(len(v))*sum(x*x for x in v)))
    assert 0<=r['ess_fraction_audit64']<=1
    assert r['ess_fraction_audit64']==pytest.approx(reference,abs=1e-15,rel=1e-15)

def test_empty_sources_are_zero_and_nonresident_slots_do_not_enter_denominator():
    assert audit([0.,0.],counts=np.zeros(2,np.int32),ea=np.zeros(2,bool))['ess_fraction_audit64']==0
    r=audit([.5,.5,0.],counts=[4,4,0],transition=[.5,.5,0.],ea=[True,True,False])
    assert r['source_count']==2 and r['ess_fraction_audit64']==1

@pytest.mark.parametrize('case',['negative','nan','mass','nonresident','cdf_decrease','cdf_boundary'])
def test_illegal_probabilities_and_cdf_are_rejected(case):
    p=np.array([.25,.75],np.float32);counts=np.array([4,4]);cdf=np.array([.25,1.],np.float32)
    if case=='negative':p[0]=-.1
    elif case=='nan':p[0]=np.nan
    elif case=='mass':p[0]=.5
    elif case=='nonresident':counts[0]=0
    elif case=='cdf_decrease':cdf=np.array([1.,.75])
    elif case=='cdf_boundary':cdf[-1]=.9
    with pytest.raises(ValueError):audit(p,counts=counts,cdf=cdf)

def test_cdf_interval_evidence_does_not_mutate_sampling_inputs():
    p=np.array([.1,.2,.7],np.float32);cdf=np.cumsum(p);before=cdf.copy()
    r=audit(p,cdf=cdf)
    assert r['cdf_sampling_mass64']==1 and r['cdf_probability_max_abs_error']<1e-6
    np.testing.assert_array_equal(cdf,before)
