"""Host-only numerical evidence; never returns inputs to the sampler."""
import numpy as np


def audit_distribution(probability, counts, transition, cdf, ea, raw_ess):
    p=np.asarray(probability,dtype=np.float64)
    counts=np.asarray(counts)
    t=np.asarray(transition,dtype=np.float64)
    cdf=np.asarray(cdf,dtype=np.float64)
    ea=np.asarray(ea,dtype=bool)
    if p.shape!=counts.shape or t.shape!=cdf.shape or t.shape!=ea.shape:
        raise ValueError('distribution shape mismatch')
    if any(not np.all(np.isfinite(x)) for x in [p,t,cdf,counts]) or not np.isfinite(raw_ess):
        raise ValueError('nonfinite distribution')
    if np.any(p<0) or np.any(t<0) or np.any(counts<0) or np.any(p[counts==0]!=0):
        raise ValueError('illegal distribution support')
    n=int(np.count_nonzero(counts));mass=float(p.sum());tmass=float(t.sum())
    if n:
        if abs(mass-1)>1e-5 or abs(tmass-1)>1e-5:
            raise ValueError('probability mass mismatch')
        if cdf.size==0 or cdf[-1]<1 or np.any(np.diff(cdf)<0) or cdf[0]<0:
            raise ValueError('illegal CDF')
        v=p[counts>0];mean=float(v.mean());variance=float(np.mean((v-mean)**2))
        fraction=mean**2/(mean**2+variance)
    else:
        if mass!=0 or tmass!=0 or np.any(cdf!=0):raise ValueError('nonempty empty distribution')
        fraction=0.
    # Actual continuous-uniform CDF intervals, including terminal rounding
    # residual. Invalid-EA interval mass falls back in the existing overlay.
    boundaries=np.clip(cdf,0.,1.)
    intervals=np.diff(np.concatenate(([0.],boundaries)))
    return dict(ess_audit_version='normalized-variance-float64-v1',
        ess_raw_float32=float(raw_ess),ess_fraction_audit64=fraction,
        effective_source_count_audit64=n*fraction,source_count=n,
        source_probability_mass64=mass,source_probability_squared_mass64=float(np.sum(p*p)),
        transition_probability_mass64=tmass,cdf_sampling_mass64=float(intervals.sum()),
        cdf_probability_max_abs_error=float(np.max(np.abs(intervals-t))) if t.size else 0.,
        cdf_non_ea_interval_mass64=float(intervals[~ea].sum()))
