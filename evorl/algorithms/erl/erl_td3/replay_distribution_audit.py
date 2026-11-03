"""Host-only numerical evidence; never returns inputs to the sampler."""
import numpy as np


def audit_distribution(probability, counts, transition, cdf, ea, raw_ess,*,cdf_in_training_sampling=True):
    p=np.asarray(probability,dtype=np.float64)
    counts=np.asarray(counts)
    t=np.asarray(transition,dtype=np.float64)
    cdf_dtype=np.asarray(cdf).dtype
    cdf=np.asarray(cdf,dtype=np.float64)
    ea=np.asarray(ea,dtype=bool)
    if p.shape!=counts.shape or t.shape!=cdf.shape or t.shape!=ea.shape:
        raise ValueError('distribution shape mismatch')
    if any(not np.all(np.isfinite(x)) for x in [p,t,cdf,counts]) or not np.isfinite(raw_ess):
        raise ValueError('nonfinite distribution')
    if np.any(p<0) or np.any(t<0) or np.any(counts<0) or np.any(p[counts==0]!=0):
        raise ValueError('illegal distribution support')
    n=int(np.count_nonzero(counts));mass=float(p.sum());tmass=float(t.sum())
    delta=np.diff(cdf)
    cdf_monotonic=bool(np.all(delta>=0))
    reference=np.cumsum(t,dtype=np.float64)
    # Standard gamma_k forward-error bound, independent of summation order.
    # Never admit more than the existing 1e-5 probability-validity tolerance.
    # Include the sampler's existing final max(CDF[-1], 1) boundary correction.
    unit=np.finfo(cdf_dtype).eps/2 if np.issubdtype(cdf_dtype,np.floating) else 0.
    additions=max(0,t.size-1)
    gamma=(additions*unit)/(1-additions*unit) if additions*unit<1 else float('inf')
    bound=min(1e-5,gamma*float(np.sum(np.abs(t)))+(abs(1-tmass) if n else 0.))
    error=float(np.max(np.abs(cdf-reference))) if t.size else 0.
    monotonic_within_bound=bool(np.all(delta>=-2*bound))
    cdf_valid=bool(cdf.size and np.all(cdf>=0) and cdf[-1]>=1 and monotonic_within_bound and error<=bound) if n else bool(np.all(cdf==0))
    if n:
        if abs(mass-1)>1e-5 or abs(tmass-1)>1e-5:
            raise ValueError('probability mass mismatch')
        if cdf_in_training_sampling and not cdf_valid:
            raise ValueError(f'illegal sampling CDF: first={cdf[0] if cdf.size else None}, last={cdf[-1] if cdf.size else None}, monotonic={cdf_monotonic}, min_delta={float(delta.min()) if delta.size else 0.}, reference_error={error}, bound={bound}')
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
        cdf_non_ea_interval_mass64=float(intervals[~ea].sum()),
        cdf_in_training_sampling=bool(cdf_in_training_sampling),cdf_valid_for_sampling=cdf_valid,
        cdf_finite_raw=True,cdf_monotonic_raw=cdf_monotonic,
        cdf_monotonic_within_bound=monotonic_within_bound,cdf_accumulation_error_bound=bound,
        cdf_first_raw=float(cdf[0]) if cdf.size else None,cdf_last_raw=float(cdf[-1]) if cdf.size else None,
        cdf_min_delta_raw=float(delta.min()) if delta.size else 0.,cdf_negative_delta_count=int(np.count_nonzero(delta<0)),
        cdf_max_abs_reference_error=error)
