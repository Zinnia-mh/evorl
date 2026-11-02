"""Offline-only cached simulation. Original sampler and benchmark calls retained."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .utility_replay_calibration_pipeline import (
    validate_snapshot_inputs,sync,_combine,PROTOCOL,SIMULATION_TAG)
from .source_replay import (init_source_score_table,compute_source_eligibility,
    build_weighted_source_distribution,sample_weighted_ea_indices)

IMPLEMENTATION='offline-v2-cached-source-map-1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_create(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)  # Exclusive publication; never overwrites a measurement.
    temporary.unlink()


class CandidateStore:
    def __init__(self,root,identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.identity=dict(identity);self.identity_sha256=digest(identity)
        path=self.root/'identity.json'
        if path.exists():
            if json.loads(path.read_text())!=identity: raise ValueError('checkpoint analysis identity mismatch')
        else:
            if any(self.root.iterdir()): raise ValueError('incomplete checkpoint identity')
            atomic_create(path,identity)

    def path(self,generation,candidate_id):
        return self.root/f'generation_{generation:06d}'/f'candidate_{candidate_id:03d}.json'

    def save(self,generation,candidate_id,row,details):
        record=dict(identity_sha256=self.identity_sha256,generation=generation,candidate_id=candidate_id,row=row,details=details)
        record['sha256']=digest(record)
        atomic_create(self.path(generation,candidate_id),record)

    def load(self,generation,candidate_id):
        path=self.path(generation,candidate_id)
        if not path.exists(): return None
        record=json.loads(path.read_text());sha=record.pop('sha256')
        if digest(record)!=sha or record['identity_sha256']!=self.identity_sha256 or record['generation']!=generation or record['candidate_id']!=candidate_id:
            raise ValueError('corrupt or foreign candidate checkpoint')
        return record['row'],record['details']


@jax.jit
def expand_transition_probabilities(source_transition,slots,ea):
    return jnp.where(ea,source_transition[slots],0.).astype(jnp.float32)


@jax.jit
def transition_cdf(probability,has_ea):
    cdf=jnp.cumsum(probability)
    cdf=cdf.at[-1].set(jnp.where(has_ea,jnp.maximum(cdf[-1],1.),0.))
    return cdf


@jax.jit
def device_checks(probability,source_probability,chosen,ea,counts,eligible,alpha):
    q=counts.astype(jnp.float32)/jnp.sum(counts,dtype=jnp.float32)
    correct=(jnp.all(jnp.isfinite(probability)) & jnp.all(probability>=0)
        & (jnp.abs(jnp.sum(probability)-1)<1e-5) & jnp.all(ea[chosen])
        & jnp.all(jnp.where(ea,probability >= (1-alpha)/jnp.sum(ea)-1e-7,True))
        & jnp.all(jnp.where(eligible,True,jnp.isclose(source_probability,q,atol=1e-6,rtol=1e-5))))
    ordered=jnp.sort(chosen)
    distinct=1+jnp.sum(ordered[1:]!=ordered[:-1])
    return correct,distinct


class SnapshotSimulator:
    def __init__(self,meta,arrays):
        start=time.perf_counter()
        self.meta=meta
        self.generation,self.ea,self.tokens,self.counts,self.history,self.measured=validate_snapshot_inputs(meta,arrays)
        slots=np.searchsorted(self.tokens,arrays['source_token'],side='left')
        slots=np.clip(slots,0,len(self.tokens)-1).astype(np.int32)
        self.device_slots=jnp.array(slots);self.device_ea=jnp.array(self.ea)
        self.device_tokens=jnp.array(self.tokens,dtype=jnp.int32)
        self.device_counts=jnp.array(self.counts,dtype=jnp.int32)
        self.source_block=1<<(len(self.tokens)-1).bit_length()
        self.device_counts_padded=self.pad_source(self.device_counts)
        self.source_mask=jnp.ones((len(self.tokens),),dtype=jnp.bool_)
        self.tables={};self.eligibility={};self.padded_eligibility={}
        for kind,kind_id in [('fitness',1),('td',2),('utility',3)]:
            raw=[];rank=[];valid=[];generation=[];versions=[]
            for token in self.tokens:
                record=self.history.get(int(token));value=None if record is None else record[kind]
                raw.append(np.nan if value is None else value)
                rank.append(0. if record is None else record['ranks'][kind])
                valid.append(False if record is None else record['valid'])
                generation.append(-1 if record is None else record['measured_generation'])
                versions.append(-1 if record is None else record['observer_score_version'])
            self.tables[kind]=init_source_score_table(len(self.tokens))._replace(
                source_token=self.device_tokens,resident_count=self.device_counts,
                raw_score=jnp.array(raw),rank_normalized_score=jnp.array(rank),validity=jnp.array(valid),
                measured_generation=jnp.array(generation,dtype=jnp.int32),version=jnp.array(versions,dtype=jnp.int32),
                score_kind=jnp.full((len(self.tokens),),kind_id,dtype=jnp.int8))
        sync((self.device_slots,self.device_ea,self.tables))
        self.preparation_ms=(time.perf_counter()-start)*1000

    def pad_source(self,array):
        return jnp.pad(array,(0,self.source_block-len(self.tokens)),constant_values=0)

    def prepare_candidates(self,candidates):
        start=time.perf_counter()
        for kind in self.tables:
            for window,minimum in sorted({(c['W'],c['M_min']) for c in candidates}):
                key=(kind,window,minimum)
                if key not in self.eligibility:
                    self.eligibility[key]=compute_source_eligibility(self.tables[kind],self.generation,window,minimum)
                self.padded_eligibility[key]=self.pad_source(self.eligibility[key].eligible)
        sync((self.eligibility,self.padded_eligibility,self.device_counts_padded))
        self.preparation_ms+=(time.perf_counter()-start)*1000

    def distribution(self,candidate,kind):
        key=(kind,candidate['W'],candidate['M_min'])
        if key not in self.eligibility:
            self.eligibility[key]=compute_source_eligibility(
                self.tables[kind],self.generation,candidate['W'],candidate['M_min'])
        eligibility=self.eligibility[key]
        # The authoritative primitive calculates all source weights and each
        # source's per-transition probability on a compact identity mapping.
        compact=build_weighted_source_distribution(eligibility.table,self.device_tokens,self.source_mask,
            candidate['alpha'],candidate['tau'])
        # Separate executables keep the million-element prefix sum independent
        # of source-count/static-weight specialization and gather fusion.
        probability=expand_transition_probabilities(self.pad_source(compact.transition_probability),
            self.device_slots,self.device_ea)
        cdf=transition_cdf(probability,compact.has_resident_ea)
        return eligibility,compact._replace(transition_probability=probability,transition_cdf=cdf)

    def measure(self,candidate_id,candidate,draws=1024):
        if draws<1: raise ValueError('positive draw count required')
        observations=[];details=[]
        for kind,kind_id in [('fitness',1),('td',2),('utility',3)]:
            start=time.perf_counter();eligibility,distribution=self.distribution(candidate,kind)
            sync(distribution);simulation_ms=(time.perf_counter()-start)*1000
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(104729),SIMULATION_TAG),self.generation*3+kind_id)
            # Unchanged warmup, sample count, key, sampler and timing boundary.
            first=sample_weighted_ea_indices(distribution,key,draws);sync(first)
            start=time.perf_counter();sample=sample_weighted_ea_indices(distribution,key,draws);sync(sample)
            overhead_ms=(time.perf_counter()-start)*1000
            padded_source=self.pad_source(distribution.source_probability)
            eligible_key=(kind,candidate['W'],candidate['M_min'])
            if eligible_key not in self.padded_eligibility:
                self.padded_eligibility[eligible_key]=self.pad_source(eligibility.eligible)
            check,distinct=device_checks(distribution.transition_probability,padded_source,sample.sampled_indices,self.device_ea,
                self.device_counts_padded,self.padded_eligibility[eligible_key],candidate['alpha'])
            host=jax.device_get((check,distinct,distribution.ess_fraction,distribution.max_source_probability,
                distribution.source_probability,eligibility.eligible))
            correct,unique,ess,max_mass,p_source,eligible=host
            allocations={}
            for array in jtu.tree_leaves((self.tables,self.eligibility,distribution,first,sample,
                                          self.device_slots,self.device_ea,self.source_mask,
                                          self.device_counts_padded,self.padded_eligibility,padded_source)):
                if hasattr(array,'unsafe_buffer_pointer'):
                    pointer=array.unsafe_buffer_pointer();allocations[pointer]=max(allocations.get(pointer,0),array.nbytes)
            measured_generation=np.array([-1 if int(t) not in self.history else self.history[int(t)]['measured_generation'] for t in self.tokens])
            age=self.generation-measured_generation
            fresh=(age>=0)&(age<=candidate['W'])&self.measured
            denominator=self.counts[self.measured].sum()
            row=dict(candidate_id=candidate_id,ess_fraction=float(ess),max_source_probability=float(max_mass),
                uniform_floor=1-candidate['alpha'],fresh_coverage=float(self.counts[fresh].sum()/denominator),
                expired_coverage=float(self.counts[self.measured&~fresh].sum()/denominator),
                repeat_rate=1-int(unique)/draws,numerical_ok=bool(correct),
                memory_mib=sum(allocations.values())/2**20,overhead_ms=overhead_ms)
            observations.append(row)
            details.append(dict(row,kind=kind,generation=self.generation,simulation_ms=simulation_ms,
                snapshot_preparation_ms=self.preparation_ms,analysis_implementation=IMPLEMENTATION,
                memory_measurement=PROTOCOL['memory_cost']+'; includes resident offline caches',
                unscored_coverage=float(self.counts[~self.measured].sum()/self.counts.sum()),
                source_probability=p_source.tolist(),source_token=self.tokens.tolist(),eligible=eligible.tolist(),
                diagnostic_ms=self.meta['timing']['diagnostic_ms'],real_uniform_sampling_ms=self.meta['timing']['sampling_ms']))
        return _combine(observations,candidate_id),details


def simulate_snapshot(meta,arrays,reg,*,draws=1024,checkpoint=None,progress=None):
    plan=None;rows=[];details=[]
    for candidate_id,candidate in enumerate(reg['candidates']):
        saved=None if checkpoint is None else checkpoint.load(meta['generation'],candidate_id)
        if saved is None:
            if plan is None:
                plan=SnapshotSimulator(meta,arrays)
                plan.prepare_candidates(reg['candidates'])
            row,measurements=plan.measure(candidate_id,candidate,draws)
            if checkpoint is not None: checkpoint.save(meta['generation'],candidate_id,row,measurements)
        else: row,measurements=saved
        rows.append(row);details.extend(measurements)
        if progress is not None:
            progress(dict(event='candidate_complete',generation=meta['generation'],candidate_id=candidate_id,
                score_kinds_complete=3,cached=saved is not None))
    return rows,details
