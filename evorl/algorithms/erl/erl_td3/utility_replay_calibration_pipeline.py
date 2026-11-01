"""Calibration-only observer and offline simulation using the Pilot sampler."""
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from .utility_replay_calibration_capture import capture, write_snapshot
from .source_replay import (init_source_score_table, compute_source_eligibility,
    build_weighted_source_distribution, sample_weighted_ea_indices,
    rank_normalize_within_checkpoint)

PROTOCOL = dict(version='calibration-observer-v1', boundary='post_step_pre_next_rollout',
                diagnostic_interval=10, warmup=10, generations=60, first_analysis_generation=20,
                draws=1024, aggregation='mean ESS/coverage; max mass/repeat/memory/sampling-ms; all numerical-ok',
                sampling_cost='absolute synchronized 1024 conditional-EA draws in ms; not a training-time delta',
                memory_cost='deduplicated live device buffer bytes; not allocator/process peak',
                coverage_denominator='resident transitions with measured scores; unscored mass logged separately')
SIMULATION_TAG=0x53494D55
BENCHMARK_TAG=0x42454E43
METRIC_FIELDS={'candidate_id','ess_fraction','max_source_probability','uniform_floor','fresh_coverage',
              'expired_coverage','repeat_rate','numerical_ok','memory_mib','overhead_ms'}


def sync(tree):
    for leaf in jtu.tree_leaves(tree):
        if hasattr(leaf,'block_until_ready'): leaf.block_until_ready()


class CalibrationObserver:
    def __init__(self, directory=None, *, interval=10):
        if interval != 10: raise ValueError('registered diagnostic interval is 10')
        self.directory=directory; self.history={}; self.version=-1; self.last_generation=None
        self.checkpoints=[]
        self.snapshots=[]; self.records=(); self.diagnostic_ms=0.

    def begin(self, workflow, state):
        if workflow.config.arm!='uniform' or float(workflow.config.alpha)!=0:
            raise ValueError('observer requires unchanged uniform calibration trajectory')
        self.context=workflow._make_pre_exposure_context(state)
        self.records=(); self.diagnostic_ms=0.

    def score(self, workflow, replay_before, metrics, flat_trajectory):
        generation=self.context.generation
        if generation % 10: return
        pop=int(workflow.config.pop_size); episodes=int(workflow.config.episodes_for_fitness)
        time_size=jtu.tree_leaves(flat_trajectory)[0].shape[0]//(pop*episodes)
        def reshape(x):
            return x.reshape((time_size,pop,episodes,*x.shape[1:])).swapaxes(0,1)
        raw=jtu.tree_map(reshape,flat_trajectory)
        dones=(jnp.arange(time_size)[None,:,None] >= metrics.episode_lengths[:,None,:]-1)
        raw=raw.replace(dones=dones)
        self.context.fitnesses=metrics.episode_returns.mean(-1)
        if hasattr(workflow,'_diagnostic_context'): raise ValueError('diagnostic context collision')
        workflow._diagnostic_context=self.context
        start=time.perf_counter()
        try: self.records=workflow._run_pre_exposure_diagnostic(raw,replay_before)
        finally: del workflow._diagnostic_context
        self.diagnostic_ms=(time.perf_counter()-start)*1000
        self.version+=1
        self.checkpoints.append(generation)
        scores={kind:[] for kind in ['fitness','td','utility']}
        for r in self.records:
            scores['fitness'].append(r.fitness); scores['td'].append(r.td_error); scores['utility'].append(r.utility)
        valid=np.array([r.eligible for r in self.records],bool)
        ranks={kind:np.asarray(rank_normalize_within_checkpoint(
            jnp.array([np.nan if x is None else x for x in values]),jnp.array(valid))) for kind,values in scores.items()}
        for i,r in enumerate(self.records):
            token=generation*pop+r.individual_id
            self.history[token]=dict(source_token=token,measured_generation=generation,
                observer_score_version=self.version,valid=bool(r.eligible),
                fitness=r.fitness,td=r.td_error,utility=r.utility,
                reference_fingerprint=r.reference_fingerprint,
                ranks={kind:float(values[i]) for kind,values in ranks.items()})

    def finish(self, workflow, state):
        generation=int(state.metrics.iterations)
        if self.last_generation is not None and generation!=self.last_generation+1:
            raise ValueError('missing per-generation observation')
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(104729),BENCHMARK_TAG),generation)
        # Same real Replay sampling implementation, separately timed and keyed.
        batch=workflow.replay_buffer.sample(state.replay_buffer_state,key); sync(batch)
        start=time.perf_counter(); batch=workflow.replay_buffer.sample(state.replay_buffer_state,key); sync(batch)
        sampling_ms=(time.perf_counter()-start)*1000
        replay=state.replay_buffer_state
        memory=sum(np.asarray(getattr(replay,k)).nbytes for k in
                   ['origin_type','source_token','source_generation','score_version'])/2**20
        metadata,arrays=capture(replay,self.records,generation,population_size=int(workflow.config.pop_size),
            timing=dict(diagnostic_ms=self.diagnostic_ms,sampling_ms=sampling_ms),memory_mib=memory)
        resident=set(map(int,np.unique(arrays['source_token'][:int(replay.buffer_size)])))
        self.history={k:v for k,v in self.history.items() if k in resident}
        metadata.update(boundary=PROTOCOL['boundary'],score_history=list(self.history.values()),
                        scored_generations=list(self.checkpoints),
                        memory_measurement='host-sidecar-copy-bytes',
                        diagnostic_reference='pre-insertion replay and pre-update learner',
                        real_sampling_measurement='uniform Replay.sample, synchronized, compile warmup excluded')
        if self.directory: write_snapshot(self.directory,metadata,arrays)
        else: self.snapshots.append((metadata,arrays))
        self.last_generation=generation
        del self.context


def simulate(meta, arrays, reg, *, draws=1024):
    def reject_outcomes(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if any(word in str(key).lower() for word in ['return','auc','performance','reward']):
                    raise ValueError('outcome leakage in snapshot')
                reject_outcomes(item)
        elif isinstance(value,list):
            for item in value: reject_outcomes(item)
    reject_outcomes(meta)
    if set(arrays)!={'origin_type','source_token','source_generation','score_version'}:
        raise ValueError('unexpected payload or outcome arrays in calibration snapshot')
    if any(x.ndim!=1 or x.dtype.kind not in 'iu' for x in arrays.values()):
        raise ValueError('invalid sidecar shape or dtype')
    allowed={'schema','generation','capacity','buffer_size','current_index','resident_counts','scores','timing','population_size',
             'memory_mib','simulation_prng','score_history','boundary','memory_measurement',
             'diagnostic_reference','real_sampling_measurement','scored_generations'}
    if set(meta)-allowed: raise ValueError('unknown or outcome snapshot fields')
    if meta.get('boundary')!=PROTOCOL['boundary']: raise ValueError('unknown temporal boundary')
    generation=int(meta['generation']); size=int(meta['buffer_size']); capacity=int(meta['capacity'])
    if not 0<size<=capacity or any(len(x)!=capacity for x in arrays.values()): raise ValueError('invalid replay')
    ea=(np.arange(capacity)<size)&(arrays['origin_type']==0)
    if np.any(~np.isin(arrays['origin_type'][:size],[0,1])) or np.any(arrays['source_generation'][ea]>generation):
        raise ValueError('invalid or future replay origin')
    tokens,counts=np.unique(arrays['source_token'][ea],return_counts=True)
    if len(tokens)==0 or np.any(tokens<0): raise ValueError('no identifiable resident EA sources')
    if dict(zip(map(str,tokens),map(int,counts)))!=meta['resident_counts']: raise ValueError('resident count mismatch')
    history=meta['score_history']; by_token={int(x['source_token']):x for x in history}
    if len(by_token)!=len(history): raise ValueError('duplicate source score')
    checkpoints=meta.get('scored_generations',[])
    if any(g>generation for g in checkpoints): raise ValueError('future measurement checkpoint')
    pop=meta.get('population_size')
    if pop is not None and any(int(t)//pop in checkpoints and int(t) not in by_token for t in tokens):
        raise ValueError('measured resident source score missing')
    for r in history:
        if set(r)-{'source_token','measured_generation','observer_score_version','valid','fitness','td','utility','ranks','reference_fingerprint'}:
            raise ValueError('unknown or outcome score fields')
        if r['measured_generation']>generation: raise ValueError('future score leakage')
        if type(r['valid']) is not bool or r['measured_generation']<0 or r['observer_score_version']<0:
            raise ValueError('invalid score validity/version')
        if r['valid'] and any(r.get(k) is None or not np.isfinite(r[k]) for k in ['fitness','td','utility']):
            raise ValueError('invalid eligible score')
        if set(r['ranks'])!={'fitness','td','utility'} or any(not np.isfinite(x) or not 0<=x<=1 for x in r['ranks'].values()):
            raise ValueError('invalid registered checkpoint rank')
    measured=np.array([int(t) in by_token for t in tokens])
    if not measured.any(): raise ValueError('no measured resident sources at analysis boundary')
    rows=[]; details=[]
    for candidate_id,candidate in enumerate(reg['candidates']):
        observations=[]
        for kind,kind_id in [('fitness',1),('td',2),('utility',3)]:
            start=time.perf_counter()
            table=init_source_score_table(len(tokens))
            raw=[]; rank=[]; validity=[]; measured_generation=[]; versions=[]
            for token in tokens:
                r=by_token.get(int(token)); value=None if r is None else r[kind]
                raw.append(np.nan if value is None else value); rank.append(0. if r is None else r['ranks'][kind])
                validity.append(False if r is None else r['valid']); measured_generation.append(-1 if r is None else r['measured_generation'])
                versions.append(-1 if r is None else r['observer_score_version'])
            table=table._replace(source_token=jnp.array(tokens,dtype=jnp.int32),resident_count=jnp.array(counts,dtype=jnp.int32),
                raw_score=jnp.array(raw),rank_normalized_score=jnp.array(rank),validity=jnp.array(validity),
                measured_generation=jnp.array(measured_generation,dtype=jnp.int32),version=jnp.array(versions,dtype=jnp.int32),
                score_kind=jnp.full((len(tokens),),kind_id,dtype=jnp.int8))
            eligibility=compute_source_eligibility(table,generation,candidate['W'],candidate['M_min'])
            distribution=build_weighted_source_distribution(eligibility.table,jnp.array(arrays['source_token']),jnp.array(ea),candidate['alpha'],candidate['tau'])
            sync(distribution); simulation_ms=(time.perf_counter()-start)*1000
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(104729),SIMULATION_TAG),generation*3+kind_id)
            first=sample_weighted_ea_indices(distribution,key,draws); sync(first)
            start=time.perf_counter(); sample=sample_weighted_ea_indices(distribution,key,draws); sync(sample)
            overhead_ms=(time.perf_counter()-start)*1000
            probabilities=np.asarray(distribution.transition_probability)
            chosen=np.asarray(sample.sampled_indices)
            correct=bool(np.all(np.isfinite(probabilities)) and np.all(probabilities>=0) and
                         abs(probabilities.sum()-1)<1e-5 and np.all(ea[chosen]))
            correct=correct and bool(np.all(probabilities[ea] >= (1-candidate['alpha'])/ea.sum()-1e-7))
            p_source=np.asarray(distribution.source_probability)
            ineligible=~np.asarray(eligibility.eligible)
            correct=correct and bool(np.allclose(p_source[ineligible],counts[ineligible]/counts.sum(),atol=1e-6))
            allocations={}
            for array in jtu.tree_leaves((table,eligibility,distribution,sample)):
                if hasattr(array,'unsafe_buffer_pointer'):
                    allocations[array.unsafe_buffer_pointer()]=max(allocations.get(array.unsafe_buffer_pointer(),0),array.nbytes)
            age=generation-np.array(measured_generation); fresh=(age>=0)&(age<=candidate['W'])&measured
            denominator=counts[measured].sum()
            obs=dict(candidate_id=candidate_id,ess_fraction=float(distribution.ess_fraction),
                max_source_probability=float(distribution.max_source_probability),uniform_floor=1-candidate['alpha'],
                fresh_coverage=float(counts[fresh].sum()/denominator),expired_coverage=float(counts[measured&~fresh].sum()/denominator),
                repeat_rate=1-len(np.unique(chosen))/draws,numerical_ok=correct,
                memory_mib=sum(allocations.values())/2**20,
                overhead_ms=overhead_ms)
            observations.append(obs)
            details.append(dict(obs,kind=kind,generation=generation,simulation_ms=simulation_ms,
                memory_measurement=PROTOCOL['memory_cost'],
                unscored_coverage=float(counts[~measured].sum()/counts.sum()),
                source_probability=np.asarray(distribution.source_probability).tolist(),
                source_token=tokens.tolist(),eligible=np.asarray(eligibility.eligible).tolist(),
                diagnostic_ms=meta['timing']['diagnostic_ms'],real_uniform_sampling_ms=meta['timing']['sampling_ms']))
        rows.append(_combine(observations,candidate_id))
    return rows,details


def _combine(rows, candidate_id):
    result=dict(candidate_id=candidate_id,numerical_ok=all(r['numerical_ok'] for r in rows))
    for key in ['ess_fraction','fresh_coverage','expired_coverage']: result[key]=float(np.mean([r[key] for r in rows]))
    for key in ['max_source_probability','repeat_rate','memory_mib','overhead_ms']: result[key]=max(r[key] for r in rows)
    result['uniform_floor']=min(r['uniform_floor'] for r in rows)
    return result


def aggregate(generations, reg):
    if not generations: raise ValueError('missing snapshot corpus')
    expected=set(range(len(reg['candidates'])))
    for rows in generations:
        if any(set(r)!=METRIC_FIELDS for r in rows): raise ValueError('unknown or outcome aggregate fields')
        if len(rows)!=len(expected) or {r['candidate_id'] for r in rows}!=expected: raise ValueError('missing candidate data')
    return [_combine([next(r for r in rows if r['candidate_id']==i) for rows in generations],i) for i in sorted(expected)]
