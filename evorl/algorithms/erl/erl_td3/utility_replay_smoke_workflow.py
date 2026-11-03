"""Host-only Smoke audit; index reconstruction never enters the learner graph."""
import hashlib,threading,time
from functools import partial
from typing import NamedTuple
import jax,jax.numpy as jnp
import numpy as np
from .erl_utility_replay import (ERLUtilityReplayWorkflow,alpha_zero_sample,weighted_replay_sample,weighted_replay_draws,_optimizer_step_count)
from .source_replay import compute_source_eligibility,build_weighted_source_distribution,draw_baseline_indices
from .source_residency_batch import recount_resident_sources
from .replay_distribution_audit import audit_distribution

class DrawRecorder:
    def __init__(self):self.lock=threading.Lock();self.reset()
    def reset(self):
        with self.lock:
            self.values=dict(observed_batches=0,ea_slots=0,rl_slots=0,changed_ea=0,rl_changed=0,origin_changed=0)
            self.digest=hashlib.sha256()
            self.final_digest=hashlib.sha256()
    def record(self,baseline,final,ea,rl,origin_ok):
        baseline=np.asarray(baseline,dtype=np.int32);final=np.asarray(final);ea=np.asarray(ea,bool);rl=np.asarray(rl,bool)
        with self.lock:
            self.values['observed_batches']+=baseline.size//baseline.shape[-1]
            self.values['ea_slots']+=int(ea.sum());self.values['rl_slots']+=int(rl.sum())
            self.values['changed_ea']+=int(((baseline!=final)&ea).sum())
            self.values['rl_changed']+=int(((baseline!=final)&rl).sum())
            self.values['origin_changed']+=int((~np.asarray(origin_ok,bool)).sum())
            for b,m in zip(baseline.reshape(-1,baseline.shape[-1]),ea.reshape(-1,ea.shape[-1])):
                self.digest.update(b.tobytes());self.digest.update(m.tobytes())
            self.final_digest.update(np.asarray(final,dtype=np.int32).tobytes())
    def summary(self):
        with self.lock:return dict(self.values,baseline_draw_sha256=self.digest.hexdigest(),final_draw_sha256=self.final_digest.hexdigest())

class ReconstructedDraws(NamedTuple):
    baseline_indices: object
    final_indices: object
    ea_slot_mask: object
    rl_slot_mask: object
    origin_ok: object
    terminal_learning_key: object
    terminal_intervention_key: object

@partial(jax.jit,static_argnames=('num_updates','num_draws','batch_size','uniform'))
def reconstruct_replay_draws(replay,key,utility,*,num_updates,num_draws,batch_size,uniform):
    """Replay the exact sampling key schedule against immutable learner inputs.

    This is a separate executable, called after training. It returns no learner
    updates and its keys are checked then discarded, never fed into training.
    """
    def one(carry,_):
        learning_key,intervention_key=carry
        learning_key,rb_key,_=jax.random.split(learning_key,3)
        replay_keys=jax.random.split(rb_key,num_draws)
        if uniform:
            baseline=jax.vmap(lambda k:draw_baseline_indices(replay,k,batch_size))(replay_keys)
            final=baseline;ea=replay.origin_type[baseline]==0;rl=replay.origin_type[baseline]==1
        else:
            sample=weighted_replay_draws(replay,replay_keys,utility._replace(intervention_key=intervention_key),batch_size)
            baseline=sample.baseline_indices;final=sample.final_indices;ea=sample.ea_slot_mask;rl=sample.rl_slot_mask
            intervention_key=sample.utility_replay_state.intervention_key
        origin_ok=jnp.where(ea,replay.origin_type[final]==0,replay.origin_type[final]==1)
        return (learning_key,intervention_key),(baseline,final,ea,rl,origin_ok)
    (learning_key,intervention_key),trace=jax.lax.scan(one,(key,utility.intervention_key),(),length=num_updates)
    return ReconstructedDraws(*trace,learning_key,intervention_key)

def initialization_fingerprint(state):
    """Hash actual common initialization once, before any learning feedback."""
    digest=hashlib.sha256()
    for path,leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if hasattr(leaf,'dtype') and jax.dtypes.issubdtype(leaf.dtype,jax.dtypes.prng_key):leaf=jax.random.key_data(leaf)
        value=np.asarray(jax.device_get(leaf))
        digest.update(jax.tree_util.keystr(path).encode());digest.update(str((value.dtype,value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()

class SmokeWorkflow(ERLUtilityReplayWorkflow):
    @classmethod
    def _build_from_config(cls,config):
        workflow=super()._build_from_config(config);workflow.draw_recorder=DrawRecorder()
        return workflow
    def _weighted_rl_update(self,agent_state,opt_state,replay_buffer_state,key,num_updates,utility_replay_state):
        result=super()._weighted_rl_update(agent_state,opt_state,replay_buffer_state,key,num_updates,utility_replay_state)
        self._pending_sampling=(replay_buffer_state,key,num_updates,utility_replay_state)
        return result
    def _rl_update(self,agent_state,opt_state,replay_buffer_state,key,num_updates):
        result=super()._rl_update(agent_state,opt_state,replay_buffer_state,key,num_updates)
        self._pending_sampling=(replay_buffer_state,key,num_updates,self._uniform_input_utility)
        return result
    def step(self,state):
        self.draw_recorder.reset();self._pending_sampling=None
        self._uniform_input_utility=state.utility_replay_state
        start=time.perf_counter();metrics,after=super().step(state)
        jax.block_until_ready((metrics,after))
        training_seconds=time.perf_counter()-start
        if self._pending_sampling is None:raise ValueError('missing actual learner invocation')
        replay,key,updates,utility=self._pending_sampling
        if not np.array_equal(after.key,jax.random.split(state.key,4)[0]):raise ValueError('baseline PRNG transition changed')
        if _optimizer_step_count(after.opt_state.actor)-_optimizer_step_count(state.opt_state.actor)!=int(updates):raise ValueError('actual learner count differs from reconstruction')
        if _optimizer_step_count(after.opt_state.critic)-_optimizer_step_count(state.opt_state.critic)!=int(updates)*int(self.config.actor_update_interval):raise ValueError('actual critic count differs from reconstruction')
        started=time.perf_counter()
        trace=reconstruct_replay_draws(replay,key,utility,num_updates=int(updates),num_draws=int(self.config.actor_update_interval*self.config.num_rl_agents),batch_size=int(self.config.batch_size),uniform=float(self.config.alpha)==0)
        if not np.array_equal(trace.terminal_intervention_key,after.utility_replay_state.intervention_key):raise ValueError('intervention PRNG reconstruction mismatch')
        self.draw_recorder.record(*jax.device_get(trace[:5]));reconstruction_seconds=time.perf_counter()-started
        self._pending_sampling=None;del self._uniform_input_utility
        self._latest_smoke_audit=self.observe_generation(state,after,training_seconds)
        self._latest_smoke_audit.update(sampling_observation='deterministic_reconstruction',baseline_prng_transition_verified=True,intervention_terminal_key_verified=True,index_reconstruction_seconds=reconstruction_seconds)
        self._latest_smoke_audit['observer_seconds']+=reconstruction_seconds
        return metrics,after
    def observe_generation(self,before,after,training_seconds):
        start=time.perf_counter();c=self.config;g=int(after.metrics.iterations);replay=after.replay_buffer_state
        ea=(jnp.arange(replay.source_token.size)<replay.buffer_size)&(replay.origin_type==0)
        table=after.utility_replay_state.score_table
        if float(c.alpha)==0:table=recount_resident_sources(table,replay.source_token,ea).table
        eligibility=compute_source_eligibility(table,g,int(c.freshness_window),int(c.min_resident))
        if float(c.alpha)==0:
            dist=build_weighted_source_distribution(eligibility.table,replay.source_token,ea,0.,float(c.replay_temperature))
        else:dist=after.utility_replay_state.distribution
        counts,measured,valid=jax.device_get((table.resident_count,table.measured_generation,table.validity))
        total=max(1,int(np.sum(counts)));age=g-np.asarray(measured);scored=np.asarray(measured)>=0
        fraction=lambda mask:float(np.sum(counts[mask])/total)
        old=compute_source_eligibility(table,g+int(c.freshness_window)+1,int(c.freshness_window),int(c.min_resident))
        probe=build_weighted_source_distribution(old.table,replay.source_token,ea,float(c.alpha),float(c.replay_temperature))
        neutral=bool(jax.device_get(jnp.allclose(probe.source_probability,probe.baseline_source_probability,atol=1e-6,rtol=1e-5)))
        key=jax.random.fold_in(before.key,0x534D4B31)
        intervention=jax.random.fold_in(after.utility_replay_state.intervention_key,0x534D4B31)
        def sampling_probe():
            if float(c.alpha)==0:return alpha_zero_sample(replay,key,intervention,int(c.batch_size))
            return weighted_replay_sample(replay,key,intervention,dist,after.utility_replay_state.distribution_prepared,int(c.batch_size))
        jax.block_until_ready(sampling_probe());t=time.perf_counter();jax.block_until_ready(sampling_probe());sampling_seconds=time.perf_counter()-t
        finite=bool(jax.device_get(jnp.all(jnp.isfinite(dist.transition_probability))))
        for leaf in jax.tree.leaves((after.agent_state,after.opt_state)):
            if hasattr(leaf,'dtype') and jnp.issubdtype(leaf.dtype,jnp.inexact):finite=finite and bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
        lifecycle=getattr(self,'_latest_score_lifecycle_audit',{})
        # Host-only evidence; the distribution and all learner inputs remain
        # untouched. Retain the original device ESS alongside normalized ESS.
        p,t,cdf,ea_host=jax.device_get((dist.source_probability,dist.transition_probability,dist.transition_cdf,ea))
        numerical=audit_distribution(p,counts,t,cdf,ea_host,float(dist.ess_fraction))
        result=dict(self.draw_recorder.summary(),generation=g,finite=finite,training_seconds=training_seconds,
            probability_l1=float(jax.device_get(jnp.sum(jnp.abs(dist.source_probability-dist.baseline_source_probability)))),
            eligible_baseline_mass=float(dist.eligible_baseline_mass),ess_fraction=float(dist.ess_fraction),max_source_probability=float(dist.max_source_probability),
            valid_score_fraction=fraction(np.asarray(valid)&scored),fresh_fraction=fraction(scored&(age>=0)&(age<=int(c.freshness_window))),expired_fraction=fraction(scored&(age>int(c.freshness_window))),unscored_fraction=fraction(~scored),
            minimum_scored_age=int(np.min(age[scored&(counts>0)])) if np.any(scored&(counts>0)) else None,
            maximum_scored_age=int(np.max(age[scored&(counts>0)])) if np.any(scored&(counts>0)) else None,
            freshness_probe_neutral=neutral,freshness_probe_scope='read-only counterfactual; not observed expiry',
            reference_fingerprints=list(lifecycle.get('reference_fingerprints',[])),eval_reference_check='B_eval endpoint disabled' if not c.enable_eval_loss_endpoint else 'requires distinct B_eval fingerprint',
            sampling_probe_seconds=sampling_seconds,sampling_timing_scope='independent-key real-state sampler probe; not in-training sampler duration',
            diagnostic_seconds=float(lifecycle.get('diagnostic_seconds',0)),resident_count_seconds=float(lifecycle.get('resident_count_seconds',0)),distribution_refresh_seconds=float(lifecycle.get('distribution_refresh_seconds',0)),
            actor_optimizer_steps=_optimizer_step_count(after.opt_state.actor)-_optimizer_step_count(before.opt_state.actor),
            critic_optimizer_steps=_optimizer_step_count(after.opt_state.critic)-_optimizer_step_count(before.opt_state.critic))
        result.update(numerical)
        result.update(distribution_prepared=bool(after.utility_replay_state.distribution_prepared),
            eligible_source_count=int(dist.eligible_source_count),
            valid_score_source_count=int(np.sum((counts>0)&np.asarray(valid)&scored&(age>=0)&(age<=int(c.freshness_window))&(np.asarray(table.version)==int(table.current_version)))),
            effective_eligible_mass=float(c.alpha)*float(dist.eligible_baseline_mass))
        result['ess_fraction']=numerical['ess_fraction_audit64']
        result['observer_seconds']=time.perf_counter()-start
        return result
