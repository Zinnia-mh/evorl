"""R: persistent random source ranks; no Utility, virtual update or reference batch."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.budgeted_utility_scoring import score_quota,SCHEDULE_TAG
from evorl.algorithms.erl.erl_td3.learner_utility import derive_diagnostic_key
from evorl.algorithms.erl.erl_td3.erl_utility_replay import make_ea_rollout_provenance
from evorl.algorithms.erl.erl_td3.source_replay import CheckpointScoreUpdate,apply_checkpoint_score_update,raise_for_score_table_status
from scripts.run_s_utility_extension import permute_cohort_ranks

# Existing checkpoint/sampler ABI only recognizes tags1/2/3. This tag is a
# schema adapter, NOT a claim that synthetic ranks are measured Utility.
COMPATIBILITY_SCORE_TAG=3
PRIORITY_KIND="synthetic_random_cohort_rank"

def selected_source_ids(baseline_key,generation,population_size,interval,warmup_iters):
    quota=score_quota(generation,population_size,interval,warmup_iters)
    key=derive_diagnostic_key(baseline_key,generation,SCHEDULE_TAG)
    return np.sort(np.asarray(jax.random.permutation(key,population_size))[:quota])

def synthetic_update(generation,individual_ids,validity,version,population_size):
    ids=np.asarray(individual_ids,dtype=np.int32);valid=np.asarray(validity,dtype=bool)
    ranks=np.zeros(len(ids),dtype=np.float32);n=int(valid.sum())
    if n:ranks[valid]=np.linspace(0,1,n,dtype=np.float32) if n>1 else .5
    return CheckpointScoreUpdate(
        source_token=jnp.asarray(generation*population_size+ids,dtype=jnp.int32),
        raw_score=jnp.asarray(ranks),rank_normalized_score=jnp.asarray(ranks),
        measured_generation=jnp.full((len(ids),),generation,jnp.int32),
        score_kind=jnp.full((len(ids),),COMPATIBILITY_SCORE_TAG,jnp.int8),
        validity=jnp.asarray(valid),version=jnp.asarray(version,jnp.int32))

class RandomSourcePriorityWorkflow(AuditedD0):
    @classmethod
    def _build_from_config(cls,config):
        w=super()._build_from_config(config)
        def transform(table):
            started=time.perf_counter()
            result=permute_cohort_ranks(table,int(config.seed),w.permutation)
            w.permutation_seconds+=time.perf_counter()-started
            return result
        w._ablation_rank_transform=transform
        return w

    def _diagnostic_due(self,state):
        return False
    def _make_pre_exposure_context(self,*args,**kwargs):
        raise AssertionError("R must not create frozen Utility contexts")
    def _run_pre_exposure_diagnostic(self,*args,**kwargs):
        raise AssertionError("R must not run Utility diagnostics")
    def _evaluate_source_utility(self,*args,**kwargs):
        raise AssertionError("R must not evaluate Utility or virtual critic updates")
    def _sample_current_reference(self,*args,**kwargs):
        raise AssertionError("R must not create reference batches")
    def _select_diagnostic_sources(self,*args,**kwargs):
        raise AssertionError("R must not construct diagnostic source minibatches")

    def step(self,state):
        self._priority_baseline_key=state.key
        self.priority_assignment=None
        try:return super().step(state)
        finally:del self._priority_baseline_key

    def _ec_rollout(self,agent_state,replay_buffer_state,key):
        if not hasattr(self,"_priority_baseline_key") or float(self.config.alpha)==0:
            return super()._ec_rollout(agent_state,replay_buffer_state,key)
        if getattr(self,"_budget_audit",None) is not None:self._observed_ec_rollouts+=1
        g=int(self._provenance_generation);pop=int(self.config.pop_size)
        can_sample=bool(self.replay_buffer.can_sample(replay_buffer_state))
        start_index=replay_buffer_state.current_index
        # Same jitted training collector and baseline data insertion; no raw
        # diagnostic batch extraction and no additional environment interaction.
        metrics,trajectory,after=self._baseline_ec_rollout(agent_state,replay_buffer_state,key)
        ids=selected_source_ids(self._priority_baseline_key,g,pop,
            int(self.config.diagnostic_interval),int(self.config.warmup_iters))
        episode_batch=metrics.episode_lengths.shape[-1]
        flat_size=trajectory.actions.shape[0]
        time_size=flat_size//(pop*episode_batch)
        lengths=np.asarray(metrics.episode_lengths)
        valid_count=lengths.sum(axis=-1)
        valid=np.isfinite(np.asarray(metrics.episode_returns)).all(-1)
        mask=jnp.arange(time_size)[:,None,None]<metrics.episode_lengths[None,:,:]
        # Validate real finite payloads, without drawing a source minibatch.
        for leaf in (trajectory.obs,trajectory.actions,trajectory.rewards):
            finite=jnp.isfinite(leaf).reshape((time_size,pop,episode_batch,-1)).all(-1)
            valid &= np.asarray(jnp.all(~mask|finite,axis=(0,2)))
        valid=valid[ids] & (valid_count[ids]>=int(self.config.source_batch_size)) & can_sample
        started=time.perf_counter()
        u=self._active_utility_replay_state
        version=-1
        if np.any(valid):
            version=int(u.score_table.current_version)+1
            update=synthetic_update(g,ids,valid,version,pop)
            result=apply_checkpoint_score_update(u.score_table,update)
            raise_for_score_table_status(result.status)
            u=u._replace(score_table=result.table)
            self._active_utility_replay_state=u
        elapsed=time.perf_counter()-started
        provenance=make_ea_rollout_provenance(generation=g,population_size=pop,
            time_size=time_size,episode_batch=episode_batch,score_version=version)
        after=self.replay_buffer.add_provenance(after,provenance,mask.reshape(-1),current_index=start_index)
        self.priority_assignment=dict(generation=g,priority_kind=PRIORITY_KIND,
            selected_individuals=ids.tolist(),valid=valid.tolist(),valid_transition_counts=valid_count[ids].tolist(),
            new_priorities=int(valid.sum()),version=version,priority_seconds=elapsed,
            real_utility_evaluations=0,virtual_critic_updates=0,reference_batches=0,source_minibatches=0)
        audit=self._latest_score_lifecycle_audit
        audit["ordering"].extend(["synthetic_priority_update","ea_insertion_complete"])
        audit["score_version"]=version;audit["score_update_seconds"]=elapsed
        audit["synthetic_priority_kind"]=PRIORITY_KIND
        return metrics,trajectory,after
