"""D1-v1: bounded relative Utility and non-renormalized age allocation."""
import math,time
from functools import partial
import jax,jax.numpy as jnp,numpy as np
from omegaconf import OmegaConf
from .erl_utility_replay import ERLUtilityReplayWorkflow
from types import SimpleNamespace
from .source_replay import WeightedSourceDistribution,compute_source_eligibility,raise_for_score_table_status
from .d1_residency import recount_and_map,map_existing_sources
from .erl_utility_replay import ERLTrainMetric,erl_replace_td3_actor_params
from .utility_replay_checkpoint import PilotCheckpointStore

VARIANT='d1-v1'
def priority_value(utility,loss_before,scale=.01):
    if not all(math.isfinite(float(x)) for x in [utility,loss_before,scale]) or loss_before<0 or scale<=0:raise ValueError('invalid relative Utility input')
    r=float(utility)/max(abs(float(loss_before)),1e-8)
    return .5+.45*r/(abs(r)+scale)

def build_d1_distribution(table,tokens,mask,generation,*,rho=.3,half_life=15.,priority_cap=.1,source_slots=None):
    if not math.isfinite(rho) or not 0<=rho<=1 or not math.isfinite(half_life) or half_life<=0 or not math.isfinite(priority_cap) or not 0<priority_cap<=1:raise ValueError('invalid D1 parameter')
    if source_slots is None:source_slots=map_existing_sources(table,tokens,mask)
    return _build(table,tokens,mask,source_slots,jnp.asarray(generation,jnp.int32),rho=rho,half_life=half_life,priority_cap=priority_cap)

@partial(jax.jit,static_argnames=('rho','half_life','priority_cap'))
def _build(table,tokens,mask,slot,generation,*,rho,half_life,priority_cap):
    n=jnp.maximum(table.resident_count,0).astype(jnp.float32);age=generation-table.measured_generation
    eligible=(n>=4)&table.validity&(table.measured_generation>=0)&(table.version>=0)&(age>=0)&(age<=30)&jnp.isfinite(table.raw_score)&jnp.isfinite(table.rank_normalized_score)
    v=jnp.clip(table.rank_normalized_score,.05,.95)
    a=jnp.where(eligible,jnp.minimum(priority_cap,n*v/jnp.maximum(jnp.sum(jnp.where(eligible,n,0)),1))*jnp.exp2(-jnp.maximum(age,0)/half_life),0)
    mass=rho*jnp.sum(a);n_ea=jnp.sum(n,dtype=jnp.float32)
    baseline=n/jnp.maximum(n_ea,1.)
    base=SimpleNamespace(baseline_source_probability=baseline,n_ea=n_ea.astype(jnp.int32),has_resident_ea=n_ea>0,resident_source_count=jnp.sum(n>0,dtype=jnp.int32))
    p=(1-mass)*base.baseline_source_probability+rho*a
    mapped=mask&(tokens>=0)&(table.source_token[slot]==tokens)&(n[slot]>0)
    transition=jnp.where(mapped,(1-mass)/jnp.maximum(base.n_ea,1)+rho*a[slot]/jnp.maximum(n[slot],1),0)
    cdf=jnp.cumsum(transition);cdf=cdf.at[-1].set(jnp.where(base.has_resident_ea,1.,0.))
    ess=jnp.where(base.has_resident_ea,1/jnp.maximum(jnp.sum(p*p),1e-30),0.)
    # Fixed rho channel Q includes the unallocated priority's uniform backoff.
    # Do not expose normalized a/S as Q: it would conceal age renormalization.
    channel=(1-jnp.sum(a))*base.baseline_source_probability+a
    dist=WeightedSourceDistribution(baseline_source_probability=baseline,n_ea=base.n_ea,has_resident_ea=base.has_resident_ea,resident_source_count=base.resident_source_count,eligible_baseline_mass=jnp.sum(jnp.where((n>0)&table.weighted_eligible,baseline,0.)),source_probability=p,weighted_source_probability=channel,transition_probability=transition,transition_cdf=cdf,eligible_source_count=jnp.sum(eligible),effective_source_count=ess,ess_fraction=ess/jnp.maximum(base.resident_source_count,1),max_source_probability=jnp.max(p),uniform_floor_per_transition=jnp.where(base.has_resident_ea,(1-mass)/jnp.maximum(base.n_ea,1),0),uniform_component_total_mass=1-mass)
    return dist,mass

def validate_d1_config(c):
    if getattr(c,'utility_variant',None)!=VARIANT:raise ValueError('D1 checkpoint/config variant required; no D0 migration')
    if not all(math.isfinite(float(getattr(c,k))) for k in ['rho','half_life','priority_cap','utility_scale']):raise ValueError('nonfinite D1 configuration')
    if not 0<=float(c.rho)<=1 or float(c.half_life)<=0 or not 0<float(c.priority_cap)<=1 or float(c.utility_scale)<=0:raise ValueError('invalid D1 configuration')
    if int(c.freshness_window)!=30 or int(c.min_resident)!=4 or int(c.diagnostic_interval)!=1:raise ValueError('D1 requires W30, M_min4, per-generation diagnostics')

class D1CheckpointStore(PilotCheckpointStore):
    def __init__(self,directory,config,**kwargs):
        validate_d1_config(config);super().__init__(directory,config,**kwargs)

class ERLUtilityReplayD1Workflow(ERLUtilityReplayWorkflow):
    @classmethod
    def name(cls):return 'ERL-Utility-Replay-D1'
    @classmethod
    def _build_from_config(cls,config):
        validate_d1_config(config)
        c=OmegaConf.create(OmegaConf.to_container(config,resolve=True));c.alpha=float(c.rho)
        if c.rho==0:c.arm='uniform';c.score_kind='uniform';c.diagnostic_enabled=False
        elif c.arm!='utility' or c.score_kind!='utility' or not c.diagnostic_enabled:raise ValueError('active D1 requires actual Utility diagnostics')
        return super()._build_from_config(c)

    def _process_pre_exposure_ec_trajectory(self,raw_trajectory,replay_buffer_state,diagnostic,*,return_trajectory=False):
        # Parent executes the unchanged real virtual update before insertion.
        result=super()._process_pre_exposure_ec_trajectory(raw_trajectory,replay_buffer_state,diagnostic,return_trajectory=return_trajectory)
        records=result[0];u=self._active_utility_replay_state;t=u.score_table
        ids=np.asarray(jax.device_get(t.source_token));slots={int(x):i for i,x in enumerate(ids) if x>=0};values=np.array(jax.device_get(t.rank_normalized_score),copy=True);valid=np.array(jax.device_get(t.validity),copy=True)
        for r in records:
            token=int(r.generation)*int(self.config.pop_size)+int(r.individual_id);i=slots.get(token)
            if i is None:continue
            if r.eligible and r.utility is not None and r.loss_before is not None and math.isfinite(r.utility) and math.isfinite(r.loss_before) and r.loss_before>=0:values[i]=priority_value(r.utility,r.loss_before,float(self.config.utility_scale))
            else:valid[i]=False
        self._active_utility_replay_state=u._replace(score_table=t._replace(rank_normalized_score=jnp.asarray(values),validity=jnp.asarray(valid)))
        return result

    def _refresh_d1_distribution(self,u,replay,*,current_generation,freshness_window,min_resident,alpha,replay_temperature,timing_sink):
        start=time.perf_counter();mask=(jnp.arange(replay.source_token.size)<replay.buffer_size)&(replay.origin_type==0)
        recounted,slots=recount_and_map(u.score_table,replay.source_token,mask);raise_for_score_table_status(recounted.status);jax.block_until_ready((recounted,slots))
        timing_sink['resident_count_seconds']=time.perf_counter()-start
        start=time.perf_counter();eligible=compute_source_eligibility(recounted.table,current_generation,freshness_window,min_resident);jax.block_until_ready(eligible);timing_sink['eligibility_seconds']=time.perf_counter()-start
        start=time.perf_counter();dist,mass=build_d1_distribution(eligible.table,recounted.resident_source_tokens,mask,current_generation,rho=float(self.config.rho),half_life=float(self.config.half_life),priority_cap=float(self.config.priority_cap),source_slots=slots);jax.block_until_ready((dist,mass))
        timing_sink['distribution_refresh_seconds']=time.perf_counter()-start;timing_sink['discarded_d0_refresh_seconds']=0.;timing_sink['distribution_build_count']=1
        self._latest_d1_mass=mass
        return u._replace(score_table=eligible.table,distribution=dist,distribution_prepared=mass>0,distribution_generation=jnp.int32(current_generation),distribution_version=eligible.table.current_version)

    def _weighted_rl_update(self,agent_state,opt_state,replay_buffer_state,key,num_updates,utility_replay_state):
        g=int(self._provenance_generation);u=utility_replay_state;dist=u.distribution
        self._latest_d1_inputs=(replay_buffer_state,key,num_updates,u)
        t=u.score_table;counts,eligible,measured=jax.device_get((t.resident_count,t.weighted_eligible,t.measured_generation));p,b=jax.device_get((dist.source_probability,dist.baseline_source_probability));ages=g-np.asarray(measured)[np.asarray(eligible)]
        mass=getattr(self,'_latest_d1_mass',1-dist.uniform_component_total_mass)
        self._latest_d1_metrics=dict(generation=g,eligible_count=int(dist.eligible_source_count),score_coverage=float(np.sum(np.asarray(counts)[np.asarray(eligible)])/max(1,np.sum(counts))),priority_mass=float(mass),probability_l1=float(np.abs(p-b).sum()),ESS=float(dist.ess_fraction),score_age_distribution={str(int(a)):int(np.sum(ages==a)) for a in np.unique(ages)},d1_refresh_seconds=self._latest_score_lifecycle_audit.get('distribution_refresh_seconds',0.))
        result=super()._weighted_rl_update(agent_state,opt_state,replay_buffer_state,key,num_updates,u)
        self._active_utility_replay_state=result[-1]
        return result

    def _weighted_step(self, state):
        """Mirror Original ERL while carrying only pilot intervention state."""
        start_t = time.perf_counter()
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        utility_replay_state = state.utility_replay_state
        self._active_utility_replay_state = utility_replay_state
        iterations = state.metrics.iterations + 1
        sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)
        sampled_episodes = jnp.zeros((), dtype=jnp.uint32)
        key, ec_rollout_key, rl_rollout_key, learn_key = jax.random.split(
            state.key, num=4
        )

        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        pop_agent_state = erl_replace_td3_actor_params(
            agent_state, pop_actor_params
        )
        ec_eval_metrics, _, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )
        utility_replay_state = self._active_utility_replay_state
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(
            jnp.uint32
        )
        sampled_episodes += jnp.uint32(
            self.config.episodes_for_fitness * pop_size
        )
        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        rl_eval_metrics, _, replay_buffer_state = self._rl_rollout(
            agent_state, replay_buffer_state, rl_rollout_key
        )
        refresh_start = time.perf_counter()
        utility_replay_state = self._refresh_d1_distribution(
            utility_replay_state,
            replay_buffer_state,
            current_generation=int(jax.device_get(iterations)),
            freshness_window=int(getattr(self.config, "freshness_window", 3)),
            min_resident=int(getattr(self.config, "min_resident", 1)),
            alpha=float(self.config.alpha),
            replay_temperature=float(self.config.replay_temperature),
            timing_sink=self._latest_score_lifecycle_audit,
        )
        self._active_utility_replay_state = utility_replay_state
        self._latest_score_lifecycle_audit["resident_and_refresh_seconds"] = (
            time.perf_counter() - refresh_start
        )
        self._latest_score_lifecycle_audit["distribution_generation"] = int(
            jax.device_get(utility_replay_state.distribution_generation)
        )
        distribution = utility_replay_state.distribution
        self._latest_score_lifecycle_audit.update(
            eligible_source_count=int(
                jax.device_get(distribution.eligible_source_count)
            ),
            n_ea=int(jax.device_get(distribution.n_ea)),
            max_source_probability=float(
                jax.device_get(distribution.max_source_probability)
            ),
            ess=float(jax.device_get(distribution.effective_source_count)),
        )
        self._latest_score_lifecycle_audit["ordering"].append(
            "distribution_refresh_complete"
        )
        if self.config.rl_updates_mode == "global":
            total_timesteps = (
                state.metrics.sampled_timesteps + sampled_timesteps
            )
            num_updates = (
                jnp.ceil(total_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        elif self.config.rl_updates_mode == "iter":
            num_updates = (
                jnp.ceil(sampled_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        else:
            raise ValueError(
                f"Unknown rl_updates_mode: {self.config.rl_updates_mode}"
            )
        self._latest_score_lifecycle_audit["ordering"].append(
            "learner_sampling_start"
        )
        learner_start = time.perf_counter()
        (
            td3_metrics,
            agent_state,
            opt_state,
            utility_replay_state,
        ) = self._weighted_rl_update(
            agent_state,
            opt_state,
            replay_buffer_state,
            learn_key,
            num_updates,
            utility_replay_state,
        )
        self._latest_score_lifecycle_audit["weighted_learner_phase_seconds"] = (
            time.perf_counter() - learner_start
        )
        td3_metrics = td3_metrics.replace(
            actor_loss=td3_metrics.actor_loss / self.config.num_rl_agents,
            critic_loss=td3_metrics.critic_loss / self.config.num_rl_agents,
        )
        train_metrics = train_metrics.replace(
            num_updates_per_iter=num_updates,
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
        )
        rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
            jnp.uint32
        )
        sampled_timesteps += rl_sampled_timesteps
        sampled_episodes += jnp.uint32(
            self.config.num_rl_agents * self.config.rollout_episodes
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        if iterations % self.config.rl_injection_interval == 0:
            ec_metrics, ec_opt_state = self._ec_update_with_rl_injection(
                ec_opt_state, agent_state, fitnesses
            )
        else:
            ec_metrics, ec_opt_state = self._ec_update(
                ec_opt_state, fitnesses
            )
        train_metrics = train_metrics.replace(
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
            time_cost_per_iter=time.perf_counter() - start_t,
        )
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=(
                state.metrics.sampled_timesteps + sampled_timesteps
            ),
            sampled_episodes=(
                state.metrics.sampled_episodes + sampled_episodes
            ),
            rl_sampled_timesteps=(
                state.metrics.rl_sampled_timesteps + rl_sampled_timesteps
            ),
            iterations=iterations,
        )
        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
            opt_state=opt_state,
            utility_replay_state=utility_replay_state,
        )
        del self._active_utility_replay_state
        return train_metrics, state
