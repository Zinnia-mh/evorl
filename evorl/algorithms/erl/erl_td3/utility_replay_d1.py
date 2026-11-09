"""D1-v1: bounded relative Utility and non-renormalized age allocation."""
import math,time
from functools import partial
import jax,jax.numpy as jnp,numpy as np
from omegaconf import OmegaConf
from .erl_utility_replay import ERLUtilityReplayWorkflow
from .source_replay import build_weighted_source_distribution
from .utility_replay_checkpoint import PilotCheckpointStore

VARIANT='d1-v1'
def priority_value(utility,loss_before,scale=.01):
    if not all(math.isfinite(float(x)) for x in [utility,loss_before,scale]) or loss_before<0 or scale<=0:raise ValueError('invalid relative Utility input')
    r=float(utility)/max(abs(float(loss_before)),1e-8)
    return .5+.45*r/(abs(r)+scale)

def build_d1_distribution(table,tokens,mask,generation,*,rho=.3,half_life=15.,priority_cap=.1):
    if not math.isfinite(rho) or not 0<=rho<=1 or not math.isfinite(half_life) or half_life<=0 or not math.isfinite(priority_cap) or not 0<priority_cap<=1:raise ValueError('invalid D1 parameter')
    return _build(table,tokens,mask,jnp.asarray(generation,jnp.int32),rho=rho,half_life=half_life,priority_cap=priority_cap)

@partial(jax.jit,static_argnames=('rho','half_life','priority_cap'))
def _build(table,tokens,mask,generation,*,rho,half_life,priority_cap):
    n=jnp.maximum(table.resident_count,0).astype(jnp.float32);age=generation-table.measured_generation
    eligible=(n>=4)&table.validity&(table.measured_generation>=0)&(table.version>=0)&(age>=0)&(age<=30)&jnp.isfinite(table.raw_score)&jnp.isfinite(table.rank_normalized_score)
    v=jnp.clip(table.rank_normalized_score,.05,.95)
    a=jnp.where(eligible,jnp.minimum(priority_cap,n*v/jnp.maximum(jnp.sum(jnp.where(eligible,n,0)),1))*jnp.exp2(-jnp.maximum(age,0)/half_life),0)
    mass=rho*jnp.sum(a);base=build_weighted_source_distribution(table,tokens,mask,alpha=0.,replay_temperature=.5)
    p=(1-mass)*base.baseline_source_probability+rho*a
    order=jnp.argsort(table.source_token);sorted_tokens=table.source_token[order];loc=jnp.clip(jnp.searchsorted(sorted_tokens,tokens),0,len(order)-1);slot=order[loc]
    mapped=mask&(tokens>=0)&(sorted_tokens[loc]==tokens)&(n[slot]>0)
    transition=jnp.where(mapped,(1-mass)/jnp.maximum(base.n_ea,1)+rho*a[slot]/jnp.maximum(n[slot],1),0)
    cdf=jnp.cumsum(transition);cdf=cdf.at[-1].set(jnp.where(base.has_resident_ea,1.,0.))
    ess=jnp.where(base.has_resident_ea,1/jnp.maximum(jnp.sum(p*p),1e-30),0.)
    # Fixed rho channel Q includes the unallocated priority's uniform backoff.
    # Do not expose normalized a/S as Q: it would conceal age renormalization.
    channel=(1-jnp.sum(a))*base.baseline_source_probability+a
    dist=base._replace(source_probability=p,weighted_source_probability=channel,transition_probability=transition,transition_cdf=cdf,eligible_source_count=jnp.sum(eligible),effective_source_count=ess,ess_fraction=ess/jnp.maximum(base.resident_source_count,1),max_source_probability=jnp.max(p),uniform_floor_per_transition=jnp.where(base.has_resident_ea,(1-mass)/jnp.maximum(base.n_ea,1),0),uniform_component_total_mass=1-mass)
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

    def _weighted_rl_update(self,agent_state,opt_state,replay_buffer_state,key,num_updates,utility_replay_state):
        start=time.perf_counter();g=int(self._provenance_generation);u=utility_replay_state
        resident=(jnp.arange(replay_buffer_state.source_token.size)<replay_buffer_state.buffer_size)&(replay_buffer_state.origin_type==0)
        dist,mass=build_d1_distribution(u.score_table,replay_buffer_state.source_token,resident,g,rho=float(self.config.rho),half_life=float(self.config.half_life),priority_cap=float(self.config.priority_cap));jax.block_until_ready((dist,mass))
        u=u._replace(distribution=dist,distribution_prepared=mass>0,distribution_generation=jnp.int32(g),distribution_version=u.score_table.current_version)
        self._latest_d1_inputs=(replay_buffer_state,key,num_updates,u)
        t=u.score_table;counts,eligible,measured=jax.device_get((t.resident_count,t.weighted_eligible,t.measured_generation));p,b=jax.device_get((dist.source_probability,dist.baseline_source_probability));ages=g-np.asarray(measured)[np.asarray(eligible)]
        self._latest_d1_metrics=dict(generation=g,eligible_count=int(dist.eligible_source_count),score_coverage=float(np.sum(np.asarray(counts)[np.asarray(eligible)])/max(1,np.sum(counts))),priority_mass=float(mass),probability_l1=float(np.abs(p-b).sum()),ESS=float(dist.ess_fraction),score_age_distribution={str(int(a)):int(np.sum(ages==a)) for a in np.unique(ages)},d1_refresh_seconds=time.perf_counter()-start)
        result=super()._weighted_rl_update(agent_state,opt_state,replay_buffer_state,key,num_updates,u)
        self._active_utility_replay_state=result[-1]
        return result
