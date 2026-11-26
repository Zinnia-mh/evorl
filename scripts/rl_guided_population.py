"""Minimal online RL-seeded GA intervention; unchanged TD3/Replay kernels."""
import math,time
import jax
import jax.numpy as jnp
import numpy as np
from evorl.types import PyTreeDict
from evorl.ec.optimizers.erl_ga import ERLGAMod
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

def inject_population(pop,actor,key,*,strength=.005):
    if strength<=0:raise ValueError("positive mutation strength required")
    leaves,tree=jax.tree.flatten(actor)
    keys=jax.random.split(key,len(leaves))
    candidates=jax.tree.unflatten(tree,[
        jnp.concatenate([x+strength*jax.random.normal(k,(4,*x.shape),x.dtype),
                         x-strength*jax.random.normal(k,(4,*x.shape),x.dtype)],axis=0)
        for x,k in zip(leaves,keys)])
    return jax.tree.map(lambda p,c:p.at[-8:].set(c),pop,candidates),candidates

def initial_lineage(size):
    return [dict(roots=[],depth=-1,standard_rl=False) for _ in range(size)]

def propagate_lineage(lineage,parent_a,parent_b):
    result=[]
    for a,b in zip(np.asarray(parent_a),np.asarray(parent_b)):
        if a<0:result.append(dict(roots=[],depth=-1,standard_rl=True));continue
        ancestors=[lineage[int(a)],lineage[int(b)]]
        roots=sorted(set().union(*(set(x["roots"]) for x in ancestors)))
        depth=min((x["depth"] for x in ancestors if x["roots"]),default=-2)+1 if roots else -1
        result.append(dict(roots=roots,depth=depth,standard_rl=any(x["standard_rl"] for x in ancestors)))
    return result

class PedigreeGA(ERLGAMod):
    def _parents(self,state,fitness,external):
        order=jnp.argsort(fitness,descending=True)
        elites=order[:self.num_elites]
        n=self.pop_size-self.num_elites-(self.external_size if external else 0)
        pool=order[:-self.external_size] if external else jnp.arange(self.pop_size)
        _,key,_,_=jax.random.split(state.key,4)
        count=math.ceil(n/2)*2 if self.enable_crossover else n
        p=pool[self.select_parents(fitness[pool],count,key)]
        if self.enable_crossover:
            a=jnp.concatenate([p[::2],p[1::2]])[:n]
            b=jnp.concatenate([p[1::2],p[::2]])[:n]
        else:a=b=p
        ext=jnp.full((self.external_size if external else 0,),-1,dtype=jnp.int32)
        return PyTreeDict(parent_a=jnp.concatenate([elites,a,ext]),
                          parent_b=jnp.concatenate([elites,b,ext]))
    def tell(self,state,fitnesses):
        _,after=super().tell(state,fitnesses)
        return self._parents(state,fitnesses,False),after
    def tell_external(self,state,fitnesses):
        _,after=super().tell_external(state,fitnesses)
        return self._parents(state,fitnesses,True),after

def install_tracking_ga(workflow):
    old=workflow.ec_optimizer
    names=("pop_size","num_elites","external_size","tournament_size","weight_max_magnitude",
        "mut_strength","num_mutation_frac","super_mut_strength","super_mut_prob","reset_prob",
        "vec_relative_prob","enable_crossover","num_crossover_frac")
    workflow.ec_optimizer=PedigreeGA(**{k:getattr(old,k) for k in names})

def action_geometry(workflow,pop,actor,obs,lineage):
    if workflow.agent.normalize_obs:raise ValueError("Pilot requires fixed unnormalized observation contract")
    actions=jax.vmap(workflow.agent.actor_network.apply,in_axes=(0,None))(pop,obs)
    base=workflow.agent.actor_network.apply(actor,obs)
    pair=jnp.mean(jnp.linalg.norm(actions[:,None]-actions[None,:],axis=-1),axis=-1)
    per_actor=jnp.mean(jnp.linalg.norm(actions-base[None],axis=-1),axis=-1)
    matrix=np.asarray(pair);dist=np.asarray(per_actor)
    groups={"all":np.arange(len(lineage)),"M_ancestry":np.array([i for i,x in enumerate(lineage) if x["roots"]],int),
            "no_M_ancestry":np.array([i for i,x in enumerate(lineage) if not x["roots"]],int)}
    summary={}
    for label,indices in groups.items():
        sub=matrix[np.ix_(indices,indices)]
        summary[label]=dict(count=len(indices),mean_distance_to_RL=float(dist[indices].mean()) if len(indices) else None,
            mean_pairwise_distance=float(sub[np.triu_indices(len(indices),1)].mean()) if len(indices)>1 else None)
    return summary

class OnlinePopulationWorkflow(AuditedD0):
    def _ec_rollout(self,agent_state,replay_buffer_state,key):
        expected=getattr(self,"expected_population_sha",None)
        if expected is not None:
            actual=initialization_fingerprint(agent_state.params.actor_params)
            if actual!=expected:raise ValueError("collector did not receive the injected population")
            start=int(replay_buffer_state.current_index)
        metrics,trajectory,after=super()._ec_rollout(agent_state,replay_buffer_state,key)
        if expected is not None:
            size=trajectory.actions.shape[0]
            if int(jnp.sum(metrics.episode_lengths))!=size:raise ValueError("receipt expects full-length Swimmer trajectories")
            ix=(start+np.arange(size))%self.replay_buffer.capacity
            np.testing.assert_array_equal(np.asarray(after.data.actions)[ix],np.asarray(trajectory.actions))
            np.testing.assert_array_equal(np.asarray(after.data.obs)[ix],np.asarray(trajectory.obs))
            self.ec_rollout_receipt=dict(population_sha256=actual,transitions=size,
                replay_obs_actions_equal_actual_collector=True,max_copy_error=0.)
        return metrics,trajectory,after

    def step(self,state):
        started=time.perf_counter()
        g=int(state.metrics.iterations)+1
        self.expected_population_sha=None
        pop=state.ec_opt_state.pop
        actor=jax.tree.map(lambda x:x[0],state.agent_state.params.actor_params)
        before_lineage=[dict(x,roots=list(x["roots"])) for x in self.lineage]
        injection=self.online_enabled and g>=self.start_generation and (g-self.start_generation)%10==0
        pre_geometry=None;injection_metadata=None
        if injection:
            pre_geometry=action_geometry(self,pop,actor,self.fixed_obs,self.lineage)
            root=jax.random.fold_in(jax.random.PRNGKey(int(self.config.seed)),0x4d504f50)
            key=jax.random.fold_in(root,g)
            changed,candidates=inject_population(pop,actor,key,strength=self.strength)
            for a,b in zip(jax.tree.leaves(pop),jax.tree.leaves(changed)):
                np.testing.assert_array_equal(a[:24],b[:24])
            center_error=max(float(jnp.max(jnp.abs((c[:4]+c[4:])/2-a))) for c,a in zip(jax.tree.leaves(candidates),jax.tree.leaves(actor)))
            injection_metadata=dict(parent_RL_actor_sha256=initialization_fingerprint(actor),
                candidate_sha256=initialization_fingerprint(candidates),mutation_key=np.asarray(key).tolist(),
                strength=self.strength,paired_center_max_error=center_error,other24_unchanged=True)
            state=state.replace(ec_opt_state=state.ec_opt_state.replace(pop=changed))
            pop=changed
            self.expected_population_sha=initialization_fingerprint(pop)
            for j in range(8):self.lineage[24+j]=dict(roots=[g*100+j],depth=0,standard_rl=False)
        evaluated_lineage=[dict(x,roots=list(x["roots"])) for x in self.lineage]
        geometry=action_geometry(self,pop,actor,self.fixed_obs,self.lineage) if g%10==0 or injection else None
        metrics,after=super().step(state)
        info=metrics.ec_info
        self.lineage=propagate_lineage(self.lineage,info.parent_a,info.parent_b)
        replay=after.replay_buffer_state
        action_error=None
        if injection:
            origins,tokens=map(np.asarray,(replay.origin_type,replay.source_token))
            for i in range(24,32):
                ix=np.flatnonzero((origins==0)&(tokens==g*32+i))
                if len(ix)!=int(self.config.env.max_episode_steps):raise ValueError("injected real Replay source missing")
            action_error=self.ec_rollout_receipt["max_copy_error"]
            print("INJECTED REAL COLLECTOR REPLAY RECEIPT",g,self.ec_rollout_receipt,flush=True)
        selected=np.asarray(info.parent_a)
        self.evidence=dict(generation=g,injected=injection,injected_count=8 if injection else 0,
            injection_metadata=injection_metadata,collector_receipt=self.ec_rollout_receipt if injection else None,
            evaluated_lineage=evaluated_lineage,next_lineage=self.lineage,
            parent_a=np.asarray(info.parent_a).tolist(),parent_b=np.asarray(info.parent_b).tolist(),
            fitness=np.asarray(metrics.pop_episode_returns).tolist(),before_injection_geometry=pre_geometry,
            geometry=geometry,injected_replay_action_max_error=action_error,
            derived_count=sum(bool(x["roots"]) for x in evaluated_lineage),
            derived_next_count=sum(bool(x["roots"]) for x in self.lineage),
            selected_M_parent_slots=sum(bool(evaluated_lineage[int(i)]["roots"]) for i in selected if i>=0))
        self.total_step_seconds=time.perf_counter()-started
        self.learner_input=None;self.learner_trace=None
        return metrics,after
