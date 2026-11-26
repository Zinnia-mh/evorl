"""Read-only same-generation A/M experience scoring in the frozen M learner."""
import gc,hashlib,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scripts.run_rl_guided_population_pilot import ROOT,BASE,accepted_rows
from scripts.rl_guided_population import action_geometry,initial_lineage
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.learner_utility import td3_critic_loss,virtual_critic_utility

def checkpoint_segment(run,generation):
    matches=sorted(Path(run).glob(f"segments/*/checkpoints/{generation}"),key=lambda p:p.stat().st_mtime)
    sealed=[p for p in matches if (p.parent.parent/f"checkpoint_checksums/{generation}.json").exists()]
    if not sealed:raise FileNotFoundError(f"no sealed CP{generation}")
    return sealed[-1].parent.parent

def load_checkpoint(segment,generation,config):
    w=type("ReadonlyContext",(AuditedD0,),{}).build_from_config(config,enable_jit=True)
    with PilotCheckpointStore(segment/"checkpoints",config,read_only=True) as reader:
        state,runtime=reader.restore(generation,restore_template(w))
    return w,state,runtime

def pick_batch(replay,indices,key,count=256):
    if len(indices)<count:raise ValueError("insufficient real resident source")
    selected=jax.random.choice(key,jnp.asarray(indices),shape=(count,),replace=False)
    return jax.tree.map(lambda x:x[selected],replay.data)

def main(ROOT=ROOT,BASE=BASE,generations=(50,100,150)):
    if jax.default_backend()!="gpu":raise ValueError("GPU diagnostic required")
    if (ROOT/"utility_records.jsonl").exists():raise FileExistsError("diagnostic already started; preserve previous artifact")
    mresult=json.loads((ROOT/"summary-M.json").read_text())
    aresult=json.loads((BASE/"summary-A.json").read_text())
    msegment=Path(mresult["performance"]["segment_directory"])
    asegment=Path(aresult["performance"]["segment_directory"])
    cm=OmegaConf.load(ROOT/"runs/M/resolved_config.yaml");ca=OmegaConf.load(BASE/"runs/A/resolved_config.yaml")
    evol={r["generation"]:r for r in accepted_rows(ROOT/"runs/M")}
    obs=jnp.asarray(np.load(ROOT/"fixed_observations.npz")["obs"])
    summaries=[];metadata=[];started=time.perf_counter()
    with (ROOT/"utility_records.jsonl").open("x") as stream:
        for g in generations:
            w,m,_=load_checkpoint(checkpoint_segment(ROOT/"runs/M",g),g,cm)
            wa,a,_=load_checkpoint(checkpoint_segment(BASE/"runs/A",g),g,ca)
            h_m,h_a=initialization_fingerprint(m),initialization_fingerprint(a)
            context=w._make_pre_exposure_context(m);learner=context.learner_agent_state
            params=learner.params.critic_params;opt=context.critic_opt_state
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(3083),0x4d555449),g)
            refs=[w.replay_buffer.sample(m.replay_buffer_state,jax.random.fold_in(key,100+r)) for r in range(3)]
            ref_hash=[initialization_fingerprint(b) for b in refs]
            def loss(p,b,k):
                state=learner.replace(params=learner.params.replace(critic_params=p))
                return td3_critic_loss(w.agent,state,b,k)
            evaluate=jax.jit(lambda p,o,b,r,k:virtual_critic_utility(loss_fn=loss,optimizer=w.optimizer,
                params=p,opt_state=o,source_batch=b,reference_batch=r,key=k))
            source_loss=jax.jit(loss)
            all_sources=[]
            for arm,state in (("A",a),("M",m)):
                rb=state.replay_buffer_state
                origins,tokens=np.asarray(rb.origin_type),np.asarray(rb.source_token)
                for i in range(32):
                    ix=np.flatnonzero((origins==0)&(tokens==g*32+i))
                    if len(ix)!=1000:raise ValueError("same-generation EA provenance missing")
                    lineage=initial_lineage(32)[i] if arm=="A" else evol[g]["evaluated_lineage"][i]
                    group="Original-EA-A" if arm=="A" else ("M-direct" if lineage["roots"] and lineage["depth"]==0 else "M-descendant" if lineage["roots"] else "M-no-derived-ancestry")
                    batch=pick_batch(rb,ix,jax.random.fold_in(key,i))
                    fitness=float(jnp.sum(rb.data.rewards[ix]))
                    all_sources.append((f"{arm}-g{g}-i{i}",group,batch,fitness,len(ix),lineage))
            rb=m.replay_buffer_state
            ix=(int(rb.current_index)-1000+np.arange(1000))%w.replay_buffer.capacity
            if np.any(np.asarray(rb.origin_type)[ix]!=1):raise ValueError("RL-own final rollout provenance missing")
            all_sources.append((f"M-g{g}-RL-own","RL-own",pick_batch(rb,ix,jax.random.fold_in(key,33)),
                float(jnp.sum(rb.data.rewards[ix])),1000,None))
            all_sources.append((f"M-g{g}-Replay","Replay-random",
                w.replay_buffer.sample(rb,jax.random.fold_in(key,77)),None,None,None))
            records=[]
            for source_id,group,batch,fitness,valid,lineage in all_sources:
                actions=w.agent.actor_network.apply(learner.params.actor_params,batch.obs)
                behavior=float(jnp.mean(jnp.linalg.norm(actions-batch.actions,axis=-1)))
                td=float(source_loss(params,batch,jax.random.split(key)[0]))
                for ri,reference in enumerate(refs):
                    result=evaluate(params,opt,batch,reference,key)
                    row=dict(generation=g,source_id=source_id,source_group=group,
                        reference_id=ri,reference_fingerprint=ref_hash[ri],critic_snapshot_sha256=h_m,
                        utility=float(result.utility),loss_before=float(result.loss_ref_before),
                        loss_after=float(result.loss_ref_after),fitness=fitness,
                        td3_source_critic_loss=td,trajectory_distance_to_current_RL=behavior,
                        valid_transitions=valid,batch_size=256,lineage=lineage,step_scale=1.)
                    if not all(np.isfinite(row[k]) for k in ("utility","loss_before","loss_after","td3_source_critic_loss","trajectory_distance_to_current_RL")):
                        raise ValueError("nonfinite real Utility measurement")
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();records.append(row)
            for group in sorted({r["source_group"] for r in records}):
                groups={}
                for row in records:
                    if row["source_group"]==group:groups.setdefault(row["source_id"],[]).append(row)
                values=np.array([np.mean([r["utility"] for r in rr]) for rr in groups.values()])
                sample_rows=[rr[0] for rr in groups.values()]
                summaries.append(dict(generation=g,group=group,distinct_sources=len(groups),
                    reference_measurements=sum(len(rr) for rr in groups.values()),
                    utility_mean=float(values.mean()),utility_median=float(np.median(values)),
                    source_mean_positive_fraction=float((values>0).mean()),
                    measurement_positive_fraction=float(np.mean([r["utility"]>0 for rr in groups.values() for r in rr])),
                    fitness_mean=float(np.mean([r["fitness"] for r in sample_rows])) if sample_rows[0]["fitness"] is not None else None,
                    behavior_mean=float(np.mean([r["trajectory_distance_to_current_RL"] for r in sample_rows]))))
            actor=jax.tree.map(lambda x:x[0],m.agent_state.params.actor_params)
            actor_a=jax.tree.map(lambda x:x[0],a.agent_state.params.actor_params)
            geometry=dict(M=action_geometry(w,m.ec_opt_state.pop,actor,obs,evol[g]["next_lineage"]),
                A=action_geometry(wa,a.ec_opt_state.pop,actor_a,obs,initial_lineage(32)))
            if initialization_fingerprint(m)!=h_m or initialization_fingerprint(a)!=h_a:raise ValueError("diagnostic modified full frozen state")
            metadata.append(dict(generation=g,frozen_M_sha256=h_m,frozen_A_sha256=h_a,
                state_unchanged=True,reference_fingerprints=ref_hash,groups=summaries[-len({r["source_group"] for r in records}):],
                post_GA_population_geometry=geometry,phase="post-update checkpoint,current same-generation resident sources",
                cross_run_A_experiences_only_for_diagnostic=True,extra_environment_steps=0))
            print("DIAGNOSTIC",g,summaries[-len({r["source_group"] for r in records}):],flush=True)
            del evaluate,source_loss,w,wa,m,a,context,learner,all_sources,records,refs,params,opt
            jax.clear_caches();gc.collect()
    (ROOT/"utility_summary.json").write_text(json.dumps(dict(contexts=metadata,summaries=summaries,
        elapsed_seconds=time.perf_counter()-started,extra_environment_steps=0,
        interpretation="source units distinct from 3 reference repeats; all groups evaluated in frozen M learner"),indent=2,allow_nan=False))
if __name__=="__main__":main()
