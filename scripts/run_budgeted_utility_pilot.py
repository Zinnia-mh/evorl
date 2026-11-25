"""Small preregistered A / corrected-D0 / distributed-S performance Pilot."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.budgeted_utility_scoring import DistributedScoringMixin
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template

ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-v2")
HISTORY=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/replication/swimmer/seed_2047/ad300-head43221c7-v1/runs/D0/resolved_config.yaml")
SEED=3079

def write(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False)

def validate_shared_parent(parent, child):
    """Only declared arm controls may differ at an exact shared warmup fork."""
    excluded={"arm","score_kind","alpha","diagnostic_enabled","output_dir","arm_manifest_path"}
    values=[{k:v for k,v in OmegaConf.to_container(c,resolve=True).items() if k not in excluded}
            for c in (parent,child)]
    if values[0]!=values[1]:
        raise ValueError("shared warmup configuration differs beyond arm controls")

class ObservedDistributed(DistributedScoringMixin,AuditedD0):
    pass

class PilotObserver:
    audit_before_checkpoint=True
    def __init__(self,label,stage,plan,parent=None):
        self.label=label;self.stage=stage;self.plan=plan;self.parent=parent
        self.identity=dict(algorithm=label,seed=SEED,stage=stage,
            plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        self.workflow_type=ObservedDistributed if label=="S" else AuditedD0
        self.rows=[]
    def fork_state(self,workflow,run_id):
        parent=Path(self.parent).resolve()
        parent_config=OmegaConf.load(parent.parents[3]/"resolved_config.yaml")
        validate_shared_parent(parent_config,workflow.config)
        with PilotCheckpointStore(parent.parent,parent_config,read_only=True) as reader:
            state,runtime=reader.restore(10,restore_template(workflow))
        if (int(state.metrics.iterations)!=10 or runtime["evaluations"]
                or runtime["budget"]["steps"] or int(state.utility_replay_state.score_table.current_version)!=-1):
            raise ValueError("parent is not an untouched complete warmup state")
        self.parent_sha256=initialization_fingerprint(state)
        return state,dict(runtime,run_id=run_id)

    def configure(self,workflow,segment):
        self.segment=segment
        if int(workflow.config.seed)!=SEED:raise ValueError("seed mismatch")
    def before_generation(self,state):
        g=int(state.metrics.iterations)
        if not self.rows:
            core=state.replace(utility_replay_state=None)
            h=initialization_fingerprint(core)
            path=ROOT/self.stage/"initialization-A.json"
            if self.label=="A":write(path,dict(core_state_sha256=h))
            elif json.loads(path.read_text())["core_state_sha256"]!=h:
                raise ValueError("arms do not share initialization/warmup state")
    def record(self,w,state,checkpoint,runtime):
        g=int(state.metrics.iterations)
        replay,key,updates,u=w.learner_input
        if not w.baseline_key_ok:raise ValueError("training PRNG advanced differently")
        table=u.score_table
        resident=np.asarray(table.resident_count)>0
        eligible=np.asarray(table.weighted_eligible)&resident
        valid=np.asarray(table.validity)&resident
        lifecycle=getattr(w,"_latest_score_lifecycle_audit",{})
        active="diagnostic_complete" in lifecycle.get("ordering",[])
        records=tuple(getattr(w,"_latest_diagnostic_records",())) if active else ()
        scores=sum(x.eligible and x.utility is not None and np.isfinite(x.utility) for x in records)
        if any(int(x.generation)!=g for x in records):raise ValueError("stale diagnostic records")
        measured=np.asarray(table.measured_generation)
        d=u.distribution
        if self.label=="A":
            origin=np.asarray(replay.origin_type);tokens=np.asarray(replay.source_token)
            all_sources=len(np.unique(tokens[origin==0]))
            tv=0.;mass=0.
        else:
            all_sources=int(resident.sum())
            tv=float(np.abs(np.asarray(d.source_probability)-np.asarray(d.baseline_source_probability)).sum()/2)
            mass=float(d.eligible_baseline_mass)
        row=dict(generation=g,new_scores=int(scores),score_records=len(records),
            resident_sources=all_sources,valid_sources=int(valid.sum()),eligible_sources=int(eligible.sum()),
            eligible_coverage=float(eligible.sum()/max(1,all_sources)),
            mean_eligible_age=float((g-measured[eligible]).mean()) if eligible.any() else None,
            max_eligible_age=int((g-measured[eligible]).max()) if eligible.any() else None,
            ea_conditional_tv=tv,eligible_probability_mass=mass,
            diagnostic_seconds=float(lifecycle.get("diagnostic_seconds",0.)),
            learner_seconds=float(lifecycle.get("weighted_learner_phase_seconds",0.)),
            step_seconds=w.step_seconds,evaluation_seconds=getattr(w,"eval_seconds",0.) if g%10==0 else 0.,
            timesteps=int(state.metrics.sampled_timesteps),actor_count=int(state.opt_state.actor[0].count),
            critic_count=int(state.opt_state.critic[0].count),baseline_prng_ok=w.baseline_key_ok)
        expected=0 if self.label=="A" else (
            int(w.config.pop_size) if self.label=="D0" and g%10==0 else (
            ((g-10)*int(w.config.pop_size)//10-(g-11)*int(w.config.pop_size)//10) if self.label=="S" else 0))
        if scores!=expected:raise ValueError(f"score budget mismatch at{g}: {scores}!={expected}")
        if self.label!="A" and g in (20,self.plan["final_generation"]):
            trace=w.learner_trace
            base,final=map(np.asarray,trace[:2])
            origin=np.asarray(replay.origin_type)
            if base.shape!=(updates,int(w.config.actor_update_interval),int(w.config.batch_size)):
                raise ValueError("actual TD3 input trace shape mismatch")
            if np.any(final<0) or np.any(final>=int(replay.buffer_size)):
                raise ValueError("invalid actual Critic input index")
            ea=origin[base]==0;rl=origin[base]==1
            if np.any(final[rl]!=base[rl]) or np.any(origin[final]!=origin[base]):
                raise ValueError("RL slots or EA/RL proportion changed")
            if not np.any((final!=base)&ea):raise ValueError("weighted actual TD3 inputs inactive")
            choose=np.linspace(0,len(base)-1,min(8,len(base)),dtype=int)
            np.savez_compressed(self.segment/f"critic-input-{g}.npz",
                baseline=base[choose],final=final[choose],origin=origin,
                source_token=np.asarray(replay.source_token),sampled_update_positions=choose)
            row.update(real_critic_input_audit=True,rl_changed=0,
                changed_ea=int(((final!=base)&ea).sum()),actual_critic_batches=int(np.prod(base.shape[:-1])))
        with (self.segment/"schedule.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        self.rows.append(row)
        w.learner_input=None;w.learner_trace=None
    def summary(self):
        return dict(total_real_scores=sum(r["new_scores"] for r in self.rows),
            eligible_coverage_mean=float(np.mean([r["eligible_coverage"] for r in self.rows])),
            eligible_coverage_final=self.rows[-1]["eligible_coverage"],
            mean_score_age=_mean([r["mean_eligible_age"] for r in self.rows]),
            ea_conditional_tv_mean=float(np.mean([r["ea_conditional_tv"] for r in self.rows])),
            diagnostic_seconds=sum(r["diagnostic_seconds"] for r in self.rows),
            learner_seconds=sum(r["learner_seconds"] for r in self.rows),
            step_seconds=sum(r["step_seconds"] for r in self.rows))

def _mean(a):
    a=[x for x in a if x is not None]
    return float(np.mean(a)) if a else None

def prepare(stage,parent=None):
    directory=ROOT/stage
    directory.mkdir(parents=True,exist_ok=False)
    small=stage=="smoke"
    final=20 if small else 50
    plan=dict(seed=SEED,final_generation=final,warmup_generations=10,
        shared_warmup_parent=str(parent) if parent else 'fresh A checkpoint10, shared exactly by D0 and S',
        evaluated_generations=list(range(20,final+1,10)),
        primary="Return AUC on common environment steps at evaluations g20..50",
        secondary=["early AUC g20..30","final Return","mean final3 evaluations","coverage/age","runtime"],
        arms=["A","D0","S"],score_budget_per_10_generations=32,alpha=.5,temperature=.5,
        schedule="floor(t*32/10)-floor((t-1)*32/10), t=g-10",
        expand_rule="AUC improves over A and corrected D0 without material final Return regression; descriptive single seed",
        no_early_return_selection=True,
        seed_search="exact seed3079 absent in prior manifest/config registry; hex-name matches ignored",
        base="404a01b3d796967656d8b9309d3c460a4c24301d",
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    if small:
        plan["primary"]="functional Smoke only; no performance inference"
    write(directory/"preregistration.json",plan)
    configs=[]
    for name in ("uniform","fitness","td","utility"):
        c=OmegaConf.load(HISTORY)
        c.seed=SEED;c.arm=name;c.score_kind=name;c.alpha=0. if name=="uniform" else .5
        c.diagnostic_enabled=name!="uniform";c.recorders=["log"]
        c.total_episodes=320+(final-10)*33
        c.eval_interval=10;c.eval_episodes=2 if small else 128;c.num_eval_envs=c.eval_episodes
        c.arm_manifest_path=str(directory/"arm_manifest.json")
        if small:
            c.env.max_episode_steps=8;c.source_batch_size=4;c.batch_size=8
            c.replay_buffer_capacity=4096
            c.agent_network.actor_hidden_layer_sizes=[8,8]
            c.agent_network.critic_hidden_layer_sizes=[8,8]
        configs.append(c)
    prepare_arm_manifest(configs,directory/"arm_manifest.json")
    for c in configs:OmegaConf.save(c,directory/f"arm_{c.arm}.yaml",resolve=True)
    write(directory/"configuration_checksums.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir() if p.is_file()})
    return plan

def run(stage,plan,parent=None):
    directory=ROOT/stage
    summaries={}
    for label in plan["arms"]:
        config=OmegaConf.load(directory/("arm_uniform.yaml" if label=="A" else "arm_utility.yaml"))
        if label!="A" and parent is None:
            parent=next((directory/"runs/A/segments").glob("*/checkpoints/10"))
        observer=PilotObserver(label,stage,plan,parent=parent)
        started=time.perf_counter()
        state,report=run_training(config,directory/"runs"/label,checkpoint_interval=10000,keep=2,ablation=observer,fork=parent is not None)
        elapsed=time.perf_counter()-started
        result=dict(wall_seconds=elapsed,performance=report,mechanism=observer.summary(),
            final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist())
        write(directory/f"summary-{label}.json",result)
        summaries[label]=result
        if label!="A":
            a=summaries["A"]
            for field in ("initial","final","auc_domain"):
                if report[field]!=a["performance"][field]:raise ValueError("matched budget failed:"+field)
            if result["final_prng"]!=a["final_prng"]:raise ValueError("baseline PRNG stream mismatch")
            if result["final_ec_prng"]!=a["final_ec_prng"]:raise ValueError("EA PRNG stream mismatch")
        if label=="S" and result["mechanism"]["total_real_scores"]!=summaries["D0"]["mechanism"]["total_real_scores"]:
            raise ValueError("distributed/burst total score budget differs")
        print("COMPLETE",stage,label,"wall",elapsed,"AUC",report["auc"],flush=True)
        del state,observer
        jax.clear_caches();gc.collect()
    write(directory/"summary.json",summaries)
    return summaries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["smoke","pilot"],required=True)
    p.add_argument("--parent",type=Path)
    args=p.parse_args()
    if jax.default_backend()!="gpu" or subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("clean committed GPU version required")
    if args.stage=="pilot":
        smoke=json.loads((ROOT/"smoke/summary.json").read_text())
        if smoke["S"]["mechanism"]["total_real_scores"]!=32:raise ValueError("Smoke incomplete")
    plan=prepare(args.stage,args.parent)
    run(args.stage,plan,args.parent)

if __name__=="__main__":main()
