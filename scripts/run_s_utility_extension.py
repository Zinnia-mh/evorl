"""Fixed S ranking placebo and native CP50->150 continuation; no algorithm redesign."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scripts.run_budgeted_utility_pilot import PilotObserver,ObservedDistributed,write
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training

OLD=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-v2/pilot")
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-cp150-v1")
SEED=3079
PERMUTATION_TAG=0x53504350

def permute_cohort_ranks(table,seed,evidence=None):
    """Stable independent uniform permutation within each eligible cohort."""
    eligible,tokens,cohorts,ranks=map(np.asarray,jax.device_get((
        table.weighted_eligible,table.source_token,table.measured_generation,
        table.rank_normalized_score)))
    changed=ranks.copy();mapping=[]
    for cohort in np.unique(cohorts[eligible]):
        slots=np.flatnonzero(eligible&(cohorts==cohort))
        slots=slots[np.argsort(tokens[slots])]
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [PERMUTATION_TAG,int(seed),int(cohort)])))
        donors=rng.permutation(slots)
        changed[slots]=ranks[donors]
        if not np.array_equal(np.sort(changed[slots]),np.sort(ranks[slots])):
            raise ValueError("cohort rank multiset changed")
        mapping.extend([[int(tokens[a]),int(tokens[b])] for a,b in zip(slots,donors)])
    if evidence is not None:
        evidence.update(eligible_sources=int(eligible.sum()),
            changed_ranks=int(np.count_nonzero(changed[eligible]!=ranks[eligible])),
            preserved_rank_multisets=True,mapping=mapping)
    return table._replace(rank_normalized_score=jnp.asarray(changed,dtype=table.rank_normalized_score.dtype))

def validate_extension_config(parent,child):
    excluded={"arm","score_kind","alpha","diagnostic_enabled","total_episodes","output_dir","arm_manifest_path"}
    a,b=({k:v for k,v in OmegaConf.to_container(c,resolve=True).items() if k not in excluded}
         for c in (parent,child))
    if a!=b or int(child.total_episodes)<int(parent.total_episodes):
        raise ValueError("continuation contract changed beyond declared budget/arm controls")

def canonical_score_rows(rows):
    """Latest accepted attempt wins; rollback generations count once."""
    by_generation={}
    for row in rows:
        g=int(row["generation"])
        if g in by_generation and by_generation[g]["new_scores"]!=row["new_scores"]:
            raise ValueError("inconsistent score budget in replayed generation")
        by_generation[g]=row
    return [by_generation[g] for g in sorted(by_generation)]

def create_bootstrap_checkpoint(segment,config,state,runtime):
    """Make the complete fork recoverable before its first new training step."""
    with PilotCheckpointStore(Path(segment)/"checkpoints",config,interval=50,keep=3) as store:
        return store.save(state,runtime,force=True)

def find_resume_checkpoint(output):
    candidates=[p for p in (Path(output)/"segments").glob("*/checkpoints/*")
        if p.is_dir() and p.name.isdigit() and (p/"runtime/metadata").exists()
        and (p.parent.parent/"checkpoint_checksums"/(p.name+".json")).exists()]
    return max(candidates,key=lambda p:(int(p.name),p.stat().st_mtime)) if candidates else None

def parent_path(label):
    completion=json.loads((OLD/"runs"/("A" if label=="P" else label)/"completion.json").read_text())
    return Path(completion["segment"])/"checkpoints/10" if label=="P" else Path(completion["checkpoint"])

class ExtensionObserver(PilotObserver):
    def __init__(self,label,parent,plan):
        # Reuse exactly the S workflow and budget/audit path for P.
        super().__init__("S" if label=="P" else label,"extension",plan,parent)
        self.actual_label=label
        self.identity=dict(algorithm=label,seed=SEED,kind="cp150-continuation",
            permutation_seed=SEED if label=="P" else None,
            plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        self.start_checkpoint=Path(parent) if parent is not None else None
    def configure(self,workflow,segment):
        super().configure(workflow,segment)
        if self.actual_label=="P":
            def transform(table):
                started=time.perf_counter()
                result=permute_cohort_ranks(table,SEED,workflow.permutation)
                workflow.permutation_seconds+=time.perf_counter()-started
                return result
            workflow._ablation_rank_transform=transform
    def fork_state(self,workflow,run_id):
        parent=Path(self.parent)
        parent_config=OmegaConf.load(parent.parents[3]/"resolved_config.yaml")
        validate_extension_config(parent_config,workflow.config)
        if self.actual_label!="P":
            for field in ("arm","score_kind","alpha","diagnostic_enabled"):
                if getattr(parent_config,field)!=getattr(workflow.config,field):
                    raise ValueError("A/S arm controls changed at continuation")
        expected=10 if self.actual_label=="P" else 50
        with PilotCheckpointStore(parent.parent,parent_config,read_only=True) as reader:
            state,runtime=reader.restore(expected,restore_template(workflow))
        if int(state.metrics.iterations)!=expected:
            raise ValueError("wrong parent generation")
        if self.actual_label=="P":
            if runtime["evaluations"] or runtime["budget"]["steps"]:
                raise ValueError("P parent is not the common untouched warmup")
        else:
            old=json.loads((OLD/f"summary-{self.actual_label}.json").read_text())
            if runtime["evaluations"]!=old["performance"]["evaluations"]:
                raise ValueError("original evaluation prefix differs")
        runtime=copy.deepcopy(runtime)
        runtime.update(run_id=run_id,target_generation=150)
        if getattr(self,"segment",None) is not None:
            create_bootstrap_checkpoint(self.segment,workflow.config,state,runtime)
        return state,runtime
    def before_generation(self,state):
        if not self.rows:
            expected=json.loads((self.start_checkpoint/"runtime/metadata").read_text())["state_sha256"]
            if initialization_fingerprint(state)!=expected:
                raise ValueError("exact restored state changed before training")
    def record(self,w,state,checkpoint,runtime):
        evidence=copy.deepcopy(w.permutation) if self.actual_label=="P" else None
        super().record(w,state,checkpoint,runtime)
        if evidence is not None:
            if not evidence.get("preserved_rank_multisets",False):
                raise ValueError("P rank transform did not execute")
            with (self.segment/"permutation.jsonl").open("a") as f:
                f.write(json.dumps(dict(generation=int(state.metrics.iterations),
                    seconds=w.permutation_seconds,**evidence),allow_nan=False)+"\n")

def prepare():
    protocol=ROOT/"protocol"
    protocol.mkdir(exist_ok=False)
    plan=dict(seed=SEED,arms=["A","S","P"],final_generation=150,
        starts=dict(A=50,S=50,P=10),
        primary="Return AUC evaluations20..150; training steps650000..4940000",
        secondary=["AUC50..150 (1640000..4940000)","mean last5 evaluation Returns","final Return","runtime"],
        fixed_evaluations=list(range(20,151,10)),eval_episodes=128,
        total_episodes=4940,score_budget_per10=32,score_total_S_P=448,
        permutation="independent PCG64 tag+seed3079+cohort; eligible members only; canonical scores unchanged",
        support_rule="positive S/A and S/P full and post50 AUC contrasts with compatible last5 Return; descriptive one-seed signal",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mechanism_digest=mechanism_digest(),no_interim_tuning=True,
        fresh_training_steps=11220000,fresh_evaluation_steps=4352000,
        parents={label:str(parent_path(label)) for label in ("A","S","P")},
        original_prefixes={label:json.loads((OLD/f"summary-{label}.json").read_text())["performance"]["evaluations"] for label in ("A","S")})
    configs=[]
    old_config=OmegaConf.load(OLD/"arm_utility.yaml")
    for name in ("uniform","fitness","td","utility"):
        c=OmegaConf.create(OmegaConf.to_container(old_config,resolve=True))
        c.total_episodes=4940;c.arm=name;c.score_kind=name
        c.alpha=0. if name=="uniform" else .5;c.diagnostic_enabled=name!="uniform"
        c.arm_manifest_path=str(protocol/"arm_manifest.json")
        configs.append(c)
    prepare_arm_manifest(configs,protocol/"arm_manifest.json")
    for c in configs:OmegaConf.save(c,protocol/f"arm_{c.arm}.yaml",resolve=True)
    write(ROOT/"preregistration.json",plan)
    return plan

def config_for(label):
    return OmegaConf.load(ROOT/"protocol"/("arm_uniform.yaml" if label=="A" else "arm_utility.yaml"))

def preflight(plan):
    checks={}
    for label in plan["arms"]:
        c=config_for(label);c.output_dir=str(ROOT/"preflight"/label)
        Path(c.output_dir).mkdir(parents=True,exist_ok=True)
        observer=ExtensionObserver(label,parent_path(label),plan)
        workflow=type("RestoreOnly",(observer.workflow_type,),{}).build_from_config(c,enable_jit=True)
        try:
            state,runtime=observer.fork_state(workflow,"restore-only")
            observer.before_generation(state)
            checks[label]=dict(parent=str(parent_path(label)),generation=int(state.metrics.iterations),
                full_state_sha256=initialization_fingerprint(state),
                actor_count=int(state.opt_state.actor[0].count),
                critic_count=int(state.opt_state.critic[0].count),
                replay_size=int(state.replay_buffer_state.buffer_size),
                evaluation_prefix=len(runtime["evaluations"]),native_contract_restored=True)
            print("RESTORE PASS",label,checks[label],flush=True)
            del state
        finally:workflow.close()
        jax.clear_caches();gc.collect()
    write(ROOT/"recovery_checks.json",checks)

def run(plan):
    results={}
    for label in plan["arms"]:
        output=ROOT/"runs"/label
        saved=ROOT/f"summary-{label}.json"
        if saved.exists():
            results[label]=json.loads(saved.read_text());continue
        c=config_for(label)
        observer=ExtensionObserver(label,parent_path(label),plan)
        resume=None
        if output.exists():
            resume=find_resume_checkpoint(output)
            if resume is None:raise ValueError("interrupted run has no committed checkpoint")
            observer.start_checkpoint=resume
        started=time.perf_counter()
        state,report=run_training(c,output,checkpoint_interval=50,keep=3,
            ablation=observer,fork=resume is None,resume_checkpoint=resume)
        journals=sorted((output/"segments").glob("*/schedule.jsonl"),key=lambda p:p.stat().st_mtime)
        score_rows=[json.loads(line) for p in journals for line in p.read_text().splitlines()]
        observer.rows=canonical_score_rows(score_rows)
        mechanism=observer.summary()
        old_score_count=128 if label=="S" else 0
        result=dict(performance=report,mechanism=mechanism,wall_seconds=time.perf_counter()-started,
            real_scores_total=old_score_count+mechanism["total_real_scores"],
            final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist(),
            final_intervention_prng=np.asarray(state.utility_replay_state.intervention_key).tolist())
        result["discarded_attempt_score_calls"]=sum(r["new_scores"] for r in score_rows)-mechanism["total_real_scores"]
        result["recorded_physical_step_seconds"]=sum(r["step_seconds"] for r in score_rows)
        if label in ("A","S") and report["evaluations"][:4]!=plan["original_prefixes"][label]:
            raise ValueError("frozen20..50 prefix changed")
        if report["final"]["generation"]!=150 or result["real_scores_total"]!=(0 if label=="A" else 448):
            raise ValueError("final generation/score budget mismatch")
        if label!="A":
            for field in ("initial","final","auc_domain"):
                if report[field]!=results["A"]["performance"][field]:raise ValueError("unequal total budget:"+field)
            for field in ("final_prng","final_ec_prng"):
                if result[field]!=results["A"][field]:raise ValueError("training PRNG mismatch:"+field)
        write(saved,result);results[label]=result
        print("COMPLETE",label,"CP150 AUC",report["auc"],"wall",result["wall_seconds"],flush=True)
        del state,observer
        jax.clear_caches();gc.collect()
    write(ROOT/"summary.json",results)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["preflight","run"])
    args=p.parse_args()
    if jax.default_backend()!="gpu" or subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("clean committed GPU code required")
    plan=json.loads((ROOT/"preregistration.json").read_text()) if (ROOT/"preregistration.json").exists() else prepare()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan["script_sha256"] or mechanism_digest()!=plan["mechanism_digest"]:
        raise ValueError("frozen source code changed")
    if args.action=="preflight":preflight(plan)
    else:
        checks=json.loads((ROOT/"recovery_checks.json").read_text())
        if set(checks)!=set(plan["arms"]):raise ValueError("restore preflight incomplete")
        run(plan)

if __name__=="__main__":main()
