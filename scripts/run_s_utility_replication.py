"""Independent seed3083 A/S/P CP150 replication; same algorithm, measured timing."""
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
from scripts.run_budgeted_utility_pilot import PilotObserver,ObservedDistributed,validate_shared_parent,write
from scripts.run_s_utility_extension import permute_cohort_ranks,canonical_score_rows,create_bootstrap_checkpoint,find_resume_checkpoint
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training

SEED=3083
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1")
OLD=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-cp150-v1")

def replication_config(template,manifest_path,output_dir):
    c=OmegaConf.create(OmegaConf.to_container(template,resolve=True))
    c.seed=SEED;c.arm_manifest_path=str(manifest_path);c.output_dir=str(output_dir)
    return c

def first_threshold(rows,threshold=350.):
    for row in sorted(rows,key=lambda r:r["generation"]):
        if row["return_mean"]>=threshold:return row
    return None

class InitTimingMixin:
    def init(self,key):
        started=time.perf_counter()
        state=super().init(key)
        jax.block_until_ready(state)
        self.measured_warmup_init_seconds=time.perf_counter()-started
        return state

class TimedUniform(InitTimingMixin,AuditedD0):pass
class TimedDistributed(InitTimingMixin,ObservedDistributed):pass

def journal_rows(directory,name):
    paths=sorted((Path(directory)/"segments").glob("*/"+name),key=lambda p:p.stat().st_mtime)
    return [json.loads(line) for p in paths for line in p.read_text().splitlines()]

class ReplicationObserver(PilotObserver):
    def __init__(self,label,parent,plan):
        super().__init__("S" if label=="P" else label,"seed3083",plan,parent)
        self.actual_label=label
        self.workflow_type=TimedUniform if label=="A" else TimedDistributed
        self.identity=dict(algorithm=label,seed=SEED,kind="independent-cp150",
            plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        self.start_checkpoint=Path(parent) if parent else None
        self.prefix_step_seconds=0.;self.run_started=None
    def configure(self,workflow,segment):
        self.segment=segment;self.workflow=workflow
        if int(workflow.config.seed)!=SEED:raise ValueError("independent seed mismatch")
        if self.actual_label=="P":
            def transform(table):
                started=time.perf_counter()
                result=permute_cohort_ranks(table,SEED,workflow.permutation)
                workflow.permutation_seconds+=time.perf_counter()-started
                return result
            workflow._ablation_rank_transform=transform
    def fork_state(self,workflow,run_id):
        parent=Path(self.parent)
        c=OmegaConf.load(parent.parents[3]/"resolved_config.yaml")
        validate_shared_parent(c,workflow.config)
        if int(c.seed)!=SEED:raise ValueError("warmup belongs to another seed")
        with PilotCheckpointStore(parent.parent,c,read_only=True) as store:
            state,runtime=store.restore(10,restore_template(workflow))
        if runtime["evaluations"] or runtime["budget"]["steps"]:
            raise ValueError("shared parent already trained")
        runtime=copy.deepcopy(runtime);runtime.update(run_id=run_id,target_generation=150)
        create_bootstrap_checkpoint(self.segment,workflow.config,state,runtime)
        return state,runtime
    def before_generation(self,state):
        if self.rows:return
        h=initialization_fingerprint(state)
        if self.start_checkpoint:
            expected=json.loads((self.start_checkpoint/"runtime/metadata").read_text())["state_sha256"]
            if h!=expected:raise ValueError("complete native state changed before training")
        if self.actual_label=="A" and not (ROOT/"shared_warmup.json").exists():
            write(ROOT/"shared_warmup.json",dict(seed=SEED,full_state_sha256=h,
                checkpoint=str(self.segment/"checkpoints/10"),
                measured_init_seconds=self.workflow.measured_warmup_init_seconds,
                counters=dict(generation=int(state.metrics.iterations),timesteps=int(state.metrics.sampled_timesteps))))
        warmup=json.loads((ROOT/"shared_warmup.json").read_text())
        if int(state.metrics.iterations)==10 and h!=warmup["full_state_sha256"]:
            raise ValueError("arms do not share the same complete warmup")
        self.shared_init_seconds=float(warmup["measured_init_seconds"])
    def record(self,w,state,checkpoint,runtime):
        evidence=copy.deepcopy(w.permutation) if self.actual_label=="P" else None
        super().record(w,state,checkpoint,runtime)
        if evidence is not None:
            if not evidence.get("preserved_rank_multisets"):raise ValueError("P transform missing")
            with (self.segment/"permutation.jsonl").open("a") as f:
                f.write(json.dumps(dict(generation=int(state.metrics.iterations),seconds=w.permutation_seconds,**evidence))+"\n")
        g=int(state.metrics.iterations)
        if runtime["evaluations"] and runtime["evaluations"][-1]["generation"]==g:
            ev=runtime["evaluations"][-1]
            elapsed=self.shared_init_seconds+self.prefix_step_seconds+sum(r["step_seconds"] for r in self.rows)
            timing=dict(generation=g,training_steps=int(state.metrics.sampled_timesteps),
                return_mean=float(ev["return_mean"]),training_elapsed_seconds=elapsed,
                cumulative_new_step_seconds=self.prefix_step_seconds+sum(r["step_seconds"] for r in self.rows),
                shared_warmup_init_seconds=self.shared_init_seconds,
                process_elapsed_at_evaluation_seconds=time.perf_counter()-self.run_started,
                evaluation_seconds=w.eval_seconds)
            with (self.segment/"evaluation_timing.jsonl").open("a") as f:f.write(json.dumps(timing)+"\n")

def prepare():
    protocol=ROOT/"protocol";protocol.mkdir(exist_ok=False)
    plan=dict(seed=SEED,arms=["A","S","P"],final_generation=150,starts=dict(A=10,S=10,P=10),
        primary="Return AUC g20..150; training steps650000..4940000",
        threshold_return=350.,threshold_rule="first scheduled evaluation mean>=350; no imputation if not reached",
        timing="measured shared init plus synchronized step wall seconds; excludes evaluation/checkpoint I/O",
        secondary=["first350 training steps and measured training time","S/P AUC70..150","final Return","all training costs"],
        evaluated_generations=list(range(20,151,10)),eval_episodes=128,
        score_budget_per10=32,total_scores_S_P=448,total_episodes=4940,
        primary_clear_S_P_margin=.10,near_S_P_absolute_margin=.10,
        route_rule="A:both beat Uniform and S/P AUC>=10%; B:both beat Uniform and absolute S/P gap<10%; C:neither beats Uniform; otherwise mixed",
        cutoffs_are_descriptive=True,no_interim_tuning=True,shared_warmup="new complete A CP10, used by S/P",
        physical_training_steps=14180000,physical_evaluation_steps=5376000,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mechanism_digest=mechanism_digest())
    for name in ("uniform","fitness","td","utility"):
        old=OmegaConf.load(OLD/"protocol"/f"arm_{name}.yaml")
        c=replication_config(old,protocol/"arm_manifest.json",ROOT/"runs"/name)
        OmegaConf.save(c,protocol/f"arm_{name}.yaml",resolve=True)
    configs=[OmegaConf.load(protocol/f"arm_{n}.yaml") for n in ("uniform","fitness","td","utility")]
    prepare_arm_manifest(configs,protocol/"arm_manifest.json")
    write(ROOT/"preregistration.json",plan)
    return plan

def run(plan):
    results={}
    for label in plan["arms"]:
        output=ROOT/"runs"/label;saved=ROOT/f"summary-{label}.json"
        if saved.exists():results[label]=json.loads(saved.read_text());continue
        c=OmegaConf.load(ROOT/"protocol"/("arm_uniform.yaml" if label=="A" else "arm_utility.yaml"))
        parent=None if label=="A" else Path(json.loads((ROOT/"shared_warmup.json").read_text())["checkpoint"])
        observer=ReplicationObserver(label,parent,plan)
        resume=find_resume_checkpoint(output) if output.exists() else None
        if output.exists() and resume is None:raise ValueError("existing run has no sealed recovery checkpoint")
        if resume:
            observer.start_checkpoint=resume
            old_rows=canonical_score_rows(journal_rows(output,"schedule.jsonl"))
            observer.prefix_step_seconds=sum(r["step_seconds"] for r in old_rows if r["generation"]<=int(resume.name))
        observer.run_started=time.perf_counter()
        state,report=run_training(c,output,checkpoint_interval=50,keep=4,
            ablation=observer,fork=parent is not None and resume is None,resume_checkpoint=resume)
        wall=time.perf_counter()-observer.run_started
        physical_rows=journal_rows(output,"schedule.jsonl")
        observer.rows=canonical_score_rows(physical_rows)
        timings={r["generation"]:r for r in journal_rows(output,"evaluation_timing.jsonl")}
        ordered=[timings[g] for g in sorted(timings)]
        if sorted(timings)!=list(range(20,151,10)):raise ValueError("timing evaluation schedule incomplete")
        mechanism=observer.summary()
        warmup=json.loads((ROOT/"shared_warmup.json").read_text())
        result=dict(performance=report,mechanism=mechanism,wall_seconds=wall,
            real_scores_total=mechanism["total_real_scores"],
            shared_warmup_init_seconds=warmup["measured_init_seconds"],
            measured_total_training_seconds=warmup["measured_init_seconds"]+mechanism["step_seconds"],
            first350=first_threshold(ordered),evaluation_timings=ordered,
            discarded_attempt_score_calls=sum(r["new_scores"] for r in physical_rows)-mechanism["total_real_scores"],
            recorded_physical_step_seconds=sum(r["step_seconds"] for r in physical_rows),
            final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist(),
            final_intervention_prng=np.asarray(state.utility_replay_state.intervention_key).tolist())
        if result["real_scores_total"]!=(0 if label=="A" else 448) or report["final"]["generation"]!=150:
            raise ValueError("final budget mismatch")
        if label!="A":
            for field in ("initial","final","auc_domain"):
                if report[field]!=results["A"]["performance"][field]:raise ValueError("unequal budget:"+field)
            for field in ("final_prng","final_ec_prng"):
                if result[field]!=results["A"][field]:raise ValueError("baseline PRNG mismatch")
        write(saved,result);results[label]=result
        print("COMPLETE",label,"CP150 AUC",report["auc"],"first350",result["first350"],"wall",wall,flush=True)
        del state,observer
        jax.clear_caches();gc.collect()
    write(ROOT/"summary.json",results)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if jax.default_backend()!="gpu" or subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("clean committed GPU version required")
    plan=json.loads((ROOT/"preregistration.json").read_text()) if (ROOT/"preregistration.json").exists() else prepare()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan["script_sha256"] or mechanism_digest()!=plan["mechanism_digest"]:
        raise ValueError("preregistered code changed")
    run(plan)
if __name__=="__main__":main()
