"""Frozen M-Late: original Uniform A CP100, first intervention110, final150."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
import jax
import numpy as np
from omegaconf import OmegaConf
from scripts.run_rl_guided_population_pilot import Observer,BASE,ROOT as OLD_M_ROOT,accepted_rows
from scripts.rl_guided_population import initial_lineage
from scripts.run_budgeted_utility_pilot import validate_shared_parent,write
from scripts.run_s_utility_extension import create_bootstrap_checkpoint,find_resume_checkpoint
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training

ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-late-seed3083-v1")

def original_A_parent(summary,baseline):
    parent=Path(summary["performance"]["segment_directory"])/"checkpoints/100"
    if not parent.resolve().is_relative_to((Path(baseline)/"runs/A").resolve()):
        raise ValueError("shared parent must be original A, never old M")
    meta=json.loads((parent/"runtime/metadata").read_text())
    if meta["generation"]!=100:raise ValueError("parent must be CP100")
    return parent

def interval_auc(evaluations):
    rows=sorted((r for r in evaluations if 100<=r["generation"]<=150),key=lambda r:r["generation"])
    if not rows or rows[0]["generation"]!=100 or rows[-1]["generation"]!=150:
        raise ValueError("primary AUC must include CP100 andCP150")
    return sum(.5*(a["return_mean"]+b["return_mean"])*(b["environment_timesteps"]-a["environment_timesteps"]) for a,b in zip(rows,rows[1:]))

def source_hash():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),Path(__file__).with_name("rl_guided_population.py"),
        Path(__file__).with_name("run_rl_guided_population_pilot.py"),
        Path(__file__).with_name("diagnose_rl_guided_population.py")]}

def prepare():
    old=json.loads((OLD_M_ROOT/"preregistration.json").read_text())
    assert hashlib.sha256(Path(__file__).with_name("rl_guided_population.py").read_bytes()).hexdigest()==old["source_hash"]["rl_guided_population.py"]
    if json.loads((OLD_M_ROOT/"smoke.json").read_text())["status"]!="PASS":raise ValueError("existing online M smoke not complete")
    a=json.loads((BASE/"summary-A.json").read_text())
    parent=original_A_parent(a,BASE)
    meta=json.loads((parent/"runtime/metadata").read_text())
    prefix=next(r["training_elapsed_seconds"] for r in a["evaluation_timings"] if r["generation"]==100)
    protocol=ROOT/"protocol";protocol.mkdir(exist_ok=False)
    plan=dict(algorithm="M-Late",seed=3083,parent_base_commit="1f16fb2ea85260f8e06b192ef52864a6c00fef13",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        shared_parent=dict(checkpoint=str(parent),generation=100,full_state_sha256=meta["state_sha256"],common_prefix_training_seconds=prefix),
        start_generation=110,interval=10,replacements=8,positions=list(range(24,32)),
        pairs=4,mutation_strength=.005,final_generation=150,uniform_replay=True,
        original_periodic_single_actor_injection=True,no_new_online_utility=True,
        primary="Return AUC100..150,3290000..4940000 steps; shared CP100 endpoint included",
        primary_generations=list(range(100,151,10)),descriptive_near_band=.02,
        route_rule="AUC gain>2% positive signal; abs(gain)<=2% near; gain<-2% negative; descriptive not statistical",
        secondary=["Return at110..150","final Return","fresh8/other24 Fitness","real GA breeding","measured time"],
        auxiliary_old_M="recompute old M on the SAME100..150 interval; never compare its60..150 percent directly",
        eval_interval=10,eval_episodes=128,physical_new_training_steps=1650000,physical_new_evaluation_steps=640000,
        new_actor_updates=800000,new_critic_updates=1600000,
        utility_generations=[150],utility_phase="optional readonly post-update same-generation resident data in frozen M-Late learner;3 shared references; no new rollout",
        fixed_observations="shared A CP100; no future samples",
        pedigree="M means any M-Late ancestor; known standard_rl ancestry tracked only since CP100",
        no_interim_tuning=True,source_hash=source_hash(),mechanism_digest=mechanism_digest())
    configs=[]
    for arm in ("uniform","fitness","td","utility"):
        c=OmegaConf.load(BASE/"protocol"/f"arm_{arm}.yaml")
        c.arm_manifest_path=str(protocol/"arm_manifest.json");c.output_dir=str(ROOT/"runs"/arm)
        OmegaConf.save(c,protocol/f"arm_{arm}.yaml",resolve=True);configs.append(c)
    prepare_arm_manifest(configs,protocol/"arm_manifest.json")
    write(ROOT/"preregistration.json",plan);return plan

class LateObserver(Observer):
    def __init__(self,plan):
        super().__init__(plan)
        self.identity["algorithm"]="online-M-Late"
    def fork_state(self,w,run_id):
        c=OmegaConf.load(self.parent.parents[3]/"resolved_config.yaml")
        validate_shared_parent(c,w.config)
        if not self.parent.resolve().is_relative_to((BASE/"runs/A").resolve()):raise ValueError("not original A checkpoint")
        with PilotCheckpointStore(self.parent.parent,c,read_only=True) as store:
            state,runtime=store.restore(100,restore_template(w))
        if initialization_fingerprint(state)!=self.plan["shared_parent"]["full_state_sha256"]:raise ValueError("complete A100 state differs")
        if int(state.metrics.iterations)!=100 or int(state.metrics.sampled_timesteps)!=3290000:raise ValueError("incorrect original A counters")
        runtime=copy.deepcopy(runtime);runtime.update(run_id=run_id,target_generation=150)
        create_bootstrap_checkpoint(self.segment,w.config,state,runtime)
        row=dict(generation=100,bootstrap=True,injected=False,state_sha256=initialization_fingerprint(state),
            evaluated_lineage=initial_lineage(32),next_lineage=initial_lineage(32),step_seconds=0.,core_step_seconds=0.)
        with (self.segment/"evolution.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        return state,runtime
    def before_generation(self,state):
        if not hasattr(self.w,"fixed_obs"):
            p=ROOT/"fixed_observations.npz"
            if p.exists():self.w.fixed_obs=jax.numpy.asarray(np.load(p)["obs"])
            else:
                if int(state.metrics.iterations)!=100:raise ValueError("observations must originate at shared CP100")
                obs=self.w.replay_buffer.sample(state.replay_buffer_state,jax.random.fold_in(jax.random.PRNGKey(3083),0x4d4f4253)).obs[:128]
                np.savez_compressed(p,obs=np.asarray(obs));self.w.fixed_obs=obs

def train():
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise ValueError("clean committed version required")
    plan=json.loads((ROOT/"preregistration.json").read_text()) if (ROOT/"preregistration.json").exists() else prepare()
    if source_hash()!=plan["source_hash"] or mechanism_digest()!=plan["mechanism_digest"]:raise ValueError("frozen code changed")
    output=ROOT/"runs/M"
    if (output/"completion.json").exists():raise ValueError("already complete")
    resume=find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:raise ValueError("no sealed recovery checkpoint")
    c=OmegaConf.load(ROOT/"protocol/arm_uniform.yaml")
    observer=LateObserver(plan)
    started=time.perf_counter()
    state,report=run_training(c,output,checkpoint_interval=50,keep=3,
        ablation=observer,fork=resume is None,resume_checkpoint=resume)
    rows=accepted_rows(output);new=[r for r in rows if not r.get("bootstrap")]
    a=json.loads((BASE/"summary-A.json").read_text())
    if report["initial"]!=a["performance"]["initial"] or report["final"]!=a["performance"]["final"]:raise ValueError("A/Late budgets differ")
    if report["evaluations"][:9]!=a["performance"]["evaluations"][:9]:raise ValueError("shared A prefix through100 changed")
    if np.asarray(state.key).tolist()!=a["final_prng"] or np.asarray(state.ec_opt_state.key).tolist()!=a["final_ec_prng"]:raise ValueError("baseline PRNG stream differs")
    if [r["generation"] for r in new]!=list(range(101,151)):raise ValueError("new training schedule differs")
    if [r["generation"] for r in new if r["injected"]]!=[110,120,130,140,150]:raise ValueError("injection schedule differs")
    total=plan["shared_parent"]["common_prefix_training_seconds"];timings=[]
    for row in new:
        total+=row["step_seconds"]
        if "evaluation" in row:timings.append(dict(row["evaluation"],training_elapsed_seconds=total))
    result=dict(algorithm="M-Late",performance=report,primary_auc100_150=interval_auc(report["evaluations"]),
        total_training_seconds=total,new_step_seconds=sum(r["step_seconds"] for r in new),
        core_new_step_seconds=sum(r["core_step_seconds"] for r in new),
        process_wall_seconds=time.perf_counter()-started,evaluation_timing=timings,
        injection_events=5,candidates=40,final_M_ancestry=new[-1]["derived_next_count"],
        final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist())
    write(ROOT/"summary-M.json",result)
    print("COMPLETE M-Late primary AUC100..150",result["primary_auc100_150"],"A",interval_auc(a["performance"]["evaluations"]),
        "final",report["final_return"],"new_step_minutes",result["new_step_seconds"]/60,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["train","diagnose"],default="train");args=p.parse_args()
    if jax.default_backend()!="gpu":raise ValueError("JAX GPU required")
    if args.stage=="train":train()
    else:
        from scripts.diagnose_rl_guided_population import main as diagnose
        diagnose(ROOT=ROOT,BASE=BASE,generations=(150,))
if __name__=="__main__":main()
