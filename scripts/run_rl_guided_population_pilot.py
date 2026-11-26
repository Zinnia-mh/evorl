"""Fixed seed3083 Uniform A versus online M; native complete-state recovery."""
import argparse,copy,gc,hashlib,json,subprocess,time
from pathlib import Path
import jax
import numpy as np
from omegaconf import OmegaConf
from scripts.rl_guided_population import OnlinePopulationWorkflow,install_tracking_ga,initial_lineage
from scripts.run_budgeted_utility_pilot import validate_shared_parent,write
from scripts.run_s_utility_extension import create_bootstrap_checkpoint,find_resume_checkpoint
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training

ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-population-seed3083-v1")
BASE=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1")
def setup(w,enabled,start=40):
    install_tracking_ga(w);w.online_enabled=enabled;w.start_generation=start;w.strength=.005
    w.lineage=initial_lineage(32)

def source_hash():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name("rl_guided_population.py")]}

def prepare():
    protocol=ROOT/"protocol";protocol.mkdir(exist_ok=False)
    baseline=json.loads((BASE/"summary-A.json").read_text())
    checkpoint=Path(baseline["performance"]["segment_directory"])/"checkpoints/50"
    parent=dict(checkpoint=str(checkpoint),generation=50,
        full_state_sha256=json.loads((checkpoint/"runtime/metadata").read_text())["state_sha256"],
        common_prefix_training_seconds=next(r["training_elapsed_seconds"] for r in baseline["evaluation_timings"] if r["generation"]==50))
    plan=dict(seed=3083,algorithm="M",start_generation=60,interval=10,replacements=8,
        pairs=4,mutation_strength=.005,positions=list(range(24,32)),final_generation=150,
        primary="Return AUC g60..150,1970000..4940000 steps",
        secondary=["final Return","first scheduled128-episode mean>=350 steps and measured time","total cost"],
        evaluate_every=10,evaluate_episodes=128,utility_generations=[50,100,150],
        utility_phase="offline post-update current checkpoint; current resident generation sources; no future data",
        utility_context="M frozen learner/Adam/current replay references; cross-score same-generation A EA plus M EA and RL-own",
        mutation="4 independent Gaussian epsilons,paired+/-0.005; no adaptive mutation",
        standard_RL_injection_unchanged=True,uniform_replay=True,no_extra_rollouts=True,
        validation=dict(collector_parameter_binding="exact SHA",actual_collector_to_replay="bitwise obs/actions",
            GPU_float_smoke_tolerance=1e-4,GA_PRNG_discrete_provenance="exact",
            pedigree="any M ancestor; standard_rl flag only tracks known ancestry sinceCP50"),
        baseline_root=str(BASE),shared_parent=parent,physical_new_training_steps=3300000,
        physical_new_evaluation_steps=1280000,no_interim_tuning=True,source_hash=source_hash(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        mechanism_digest=mechanism_digest())
    configs=[]
    for arm in ("uniform","fitness","td","utility"):
        c=OmegaConf.load(BASE/"protocol"/f"arm_{arm}.yaml")
        c.arm_manifest_path=str(protocol/"arm_manifest.json");c.output_dir=str(ROOT/"runs"/arm)
        configs.append(c);OmegaConf.save(c,protocol/f"arm_{arm}.yaml",resolve=True)
    prepare_arm_manifest(configs,protocol/"arm_manifest.json")
    write(ROOT/"preregistration.json",plan);return plan

def accepted_rows(output):
    paths=sorted(Path(output).glob("segments/*/evolution.jsonl"),key=lambda p:p.stat().st_mtime)
    rows=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
    unique={r["generation"]:r for r in rows}
    return [unique[k] for k in sorted(unique)]

class Observer:
    workflow_type=OnlinePopulationWorkflow
    audit_before_checkpoint=True
    reconstruct_sample_evidence=False
    def __init__(self,plan):
        self.plan=plan;self.parent=Path(plan["shared_parent"]["checkpoint"])
        self.identity=dict(algorithm="online-M",seed=3083,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        self.rows=[]
    def configure(self,w,segment):
        self.w=w;self.segment=segment;setup(w,True,self.plan["start_generation"])
    def fork_state(self,w,run_id):
        c=OmegaConf.load(self.parent.parents[3]/"resolved_config.yaml")
        validate_shared_parent(c,w.config)
        with PilotCheckpointStore(self.parent.parent,c,read_only=True) as reader:
            state,runtime=reader.restore(50,restore_template(w))
        if initialization_fingerprint(state)!=self.plan["shared_parent"]["full_state_sha256"]:raise ValueError("shared state mismatch")
        runtime=copy.deepcopy(runtime);runtime.update(run_id=run_id,target_generation=150)
        create_bootstrap_checkpoint(self.segment,w.config,state,runtime)
        row=dict(generation=50,bootstrap=True,injected=False,state_sha256=initialization_fingerprint(state),
            evaluated_lineage=initial_lineage(32),next_lineage=initial_lineage(32),
            step_seconds=0.,core_step_seconds=0.)
        with (self.segment/"evolution.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        return state,runtime
    def validate_resume(self,checkpoint,state):
        if int(checkpoint.name)==self.plan["shared_parent"].get("generation",10):
            if initialization_fingerprint(state)!=self.plan["shared_parent"]["full_state_sha256"]:raise ValueError("bootstrap state mismatch")
            self.w.lineage=initial_lineage(32)
            return
        journal=checkpoint.parent.parent/"evolution.jsonl"
        rows=[json.loads(x) for x in journal.read_text().splitlines()]
        row=next(x for x in reversed(rows) if x["generation"]==int(checkpoint.name))
        if row["state_sha256"]!=initialization_fingerprint(state):raise ValueError("pedigree checkpoint mismatch")
        self.w.lineage=row["next_lineage"]
    def before_generation(self,state):
        if not hasattr(self.w,"fixed_obs"):
            path=ROOT/"fixed_observations.npz"
            if path.exists():self.w.fixed_obs=jax.numpy.asarray(np.load(path)["obs"])
            else:
                if int(state.metrics.iterations)!=50:raise ValueError("fixed observations must originate from shared CP50")
                obs=self.w.replay_buffer.sample(state.replay_buffer_state,jax.random.fold_in(jax.random.PRNGKey(3083),0x4d4f4253)).obs[:128]
                np.savez_compressed(path,obs=np.asarray(obs));self.w.fixed_obs=obs
    def record(self,w,state,checkpoint,runtime):
        g=int(state.metrics.iterations)
        if not w.baseline_key_ok:raise ValueError("baseline PRNG consumed differently")
        row=dict(w.evidence,step_seconds=w.total_step_seconds,core_step_seconds=w.step_seconds,
            timesteps=int(state.metrics.sampled_timesteps),
            actor_count=int(state.opt_state.actor[0].count),critic_count=int(state.opt_state.critic[0].count),
            baseline_prng_ok=w.baseline_key_ok)
        if row["timesteps"]!=(320+(g-10)*33)*1000 or row["actor_count"]!=(g-10)*16000 or row["critic_count"]!=(g-10)*32000:
            raise ValueError("training/update budget changed")
        if g%50==0:row["state_sha256"]=initialization_fingerprint(state)
        if runtime["evaluations"] and runtime["evaluations"][-1]["generation"]==g:
            row["evaluation"]=runtime["evaluations"][-1]
        with (self.segment/"evolution.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        self.rows.append(row)

def smoke():
    if (ROOT/"smoke.json").exists():raise FileExistsError("smoke already complete")
    c=OmegaConf.load(BASE/"protocol/arm_uniform.yaml")
    c.env.max_episode_steps=8;c.batch_size=8;c.source_batch_size=4;c.replay_buffer_capacity=4096
    c.agent_network.actor_hidden_layer_sizes=[8,8];c.agent_network.critic_hidden_layer_sizes=[8,8]
    c.num_eval_envs=2;c.eval_episodes=2
    base=type("SmokeBase",(AuditedD0,),{}).build_from_config(c,enable_jit=True)
    online=type("SmokeM",(OnlinePopulationWorkflow,),{}).build_from_config(c,enable_jit=True)
    setup(online,False,11)
    state=base.init(jax.random.PRNGKey(3083))
    online.fixed_obs=base.replay_buffer.sample(state.replay_buffer_state,jax.random.PRNGKey(53)).obs
    baseline_repeat_steps=0
    _,a=base.step(state);_,off=online.step(state)
    if initialization_fingerprint(a)!=initialization_fingerprint(off):
        def differences(left,right):
            rows=[]
            for (path,x),y in zip(jax.tree_util.tree_flatten_with_path(left)[0],jax.tree.leaves(right)):
                x,y=np.asarray(x),np.asarray(y)
                if not np.array_equal(x,y):
                    rows.append(dict(path=jax.tree_util.keystr(path),shape=list(x.shape),
                        changed=int(np.count_nonzero(x!=y)),max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float))))))
            return rows
        _,again=base.step(state);baseline_repeat_steps=1
        debug=dict(online_vs_baseline=differences(a,off),baseline_repeat=differences(a,again))
        (ROOT/"smoke_difference.json").write_text(json.dumps(debug,indent=2))
        # Distinguish unavoidable GPU float nondeterminism from changed algorithm state.
        limits={x["path"]:1e-4 for x in debug["online_vs_baseline"]}
        for row in debug["online_vs_baseline"]:
            if row["max_abs"]>limits.get(row["path"],1e-4):raise ValueError("disabled drift exceeds original GPU repeat envelope")
        for left,right in [(a.key,off.key),(a.ec_opt_state.key,off.ec_opt_state.key),
            (a.replay_buffer_state.source_token,off.replay_buffer_state.source_token),
            (a.replay_buffer_state.origin_type,off.replay_buffer_state.origin_type)]:
            np.testing.assert_array_equal(left,right)
        if initialization_fingerprint(a.ec_opt_state)!=initialization_fingerprint(off.ec_opt_state):
            raise ValueError("disabled GA state changed")
        print("DISABLED baseline floats within repeat envelope",flush=True)
    online.online_enabled=True;online.lineage=initial_lineage(32)
    _,one=online.step(state);first=copy.deepcopy(online.evidence)
    _,two=online.step(one);second=copy.deepcopy(online.evidence)
    if first["injected_count"]!=8 or second["injected"]:
        raise ValueError("injected population not used in real evolution")
    if second["derived_count"]!=first["derived_next_count"]:raise ValueError("lineage not carried to actual next generation")
    if int(two.metrics.sampled_episodes)!=386:raise ValueError("extra environment episodes")
    write(ROOT/"smoke.json",dict(status="PASS",disabled_discrete_and_GA_state_identical=True,gpu_float_absolute_tolerance=1e-4,
        first=first,second=second,final_episodes=int(two.metrics.sampled_episodes),
        physical_training_steps=3616+264*baseline_repeat_steps,actor_updates=512+128*baseline_repeat_steps,critic_updates=1024+256*baseline_repeat_steps))
    print("SMOKE PASS",first["injected_replay_action_max_error"],second["derived_count"],flush=True)
    base.close();online.close()

def pilot():
    if not (ROOT/"smoke.json").exists():raise ValueError("real GPU smoke required")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise ValueError("committed clean version required")
    plan=json.loads((ROOT/"preregistration.json").read_text()) if (ROOT/"preregistration.json").exists() else prepare()
    if plan["source_hash"]!=source_hash() or plan["mechanism_digest"]!=mechanism_digest():raise ValueError("frozen version changed")
    output=ROOT/"runs/M"
    if (output/"completion.json").exists():raise ValueError("already complete")
    c=OmegaConf.load(ROOT/"protocol/arm_uniform.yaml")
    observer=Observer(plan)
    resume=find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:raise ValueError("no sealed recovery checkpoint")
    started=time.perf_counter()
    state,report=run_training(c,output,checkpoint_interval=50,keep=4,
        ablation=observer,fork=resume is None,resume_checkpoint=resume)
    rows=accepted_rows(output)
    a=json.loads((BASE/"summary-A.json").read_text())
    if report["initial"]!=a["performance"]["initial"] or report["final"]!=a["performance"]["final"]:raise ValueError("A/M budgets differ")
    if report["evaluations"][:4]!=a["performance"]["evaluations"][:4]:raise ValueError("shared CP50 prefix changed")
    if np.asarray(state.key).tolist()!=a["final_prng"] or np.asarray(state.ec_opt_state.key).tolist()!=a["final_ec_prng"]:raise ValueError("baseline random streams differ")
    warmup=plan["shared_parent"]["common_prefix_training_seconds"]
    total=warmup
    timing=[]
    for row in rows:
        total+=row["step_seconds"]
        if "evaluation" in row:timing.append(dict(row["evaluation"],training_elapsed_seconds=total))
    domain=[r for r in report["evaluations"] if r["generation"]>=60]
    primary_auc=sum(.5*(a["return_mean"]+b["return_mean"])*(b["environment_timesteps"]-a["environment_timesteps"]) for a,b in zip(domain,domain[1:]))
    result=dict(performance=report,primary_auc60_150=primary_auc,total_training_seconds=total,
        core_training_seconds=warmup+sum(r["core_step_seconds"] for r in rows),
        process_wall_seconds=time.perf_counter()-started,evaluation_timing=timing,
        injection_events=sum(r["injected"] for r in rows),final_M_ancestry=rows[-1]["derived_next_count"],
        final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist())
    write(ROOT/"summary-M.json",result)
    print("COMPLETE M",report["auc"],report["final_return"],"minutes",total/60,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","pilot"],required=True);args=p.parse_args()
    if jax.default_backend()!="gpu":raise ValueError("GPU required")
    smoke() if args.stage=="smoke" else pilot()
if __name__=="__main__":main()
