"""R CP150: synthetic source ranks, frozen A/P controls, conditional second seed."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
import jax
import numpy as np
from omegaconf import OmegaConf
from scripts.random_source_priority import RandomSourcePriorityWorkflow
from scripts.run_budgeted_utility_pilot import write
from scripts.run_s_utility_extension import validate_extension_config,create_bootstrap_checkpoint,find_resume_checkpoint
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
from evorl.algorithms.erl.erl_td3.d0_shared_start import AuditedD0
from evorl.algorithms.erl.erl_td3.budgeted_utility_scoring import score_quota

ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/random-source-priority-v1")
CONTROLS={3083:Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1"),
3079:Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-cp150-v1")}
EARLY3079=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-v2/pilot")

def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("random_source_priority.py"))}

def rows_for(output,name="schedule.jsonl"):
    paths=sorted(Path(output).glob("segments/*/"+name),key=lambda p:p.stat().st_mtime)
    rows=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
    latest={r["generation"]:r for r in rows}
    return [latest[g] for g in sorted(latest)]

def initial_parent(seed):
    if seed==3083:return Path(json.loads((CONTROLS[seed]/"shared_warmup.json").read_text())["checkpoint"])
    return Path(json.loads((EARLY3079/"runs/A/completion.json").read_text())["segment"])/"checkpoints/10"

def prepare():
    configs={}
    for seed in (3083,3079):
        folder=ROOT/f"seed_{seed}";folder.mkdir(exist_ok=False)
        parent=initial_parent(seed);meta=json.loads((parent/"runtime/metadata").read_text())
        if meta["generation"]!=10 or meta["runtime"]["evaluations"] or meta["runtime"]["budget"]["steps"]:raise ValueError("not original complete A CP10")
        configs[str(seed)]=dict(control_root=str(CONTROLS[seed]),parent_checkpoint=str(parent),parent_state_sha256=meta["state_sha256"])
        protocol=folder/"protocol";protocol.mkdir()
        arms=[]
        for name in ("uniform","fitness","td","utility"):
            c=OmegaConf.load(CONTROLS[seed]/"protocol"/f"arm_{name}.yaml")
            c.output_dir=str(folder/"runs"/name);c.arm_manifest_path=str(protocol/"arm_manifest.json")
            OmegaConf.save(c,protocol/f"arm_{name}.yaml",resolve=True)
            with (protocol/f"arm_{name}.yaml").open("a") as f:
                f.write("\n# R uses legacy weighted-arm diagnostic flag/tag only as schema compatibility.\n# Effective Utility, virtual updates, source/reference batch construction are forbidden.\n")
            arms.append(c)
        prepare_arm_manifest(arms,protocol/"arm_manifest.json")
    plan=dict(algorithm="R",seed_order=[3083,3079],seeds=configs,
        base_commit="f1cdb57b63cfda9c843a4a82acd163ead1df7fa3",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_hash=hashes(),mechanism_digest=mechanism_digest(),
        alpha=.5,temperature=.5,freshness_window=30,min_resident=4,
        source_batch_minimum_count=256,quota="3,3,3,3,4 repeated;32per10g",
        random_ranks="canonical equal-spaced valid rank grid; same P stable per-eligible-cohort permutation",
        persistence="same residency/freshness/partial-cohort permutation behavior as P",
        utility_calls=0,virtual_critic_updates=0,reference_batches=0,source_minibatches=0,
        legacy_schema=dict(arm="utility",score_kind="utility",diagnostic_enabled=True,numeric_tag=3,
            interpretation="compatibility ABI only; raw score is synthetic priority, never measured Utility"),
        differences_from_P=["no true-Utility nonfinite gate","no true-Utility ties; full finite grid",
            "baseline jitted EC training collector instead of raw diagnostic batch path",
            "random rank assignment is equal in distribution absent ties, not identical realized P ranks"],
        primary="Return AUC20..150,650000..4940000 steps",final_generation=150,
        evaluation="every10g20..150 x128episodes",threshold="first scheduledmeanReturn>=350, no imputation",
        timing="post-CP10 synchronized training step wall; includes compilation/scoring/sampling, excludes evaluation/checkpoint I/O",
        gate_second_seed_relative_A_auc=.05,near_A_band=.05,near_P_band=.10,meaningful_cost_reduction_P=.25,
        no_interim_tuning=True,no_replacement_seed=True,no_formal_multiseed_training=True)
    write(ROOT/"preregistration.json",plan);return plan

class RObserver:
    workflow_type=RandomSourcePriorityWorkflow
    audit_before_checkpoint=True
    reconstruct_sample_evidence=False
    def __init__(self,plan,seed):
        self.plan=plan;self.seed=seed;self.parent=Path(plan["seeds"][str(seed)]["parent_checkpoint"])
        self.expected=plan["seeds"][str(seed)]["parent_state_sha256"]
        self.identity=dict(algorithm="R",priority_kind="synthetic_random_cohort_rank",measured_utility=False,seed=seed,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        self.rows=[]
    def configure(self,w,segment):
        self.w=w;self.segment=segment
    def fork_state(self,w,run_id):
        c=OmegaConf.load(self.parent.parents[3]/"resolved_config.yaml")
        validate_extension_config(c,w.config)
        with PilotCheckpointStore(self.parent.parent,c,read_only=True) as store:
            state,runtime=store.restore(10,restore_template(w))
        if initialization_fingerprint(state)!=self.expected or runtime["evaluations"] or runtime["budget"]["steps"]:raise ValueError("shared full A CP10 mismatch")
        runtime=copy.deepcopy(runtime);runtime.update(run_id=run_id,target_generation=150)
        create_bootstrap_checkpoint(self.segment,w.config,state,runtime)
        return state,runtime
    def before_generation(self,state):
        if int(state.metrics.iterations)==10 and initialization_fingerprint(state)!=self.expected:raise ValueError("initial state changed")
    def record(self,w,state,checkpoint,runtime):
        g=int(state.metrics.iterations)
        if not w.baseline_key_ok:raise ValueError("baseline PRNG changed")
        replay,key,updates,u=w.learner_input
        if updates!=16000:raise ValueError("TD3 update budget changed")
        priority=w.priority_assignment
        quota=score_quota(g,32,10,10)
        if priority["new_priorities"]!=quota:raise ValueError("synthetic priority eligibility/quota failed")
        if any(priority[k]!=0 for k in ("real_utility_evaluations","virtual_critic_updates","reference_batches","source_minibatches")):raise ValueError("forbidden Utility computation")
        t=u.score_table;resident=np.asarray(t.resident_count)>0;eligible=np.asarray(t.weighted_eligible)&resident
        ages=g-np.asarray(t.measured_generation)[eligible]
        if np.any(ages<0) or np.any(ages>30) or np.any(np.asarray(t.resident_count)[eligible]<4):raise ValueError("source eligibility violated")
        d=u.distribution
        row=dict(priority,step_seconds=w.step_seconds,permutation_seconds=w.permutation_seconds,
            resident_sources=int(resident.sum()),eligible_sources=int(eligible.sum()),
            eligible_coverage=float(eligible.sum()/max(1,resident.sum())),
            mean_eligible_age=float(ages.mean()) if len(ages) else None,max_eligible_age=int(ages.max()) if len(ages) else None,
            ea_conditional_tv=float(np.abs(np.asarray(d.source_probability)-np.asarray(d.baseline_source_probability)).sum()/2),
            eligible_mass=float(d.eligible_baseline_mass),ess_fraction=float(d.ess_fraction),
            timesteps=int(state.metrics.sampled_timesteps),actor_count=int(state.opt_state.actor[0].count),critic_count=int(state.opt_state.critic[0].count),
            baseline_prng_ok=w.baseline_key_ok,rank_multisets_preserved=w.permutation.get("preserved_rank_multisets",False))
        if not row["rank_multisets_preserved"]:raise ValueError("random rank grid changed")
        if row["timesteps"]!=(320+(g-10)*33)*1000 or row["actor_count"]!=(g-10)*16000 or row["critic_count"]!=(g-10)*32000:raise ValueError("interaction/update budget mismatch")
        token_array=np.asarray(t.source_token)
        slots={int(token_array[i]):int(i) for i in np.flatnonzero(token_array>=0)}
        donors=dict(w.permutation["mapping"])
        row["effective_current_random_priorities"]=[
            [int(i),float(t.rank_normalized_score[slots[donors[g*32+int(i)]]])]
            for i in priority["selected_individuals"]]
        if g in (20,150):
            baseline,final=map(np.asarray,w.learner_trace[:2]);origin=np.asarray(replay.origin_type)
            if baseline.shape!=(updates,2,256) or np.any(final<0) or np.any(final>=int(replay.buffer_size)):raise ValueError("actual TD3 index shape/range")
            rl=origin[baseline]==1;ea=origin[baseline]==0
            if np.any(final[rl]!=baseline[rl]) or np.any(origin[final]!=origin[baseline]) or not np.any((final!=baseline)&ea):raise ValueError("actual R sampling wrong")
            positions=np.linspace(0,len(baseline)-1,8,dtype=int)
            np.savez_compressed(self.segment/f"critic-input-{g}.npz",baseline=baseline[positions],final=final[positions],
                origin=origin,source_token=np.asarray(replay.source_token),sampled_update_positions=positions)
            row.update(actual_critic_audit=True,rl_changed=0,changed_ea=int(((final!=baseline)&ea).sum()),actual_critic_batches=updates*2)
        if runtime["evaluations"] and runtime["evaluations"][-1]["generation"]==g:row["evaluation"]=runtime["evaluations"][-1]
        with (self.segment/"schedule.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        self.rows.append(row);w.learner_input=None;w.learner_trace=None

def smoke():
    if (ROOT/"smoke.json").exists():raise ValueError("smoke already complete")
    import evorl.algorithms.erl.erl_td3.erl_learner_utility as forbidden_module
    def forbidden(*args,**kwargs):raise AssertionError("real Utility/source/reference batch path executed in R")
    forbidden_module.build_source_batches=forbidden;forbidden_module.virtual_critic_utility=forbidden
    c=OmegaConf.load(CONTROLS[3083]/"protocol/arm_uniform.yaml")
    c.env.max_episode_steps=8;c.replay_buffer_capacity=4096;c.batch_size=8;c.source_batch_size=4;c.num_eval_envs=2;c.eval_episodes=2
    c.agent_network.actor_hidden_layer_sizes=[8,8];c.agent_network.critic_hidden_layer_sizes=[8,8]
    base=type("RSmokeA",(AuditedD0,),{}).build_from_config(c,enable_jit=True)
    initial=base.init(jax.random.PRNGKey(3083))
    c.arm="utility";c.score_kind="utility";c.alpha=.5;c.diagnostic_enabled=True
    w=type("RSmoke",(RandomSourcePriorityWorkflow,),{}).build_from_config(c,enable_jit=True)
    state=initial;evidence=[]
    for g in (11,12):
        _,state=w.step(state)
        baseline,final=map(np.asarray,w.learner_trace[:2]);replay=w.learner_input[0];origin=np.asarray(replay.origin_type)
        rl=origin[baseline]==1;ea=origin[baseline]==0
        assert np.array_equal(final[rl],baseline[rl]) and np.array_equal(origin[final],origin[baseline])
        assert np.any((final!=baseline)&ea)
        assert w.priority_assignment["new_priorities"]==3
        evidence.append(dict(generation=g,changed_ea=int(((final!=baseline)&ea).sum()),rl_changed=0,
            priority=w.priority_assignment,eligible=int(w.learner_input[3].distribution.eligible_source_count)))
        w.learner_input=None;w.learner_trace=None
    write(ROOT/"smoke.json",dict(status="PASS",real_Brax_rollout=True,forbidden_utility_paths_tripwired=True,
        utility_calls=0,virtual_updates=0,reference_batches=0,source_minibatches=0,
        physical_environment_steps=3088,final_episodes=int(state.metrics.sampled_episodes),evidence=evidence))
    print("REAL R GPU SMOKE PASS",evidence,flush=True)
    base.close();w.close()

def train(seed):
    if not (ROOT/"smoke.json").exists():raise ValueError("real R smoke required")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise ValueError("clean committed source required")
    plan=json.loads((ROOT/"preregistration.json").read_text()) if (ROOT/"preregistration.json").exists() else prepare()
    if plan["source_hash"]!=hashes() or plan["mechanism_digest"]!=mechanism_digest():raise ValueError("preregistered R code changed")
    if seed==3079:
        earlier=json.loads((ROOT/"seed_3083/summary-R.json").read_text())
        base=json.loads((CONTROLS[3083]/"summary-A.json").read_text())
        gain=earlier["performance"]["auc"]/base["performance"]["auc"]-1
        if gain<plan["gate_second_seed_relative_A_auc"]:
            print("SECOND SEED NOT TRIGGERED",gain,flush=True);return
    folder=ROOT/f"seed_{seed}";output=folder/"runs/R"
    if (output/"completion.json").exists():raise ValueError("already complete")
    resume=find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:raise ValueError("no sealed recovery checkpoint")
    observer=RObserver(plan,seed);c=OmegaConf.load(folder/"protocol/arm_utility.yaml")
    started=time.perf_counter()
    state,report=run_training(c,output,checkpoint_interval=50,keep=4,
        ablation=observer,fork=resume is None,resume_checkpoint=resume)
    rows=rows_for(output)
    control=json.loads((CONTROLS[seed]/"summary-A.json").read_text())
    for field in ("initial","final","auc_domain"):
        if report[field]!=control["performance"][field]:raise ValueError("A/R equal budget failure:"+field)
    if np.asarray(state.key).tolist()!=control["final_prng"] or np.asarray(state.ec_opt_state.key).tolist()!=control["final_ec_prng"]:raise ValueError("baseline random streams differ")
    if sum(r["new_priorities"] for r in rows)!=448 or len(rows)!=140:raise ValueError("priority count/schedule incomplete")
    elapsed=0.;timing=[]
    for row in rows:
        elapsed+=row["step_seconds"]
        if "evaluation" in row:timing.append(dict(row["evaluation"],postwarmup_training_seconds=elapsed))
    result=dict(algorithm="R",seed=seed,performance=report,postwarmup_step_seconds=elapsed,
        process_wall_seconds=time.perf_counter()-started,synthetic_priorities=448,real_utility_calls=0,
        virtual_critic_updates=0,reference_batches=0,source_minibatches=0,
        first350=next((r for r in timing if r["return_mean"]>=350),None),evaluation_timing=timing,
        mean_eligible_coverage=float(np.mean([r["eligible_coverage"] for r in rows])),
        mean_ea_conditional_tv=float(np.mean([r["ea_conditional_tv"] for r in rows])),
        final_prng=np.asarray(state.key).tolist(),final_ec_prng=np.asarray(state.ec_opt_state.key).tolist())
    write(folder/"summary-R.json",result)
    print("COMPLETE R",seed,"AUC",report["auc"],"first350",result["first350"],"stepminutes",elapsed/60,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","train"],required=True);p.add_argument("--seed",type=int,choices=[3083,3079],default=3083);args=p.parse_args()
    if jax.default_backend()!="gpu":raise ValueError("GPU required")
    smoke() if args.stage=="smoke" else train(args.seed)
if __name__=="__main__":main()
