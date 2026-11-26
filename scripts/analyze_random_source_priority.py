"""CPU-only R versus frozen A/P: equal horizon, real post-warmup timing."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/random-source-priority-v1")
CONTROLS={3083:Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1"),
3079:Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-cp150-v1")}
EARLY=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-v2/pilot")

def journal(paths,name="schedule.jsonl"):
    files=sorted([p for root in paths for p in (root/"segments").glob("*/"+name)],key=lambda p:p.stat().st_mtime)
    result={r["generation"]:r for p in files for r in map(json.loads,p.read_text().splitlines())}
    return [result[g] for g in sorted(result)]

def performance(seed,label):
    folder=ROOT/f"seed_{seed}"
    if label=="R":data=json.loads((folder/"summary-R.json").read_text());rows=journal([folder/"runs/R"])
    else:
        data=json.loads((CONTROLS[seed]/f"summary-{label}.json").read_text())
        paths=[CONTROLS[seed]/"runs"/label]
        if seed==3079 and label=="A":paths.insert(0,EARLY/"runs/A")
        rows=journal(paths)
    if [r["generation"] for r in rows]!=list(range(11,151)):raise ValueError("equal-horizon training timings incomplete")
    p=data["performance"];ev=p["evaluations"]
    if [r["generation"] for r in ev]!=list(range(20,151,10)):raise ValueError("evaluation schedule differs")
    auc=float(np.trapz([r["return_mean"] for r in ev],[r["environment_timesteps"] for r in ev]))
    assert np.isclose(auc,p["auc"],rtol=1e-12)
    elapsed=0.;times={}
    for row in rows:elapsed+=row["step_seconds"];times[row["generation"]]=elapsed
    hit=next((dict(r,postwarmup_training_seconds=times[r["generation"]]) for r in ev if r["return_mean"]>=350),None)
    result=dict(auc20_150=auc,final_return=p["final_return"],
        last5_mean=float(np.mean([r["return_mean"] for r in ev[-5:]])),
        postwarmup_step_seconds=elapsed,first350=hit,process_wall_seconds=data.get("process_wall_seconds",data.get("wall_seconds")),
        eligible_coverage_mean=float(np.mean([r["eligible_coverage"] for r in rows])),
        ea_conditional_tv_mean=float(np.mean([r["ea_conditional_tv"] for r in rows])))
    return result,[dict(arm=label,seed=seed,**r,postwarmup_training_seconds=times[r["generation"]]) for r in ev]

def selected_id_comparison(seed):
    rrows=journal([ROOT/f"seed_{seed}/runs/R"])
    files=sorted((CONTROLS[seed]/"runs/P/segments").glob("*/permutation.jsonl"),key=lambda p:p.stat().st_mtime)
    if not files:return dict(available=False)
    prows={r["generation"]:r for p in files for r in map(json.loads,p.read_text().splitlines())}
    compared=matched=0
    for r in rrows:
        p=prows.get(r["generation"])
        if not p or "mapping" not in p:continue
        ids=sorted({int(a)%32 for a,b in p["mapping"] if int(a)//32==r["generation"]})
        if ids:compared+=1;matched+=ids==r["selected_individuals"]
    return dict(available=compared>0,generations_compared=compared,generations_equal=matched)

def main():
    plan=json.loads((ROOT/"preregistration.json").read_text());completed={};curves=[]
    for seed in (3083,3079):
        if not (ROOT/f"seed_{seed}/summary-R.json").exists():continue
        result={}
        for label in ("A","P","R"):
            result[label],curve=performance(seed,label);curves.extend(curve)
        result["R_over_A_auc"]=result["R"]["auc20_150"]/result["A"]["auc20_150"]-1
        result["R_over_P_auc"]=result["R"]["auc20_150"]/result["P"]["auc20_150"]-1
        result["R_over_P_step_time"]=result["R"]["postwarmup_step_seconds"]/result["P"]["postwarmup_step_seconds"]
        result["R_over_A_step_time"]=result["R"]["postwarmup_step_seconds"]/result["A"]["postwarmup_step_seconds"]
        result["near_P"]=abs(result["R_over_P_auc"])<=plan["near_P_band"]
        result["near_A"]=abs(result["R_over_A_auc"])<=plan["near_A_band"]
        result["meaningfully_faster_P"]=result["R_over_P_step_time"]<=1-plan["meaningful_cost_reduction_P"]
        result["source_selection_comparison_P"]=selected_id_comparison(seed)
        completed[str(seed)]=result
    gate=completed["3083"]["R_over_A_auc"]>=plan["gate_second_seed_relative_A_auc"]
    final=dict(seeds=completed,second_seed_triggered=gate,
        second_seed_completed="3079" in completed,
        interpretation="development seeds separate, no pooling; priorities not measured Utility",
        timing="post-CP10 real synchronized training steps; shared warmup excluded for all arms")
    (ROOT/"comparison.json").write_text(json.dumps(final,indent=2,allow_nan=False))
    fields=["seed","arm","generation","environment_timesteps","return_mean","postwarmup_training_seconds"]
    with (ROOT/"learning_curves.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");wr.writeheader();wr.writerows(curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(completed),2,figsize=(11,4.2*len(completed)),layout="constrained",squeeze=False)
    for i,seed in enumerate(completed):
        for label in ("A","P","R"):
            rows=[r for r in curves if r["seed"]==int(seed) and r["arm"]==label]
            for ax,xx,xlabel in [(axes[i,0],[r["environment_timesteps"]/1e6 for r in rows],"Training environment steps (million)"),
                (axes[i,1],[r["postwarmup_training_seconds"]/60 for r in rows],"Measured post-CP10 training minutes")]:
                ax.plot(xx,[r["return_mean"] for r in rows],marker="o",label=label)
                ax.axhline(350,color="gray",linestyle="--",linewidth=.8);ax.set_xlabel(xlabel);ax.set_ylabel("Mean Return (128 episodes)")
                ax.set_title("Fixed development seed"+seed);ax.grid(alpha=.25);ax.legend()
    fig.suptitle("R synthetic persistent source ranks vs frozen A/P | real Utility calls=0")
    fig.savefig(ROOT/"learning_curves.png",dpi=180);fig.savefig(ROOT/"learning_curves.pdf")
    print(json.dumps(final,indent=2))
if __name__=="__main__":main()
