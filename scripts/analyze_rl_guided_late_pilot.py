"""CPU-only pre-fixed M-Late AUC100..150, Fitness, breeding, time and curves."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-late-seed3083-v1")
BASE=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1")
OLD=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-population-seed3083-v1")
def auc(rows):
    rows=[r for r in rows if 100<=r["generation"]<=150]
    assert [r["generation"] for r in rows]==list(range(100,151,10))
    return float(np.trapz([r["return_mean"] for r in rows],[r["environment_timesteps"] for r in rows]))
def main():
    plan=json.loads((ROOT/"preregistration.json").read_text())
    a=json.loads((BASE/"summary-A.json").read_text())
    late=json.loads((ROOT/"summary-M.json").read_text())
    old=json.loads((OLD/"summary-M.json").read_text())
    paths=sorted((ROOT/"runs/M").glob("segments/*/evolution.jsonl"),key=lambda p:p.stat().st_mtime)
    bygen={r["generation"]:r for p in paths for r in map(json.loads,p.read_text().splitlines())}
    rows=[bygen[g] for g in sorted(bygen) if not bygen[g].get("bootstrap")]
    assert [r["generation"] for r in rows]==list(range(101,151))
    assert [r["generation"] for r in rows if r["injected"]]==[110,120,130,140,150]
    curves=[];result={}
    for label,data in (("A",a),("M-Late",late),("old-M",old)):
        p=data["performance"];ev=p["evaluations"]
        assert p["final"]==a["performance"]["final"]
        timed=a["evaluation_timings"] if label=="A" else (
            a["evaluation_timings"][:9]+late["evaluation_timing"] if label=="M-Late" else None)
        result[label]=dict(auc100_150=auc(ev),final_return=p["final_return"],
            return110_150=[dict(generation=r["generation"],return_mean=r["return_mean"]) for r in ev if r["generation"]>=110])
        if label=="A":result[label]["total_training_seconds"]=a["measured_total_training_seconds"]
        if label=="M-Late":result[label].update(total_training_seconds=late["total_training_seconds"],
            new_step_seconds=late["new_step_seconds"],process_wall_seconds=late["process_wall_seconds"])
        if label!="old-M":
            timed={r["generation"]:r["training_elapsed_seconds"] for r in timed}
            for r in ev:
                if r["generation"]>=100:curves.append(dict(arm=label,**r,training_elapsed_seconds=timed[r["generation"]]))
    assert np.isclose(result["M-Late"]["auc100_150"],late["primary_auc100_150"],rtol=1e-12)
    gain=result["M-Late"]["auc100_150"]/result["A"]["auc100_150"]-1
    result["primary_relative_gain"]=gain
    result["descriptive_decision"]="positive-signal" if gain>.02 else "negative" if gain<-.02 else "near"
    result["old_M_same_interval_relative_gain"]=result["old-M"]["auc100_150"]/result["A"]["auc100_150"]-1
    events=[]
    for r in rows:
        if r["injected"]:
            events.append(dict(generation=r["generation"],fresh8_mean_fitness=float(np.mean(r["fitness"][24:])),
                other24_mean_fitness=float(np.mean(r["fitness"][:24])),
                evaluated_M_ancestry=r["derived_count"],next_M_ancestry=r["derived_next_count"],
                next_current_cohort_individuals=sum(any(root//100==r["generation"] for root in x["roots"]) for x in r["next_lineage"]),
                exact_collector_receipt=r["collector_receipt"]["replay_obs_actions_equal_actual_collector"]))
    result["injections"]=events
    result["breeding"]=dict(candidates=40,evaluated_ancestry_source_slots=sum(r["derived_count"] for r in rows),
        descendant_source_slots=sum(sum(bool(x["roots"]) and x["depth"]>0 for x in r["evaluated_lineage"]) for r in rows),
        M_parent_uses=sum(r["selected_M_parent_slots"] for r in rows),
        generations_with_M_breeding=sum(r["selected_M_parent_slots"]>0 for r in rows),final_next_ancestry=rows[-1]["derived_next_count"])
    old_events=json.loads((OLD/"mechanism_details.json").read_text())["injection_summary"]
    result["old_M_same_generation_fresh_fitness"]=[dict(generation=r["generation"],fitness=r["mean_direct_fitness"]) for r in old_events if r["generation"]>=110]
    original_rows=[json.loads(line) for p in (BASE/"runs/A").glob("segments/*/schedule.jsonl") for line in p.read_text().splitlines()]
    result["A_matching_new_segment_step_seconds"]=sum(r["step_seconds"] for r in original_rows if r["generation"]>100)
    result["first_new_Late_step_seconds"]=rows[0]["step_seconds"]
    result["steady_Late_step_mean"]=float(np.mean([r["step_seconds"] for r in rows[1:]]))
    result["utility"]=json.loads((ROOT/"utility_summary.json").read_text())
    (ROOT/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    with (ROOT/"learning_curves.csv").open("w",newline="") as f:
        fields=["arm","generation","environment_timesteps","return_mean","training_elapsed_seconds"]
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");wr.writeheader();wr.writerows(curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout="constrained")
    for label in ("A","M-Late"):
        data=[r for r in curves if r["arm"]==label]
        axes[0].plot([r["generation"] for r in data],[r["return_mean"] for r in data],marker="o",label=label)
        axes[1].plot([r["training_elapsed_seconds"]/60 for r in data],[r["return_mean"] for r in data],marker="o",label=label)
    for ax in axes[:2]:ax.set_ylabel("Evaluation mean Return (128 episodes)");ax.grid(alpha=.25);ax.legend()
    axes[0].axvline(110,color="gray",linestyle="--",label="first injection")
    axes[0].set_xlabel("Generation; shared original A CP100")
    axes[1].set_xlabel("Measured cumulative training minutes")
    axes[2].plot([r["generation"] for r in rows],[r["derived_count"] for r in rows],label="evaluated M-Late ancestry")
    axes[2].plot([r["generation"] for r in rows],[r["derived_next_count"] for r in rows],label="next generation ancestry")
    axes[2].set_xlabel("Generation");axes[2].set_ylabel("Individuals /32");axes[2].set_ylim(-1,33);axes[2].grid(alpha=.25);axes[2].legend()
    fig.suptitle("M-Late: fixed seed3083 | original A CP100 | first injection110")
    fig.savefig(ROOT/"learning_curves.png",dpi=180);fig.savefig(ROOT/"learning_curves.pdf")
    print(json.dumps({k:v for k,v in result.items() if k!="utility"},indent=2))
if __name__=="__main__":main()
