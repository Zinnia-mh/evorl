"""CPU-only fixed A/M Pilot metrics and plots, with no training imports."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-population-seed3083-v1")
BASE=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1")
def auc(rows,first=60):
    rows=[r for r in rows if r["generation"]>=first]
    return float(np.trapz([r["return_mean"] for r in rows],[r["environment_timesteps"] for r in rows]))
def main():
    a=json.loads((BASE/"summary-A.json").read_text())
    m=json.loads((ROOT/"summary-M.json").read_text())
    u=json.loads((ROOT/"utility_summary.json").read_text())
    plans=json.loads((ROOT/"preregistration.json").read_text())
    paths=sorted((ROOT/"runs/M").glob("segments/*/evolution.jsonl"),key=lambda p:p.stat().st_mtime)
    unique={r["generation"]:r for p in paths for r in map(json.loads,p.read_text().splitlines())}
    rows=[unique[g] for g in sorted(unique) if not unique[g].get("bootstrap")]
    assert [r["generation"] for r in rows]==list(range(51,151))
    assert [r["generation"] for r in rows if r["injected"]]==list(range(60,151,10))
    assert sum(r["injected_count"] for r in rows)==80
    result={}
    curves=[]
    for label,data in (("A",a),("M",m)):
        p=data["performance"];evals=p["evaluations"]
        assert p["initial"]==a["performance"]["initial"] and p["final"]==a["performance"]["final"]
        assert [r["generation"] for r in evals]==list(range(20,151,10))
        timing=a["evaluation_timings"] if label=="A" else a["evaluation_timings"][:4]+m["evaluation_timing"]
        assert len(timing)==14
        timed={r["generation"]:r["training_elapsed_seconds"] for r in timing}
        first=next((r for r in evals if r["return_mean"]>=350),None)
        first=dict(first,training_elapsed_seconds=timed[first["generation"]]) if first else None
        result[label]=dict(primary_auc60_150=auc(evals,60),aux_auc20_150=auc(evals,20),
            final_return=p["final_return"],last5_mean=float(np.mean([r["return_mean"] for r in evals[-5:]])),
            first350=first,training_seconds=a["measured_total_training_seconds"] if label=="A" else m["total_training_seconds"])
        for r in evals:curves.append(dict(arm=label,**r,training_elapsed_seconds=timed[r["generation"]]))
    result["M_over_A"]={k:m_value/result["A"][k]-1 for k,m_value in result["M"].items() if k in ("primary_auc60_150","aux_auc20_150","training_seconds")}
    result["evolution"]=dict(injections=10,candidates=80,
        generations_with_M_fitness=sum(r["derived_count"]>0 for r in rows),
        actual_M_parent_selections=sum(r["selected_M_parent_slots"] for r in rows),
        generations_with_M_breeding=sum(r["selected_M_parent_slots"]>0 for r in rows),
        final_M_ancestry=rows[-1]["derived_next_count"],
        max_replay_action_check=max(r["injected_replay_action_max_error"] for r in rows if r["injected"]),
        injection_geometry=[dict(generation=r["generation"],before=r["before_injection_geometry"],after=r["geometry"]) for r in rows if r["injected"]])
    result["utility_summaries"]=u["summaries"]
    result["diagnostic_seconds"]=u["elapsed_seconds"]
    result["post_GA_geometry"]=[dict(generation=c["generation"],**c["post_GA_population_geometry"]) for c in u["contexts"]]
    result["new_physical_environment"]=dict(training=3300000,evaluation=1280000,diagnostic=0)
    (ROOT/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    fields=["arm","generation","environment_timesteps","return_mean","training_elapsed_seconds"]
    with (ROOT/"learning_curves.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout="constrained")
    for label in ("A","M"):
        rr=[r for r in curves if r["arm"]==label]
        axes[0].plot([r["environment_timesteps"]/1e6 for r in rr],[r["return_mean"] for r in rr],marker="o",label=label)
        axes[1].plot([r["training_elapsed_seconds"]/60 for r in rr],[r["return_mean"] for r in rr],marker="o",label=label)
    axes[0].axvline(1.64,color="gray",linestyle=":",label="shared CP50")
    axes[0].axvline(1.97,color="purple",linestyle="--",label="first M generation60")
    for ax in axes[:2]:ax.set_ylabel("Evaluation mean Return (128 episodes)");ax.axhline(350,color="gray",linestyle="--",linewidth=.8);ax.grid(alpha=.25);ax.legend()
    axes[0].set_xlabel("Training environment steps (million)")
    axes[1].set_xlabel("Measured training time (minutes)")
    axes[2].plot([r["generation"] for r in rows],[r["derived_count"] for r in rows],label="evaluated M ancestry")
    axes[2].plot([r["generation"] for r in rows],[r["derived_next_count"] for r in rows],label="next generation M ancestry")
    axes[2].set_ylim(-1,33);axes[2].set_xlabel("Generation");axes[2].set_ylabel("Individuals /32");axes[2].grid(alpha=.25);axes[2].legend()
    fig.suptitle("Online paired RL-seeded GA M | fixed development seed3083")
    fig.savefig(ROOT/"learning_curves.png",dpi=180);fig.savefig(ROOT/"learning_curves.pdf")
    print(json.dumps({k:result[k] for k in ("A","M","M_over_A","evolution","post_GA_geometry")},indent=2))
if __name__=="__main__":main()
