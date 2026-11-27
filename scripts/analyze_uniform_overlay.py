"""Compare frozen A/P/R and U-Overlay on the common CP20-100 interval."""

import csv
import json
from pathlib import Path

import numpy as np

from scripts.analyze_random_source_priority import journal, performance

ROOT = Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/uniform-overlay-cp100-v1")
SEEDS = (3083, 3079)
ARMS = ("A", "P", "R", "U")
FIELDS = ("seed", "arm", "generation", "environment_timesteps", "return_mean",
          "postwarmup_training_seconds")


def overlay_curve(seed):
    folder = ROOT / f"seed_{seed}"
    summary = json.loads((folder / "summary-U.json").read_text())
    rows = journal([folder / "runs/U"])
    assert [row["generation"] for row in rows] == list(range(11, 101))
    cumulative = dict(zip(range(11, 101), np.cumsum([r["step_seconds"] for r in rows])))
    return [dict(seed=seed, arm="U", **row,
                 postwarmup_training_seconds=float(cumulative[row["generation"]]))
            for row in summary["performance"]["evaluations"]]


def metrics(rows):
    assert [row["generation"] for row in rows] == list(range(20, 101, 10))
    steps = [row["environment_timesteps"] for row in rows]
    assert steps == list(range(650000, 3290001, 330000))
    returns = [row["return_mean"] for row in rows]
    auc = sum(0.5 * (a + b) * (x2 - x1) for a, b, x1, x2 in
              zip(returns, returns[1:], steps, steps[1:]))
    first350 = next((row for row in rows if row["return_mean"] >= 350), None)
    return dict(auc20_100=auc, return100=returns[-1], first350=first350,
                training_step_seconds=rows[-1]["postwarmup_training_seconds"])


def main():
    result, curves = {}, []
    for seed in SEEDS:
        by_arm = {}
        for arm in ARMS:
            full = overlay_curve(seed) if arm == "U" else performance(seed, arm)[1]
            rows = [row for row in full if row["generation"] <= 100]
            by_arm[arm] = metrics(rows)
            curves.extend(rows)
        result[str(seed)] = dict(arms=by_arm, **{
            f"{left}_vs_{right}_auc_relative": by_arm[left]["auc20_100"] / by_arm[right]["auc20_100"] - 1
            for left, right in (("U", "A"), ("U", "R"), ("R", "A"))
        })
        summary = json.loads((ROOT / f"seed_{seed}" / "summary-U.json").read_text())
        assert np.isclose(by_arm["U"]["auc20_100"], summary["performance"]["auc"], rtol=1e-12)
        result[str(seed)]["u_audits"] = summary["audits"]
        result[str(seed)]["u_process_wall_seconds"] = summary["process_wall_seconds"]
    with (ROOT / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    for i, seed in enumerate(SEEDS):
        for column, (key, scale, label) in enumerate((
            ("environment_timesteps", 1e6, "Training environment steps (millions)"),
            ("postwarmup_training_seconds", 60, "Training minutes after CP10"),
        )):
            ax = axes[i, column]
            for arm in ARMS:
                rows = [r for r in curves if r["seed"] == seed and r["arm"] == arm]
                ax.plot([r[key] / scale for r in rows], [r["return_mean"] for r in rows],
                        marker="o", label="U-Overlay" if arm == "U" else arm)
            ax.axhline(350, color="gray", linestyle="--", linewidth=0.8)
            ax.set(title=f"Seed {seed}", xlabel=label, ylabel="Mean Return (128 episodes)")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.savefig(ROOT / "learning_curves.png", dpi=160)
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
