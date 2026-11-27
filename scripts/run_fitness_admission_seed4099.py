"""Independent seed-4099 replication of frozen Fitness admission at CP100."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.source_replay import EA_ORIGIN, RL_ORIGIN
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
from scripts.run_fitness_experience_admission import (
    AdmissionObserver,
    ARMS,
    file_sha,
    find_resume_checkpoint,
    run_rows,
    scalar,
    write_json,
)


ROOT = Path(
    "/disk1/zmh_projects/evorl_artifacts/"
    "fitness_experience_admission/seed4099-cp100-v1"
)
TEMPLATE = Path(
    "/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/"
    "algorithm/budgeted-scoring-v2/pilot/runs/A/resolved_config.yaml"
)
SEED = 4099
FINAL_GENERATION = 100
FINAL_EPISODES = 3290
LABELS = ("A", "F-Top16", "F-Random16")
PROTOCOLS = {
    "A": "protocol-a",
    "F-Top16": "protocol-top",
    "F-Random16": "protocol-random",
}


def buffer_summary(workflow, state):
    replay = state.replay_buffer_state
    capacity = int(workflow.replay_buffer.capacity)
    size = scalar(replay.buffer_size)
    resident = np.arange(capacity) < size
    origin = np.asarray(jax.device_get(replay.origin_type))
    generations = np.asarray(jax.device_get(replay.source_generation))
    ea = resident & (origin == EA_ORIGIN)
    rl = resident & (origin == RL_ORIGIN)
    generation = scalar(state.metrics.iterations)
    ages = generation - generations[ea]
    ea_count = int(ea.sum())
    rl_count = int(rl.sum())
    return {
        "buffer_size": size,
        "buffer_capacity": capacity,
        "buffer_current_index": scalar(replay.current_index),
        "resident_ea_transitions": ea_count,
        "resident_rl_transitions": rl_count,
        "resident_ea_fraction": ea_count / max(1, ea_count + rl_count),
        "resident_rl_fraction": rl_count / max(1, ea_count + rl_count),
        "resident_ea_age_mean": float(ages.mean()),
        "resident_ea_age_median": float(np.median(ages)),
        "resident_ea_age_max": int(ages.max()),
    }


def make_protocol(directory, mode, output):
    directory.mkdir(parents=True, exist_ok=False)
    configs = []
    for arm in ("uniform", "fitness", "td", "utility"):
        config = OmegaConf.load(TEMPLATE)
        config.seed = SEED
        config.arm = arm
        config.score_kind = arm
        config.alpha = 0.0 if arm == "uniform" else 0.5
        config.diagnostic_enabled = arm != "uniform"
        config.total_episodes = FINAL_EPISODES
        config.eval_interval = 10
        config.eval_episodes = 128
        config.num_eval_envs = 128
        config.admission_count = 16
        config.admission_mode = mode
        config.output_dir = str(output)
        config.arm_manifest_path = str(directory / "arm_manifest.json")
        OmegaConf.save(config, directory / f"arm_{arm}.yaml", resolve=True)
        configs.append(config)
    prepare_arm_manifest(configs, directory / "arm_manifest.json")
    return {
        path.name: file_sha(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    configurations = {
        "A": make_protocol(
            ROOT / PROTOCOLS["A"], "fitness_top", ROOT / "runs/A"
        ),
        "F-Top16": make_protocol(
            ROOT / PROTOCOLS["F-Top16"],
            "fitness_top",
            ROOT / "runs/F-Top16",
        ),
        "F-Random16": make_protocol(
            ROOT / PROTOCOLS["F-Random16"],
            "random",
            ROOT / "runs/F-Random16",
        ),
    }
    implementation = Path(__file__).with_name("fitness_experience_admission.py")
    frozen_runner = Path(__file__).with_name("run_fitness_experience_admission.py")
    plan = {
        "experiment": "Fitness Admission independent seed replication",
        "seed": SEED,
        "seed_precommitted": True,
        "seed_history_check": (
            "no seed-4099 manifest/config/log use found before preparation"
        ),
        "frozen_algorithm_commit": (
            "fb534b3b76f36a252025d90f54013482aa974da1"
        ),
        "runner_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "mechanism_digest": mechanism_digest(),
        "source_hashes": {
            implementation.name: file_sha(implementation),
            frozen_runner.name: file_sha(frozen_runner),
            Path(__file__).name: file_sha(Path(__file__)),
        },
        "arms": list(LABELS),
        "shared_parent": (
            "new original Uniform-A seed4099 complete CP10; all arms branch "
            "from exactly this state"
        ),
        "primary": (
            "Return AUC CP20-100 over 650000-3290000 training environment steps"
        ),
        "secondary": [
            "Return100",
            "first scheduled Return>=350",
            "process wall time",
            "EA/RL Buffer ratio",
            "admitted/rejected Fitness distribution",
        ],
        "evaluation": "every 10 generations, 128 episodes",
        "final_generation": FINAL_GENERATION,
        "no_interim_selection": True,
        "no_hyperparameter_change": True,
        "configurations": configurations,
    }
    write_json(ROOT / "preregistration.json", plan)
    print(json.dumps(plan, indent=2), flush=True)


def validate_plan():
    plan = json.loads((ROOT / "preregistration.json").read_text())
    if plan["seed"] != SEED or plan["mechanism_digest"] != mechanism_digest():
        raise ValueError("preregistered seed or frozen mechanism changed")
    for name, expected in plan["source_hashes"].items():
        path = Path(__file__) if name == Path(__file__).name else Path(__file__).with_name(name)
        if file_sha(path) != expected:
            raise ValueError(f"preregistered source changed: {name}")
    return plan


def config_for(label):
    return OmegaConf.load(ROOT / PROTOCOLS[label] / "arm_uniform.yaml")


def completed_summary(label):
    path = ROOT / f"summary-{label}.json"
    return json.loads(path.read_text()) if path.exists() else None


def train_a():
    existing = completed_summary("A")
    if existing is not None:
        print("ALREADY COMPLETE A", flush=True)
        return existing
    config = config_for("A")
    output = ROOT / "runs/A"
    resume = find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:
        raise ValueError("existing A run has no sealed resume checkpoint")
    started = time.perf_counter()
    state, report = run_training(
        config,
        output,
        checkpoint_interval=25,
        keep=5,
        resume_checkpoint=resume,
    )
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
        ERLUtilityReplayWorkflow,
    )

    workflow = type("Seed4099BufferReader", (ERLUtilityReplayWorkflow,), {}).build_from_config(
        config, enable_jit=False
    )
    result = {
        "algorithm": "A",
        "seed": SEED,
        "performance": report,
        "process_wall_seconds": time.perf_counter() - started,
        "final_training_key": np.asarray(state.key).tolist(),
        "final_ga_key": np.asarray(state.ec_opt_state.key).tolist(),
        "final_buffer": buffer_summary(workflow, state),
    }
    workflow.close()
    write_json(ROOT / "summary-A.json", result)
    checkpoint10 = Path(report["segment_directory"]) / "checkpoints/10"
    if not checkpoint10.is_dir():
        raise ValueError("A CP10 was not retained")
    metadata = json.loads((checkpoint10 / "runtime/metadata").read_text())
    write_json(
        ROOT / "shared-parent.json",
        {
            "checkpoint": str(checkpoint10),
            "state_sha256": metadata["state_sha256"],
            "seed": SEED,
            "generation": 10,
        },
    )
    print(
        "COMPLETE", SEED, "A", "AUC", report["auc"],
        "Return100", report["final_return"], flush=True,
    )
    return result


class Seed4099AdmissionObserver(AdmissionObserver):
    def __init__(self, plan, label, parent):
        self.plan = plan
        self.seed = SEED
        self.label = label
        self.mode = ARMS[label]
        self.parent = Path(parent)
        self.rows = []
        self.cumulative_writes = 0
        self.identity = {
            "algorithm": label,
            "seed": SEED,
            "mode": self.mode,
            "plan_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode()
            ).hexdigest(),
        }


def train_admission(label, plan):
    existing = completed_summary(label)
    if existing is not None:
        print("ALREADY COMPLETE", label, flush=True)
        return existing
    parent_info = json.loads((ROOT / "shared-parent.json").read_text())
    parent = Path(parent_info["checkpoint"])
    if not parent.is_dir():
        raise ValueError("shared A CP10 is missing")
    config = config_for(label)
    output = ROOT / "runs" / label
    observer = Seed4099AdmissionObserver(plan, label, parent)
    resume = find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:
        raise ValueError("existing admission run has no sealed resume checkpoint")
    started = time.perf_counter()
    state, report = run_training(
        config,
        output,
        checkpoint_interval=25,
        keep=5,
        ablation=observer,
        fork=resume is None,
        resume_checkpoint=resume,
    )
    rows = run_rows(output)
    if [row["generation"] for row in rows] != list(range(11, 101)):
        raise ValueError("incomplete admission generation journal")
    a = completed_summary("A")
    for field in ("initial", "final", "auc_domain"):
        if report[field] != a["performance"][field]:
            raise ValueError(f"A/admission budget mismatch: {field}")
    final_training_key = np.asarray(state.key).tolist()
    final_ga_key = np.asarray(state.ec_opt_state.key).tolist()
    if (
        final_training_key != a["final_training_key"]
        or final_ga_key != a["final_ga_key"]
    ):
        raise ValueError("baseline training or GA PRNG stream changed")
    result = {
        "algorithm": label,
        "seed": SEED,
        "performance": report,
        "mechanism": observer.summary(),
        "process_wall_seconds": time.perf_counter() - started,
        "final_training_key": final_training_key,
        "final_ga_key": final_ga_key,
        "parent_state_sha256": observer.parent_state_sha256,
    }
    write_json(ROOT / f"summary-{label}.json", result)
    print(
        "COMPLETE", SEED, label, "AUC", report["auc"],
        "Return100", report["final_return"], flush=True,
    )
    return result


def train(label):
    if jax.default_backend() != "gpu":
        raise ValueError("GPU backend required")
    plan = validate_plan()
    if label == "A":
        return train_a()
    if completed_summary("A") is None or not (ROOT / "shared-parent.json").exists():
        raise ValueError("A and its shared CP10 parent must complete first")
    return train_admission(label, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--arm", choices=LABELS)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    if args.arm is None:
        raise ValueError("train requires --arm")
    train(args.arm)


if __name__ == "__main__":
    main()
