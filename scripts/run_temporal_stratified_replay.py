"""Run the fixed seed-5119 A/B/C Temporal Stratified Replay experiment."""

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.temporal_replay import (
    MODE_ORIGINAL,
    MODE_RECENT_30,
    MODE_STRATIFIED_5_25,
    TemporalStratifiedReplayWorkflow,
)
from evorl.algorithms.erl.erl_td3.utility_replay_budget import BudgetAudit
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import (
    PilotCheckpointStore,
    mechanism_digest,
    restore_template,
)
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import (
    initialization_fingerprint,
    reconstruct_replay_draws,
)
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
from evorl.replay_buffers.source_aware_replay_buffer import EA_ORIGIN, RL_ORIGIN


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    "/disk1/zmh_projects/evorl_artifacts/temporal_stratified_replay"
)
SEED = 5119
ARMS = {
    "A": MODE_ORIGINAL,
    "B": MODE_RECENT_30,
    "C": MODE_STRATIFIED_5_25,
}
PROFILES = {
    "smoke": {
        "root": ARTIFACT_ROOT / "smoke-seed5119-cp31-v1",
        "final_generation": 31,
        "total_episodes": 1013,
        "eval_episodes": 4,
        "checkpoint_interval": 10,
        "keep": 5,
    },
    "pilot": {
        "root": ARTIFACT_ROOT / "seed5119-cp100-v1",
        "final_generation": 100,
        "total_episodes": 3290,
        "eval_episodes": 128,
        "checkpoint_interval": 25,
        "keep": 5,
    },
}


def _json(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=_json)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_config(config, *, common=False):
    data = dict(OmegaConf.to_container(config, resolve=True))
    for key in ("output_dir", "arm_manifest_path"):
        data.pop(key, None)
    if common:
        data.pop("temporal_mode", None)
    return data


def config_digest(config, *, common=False):
    payload = json.dumps(
        canonical_config(config, common=common),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def make_base_config(profile):
    settings = PROFILES[profile]
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "agent=erl/erl-temporal-stratified-replay",
                "env=brax/swimmer",
                f"seed={SEED}",
                f"total_episodes={settings['total_episodes']}",
                "eval_interval=10",
                f"eval_episodes={settings['eval_episodes']}",
                f"num_eval_envs={settings['eval_episodes']}",
                "checkpoint.enable=false",
                "save_replay_buffer=true",
            ],
        )
    return config


def prepare(profile):
    settings = PROFILES[profile]
    root = settings["root"]
    root.mkdir(parents=True, exist_ok=False)
    config_dir = root / "configs"
    config_dir.mkdir()
    configs = {}
    for label, mode in ARMS.items():
        config = make_base_config(profile)
        config.temporal_mode = mode
        config.output_dir = str(root / "runs" / label)
        config.arm_manifest_path = str(root / "arm_manifest.json")
        OmegaConf.save(config, config_dir / f"arm_{label}.yaml", resolve=True)
        configs[label] = config
    common = {config_digest(config, common=True) for config in configs.values()}
    if len(common) != 1:
        raise ValueError("A/B/C differ beyond temporal_mode")
    manifest = {
        "schema": 1,
        "kind": "temporal-stratified-replay-abc",
        "common_hash": next(iter(common)),
        "arm_hashes": {label: config_digest(config) for label, config in configs.items()},
    }
    write_json(root / "arm_manifest.json", manifest)
    sources = [
        Path(__file__),
        REPO / "evorl/algorithms/erl/erl_td3/temporal_replay.py",
        REPO / "evorl/algorithms/erl/erl_td3/erl_utility_replay.py",
        REPO / "evorl/algorithms/erl/erl_td3/utility_replay_training.py",
    ]
    preregistration = {
        "experiment": "Temporal Stratified Replay A/B/C",
        "profile": profile,
        "seed": SEED,
        "seed_precommitted": True,
        "seed_history_check": "no structured seed=5119 artifact record found before preparation",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mechanism_digest": mechanism_digest(),
        "source_hashes": {str(path.relative_to(REPO)): file_sha(path) for path in sources},
        "arms": ARMS,
        "shared_parent": "all arms fork from the newly generated original-A CP10 full state",
        "final_generation": settings["final_generation"],
        "primary": "Return AUC CP20-100" if profile == "pilot" else "mechanism smoke only",
        "primary_contrast": "C minus B",
        "fixed_windows": {"recent": 5, "history": 25, "mass": [0.5, 0.5]},
        "no_parameter_search": True,
        "no_interim_selection": True,
    }
    write_json(root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2), flush=True)


def load_plan(profile):
    root = PROFILES[profile]["root"]
    plan = json.loads((root / "preregistration.json").read_text())
    if plan["seed"] != SEED or plan["profile"] != profile:
        raise ValueError("preregistered seed/profile changed")
    if plan["mechanism_digest"] != mechanism_digest():
        raise ValueError("training implementation changed after preregistration")
    for relative, expected in plan["source_hashes"].items():
        if file_sha(REPO / relative) != expected:
            raise ValueError(f"preregistered source changed: {relative}")
    return plan


def load_config(profile, label):
    return OmegaConf.load(PROFILES[profile]["root"] / "configs" / f"arm_{label}.yaml")


class TemporalBudgetAudit(BudgetAudit):
    @classmethod
    def _validate_config(cls, config):
        del cls
        TemporalStratifiedReplayWorkflow._validate_replay_config(config)


class ObservedTemporalWorkflow(TemporalStratifiedReplayWorkflow):
    def _weighted_rl_update(self, agent, optimizer, replay, key, num_updates, utility):
        self.learner_input = (replay, key, int(num_updates), utility)
        return super()._weighted_rl_update(
            agent, optimizer, replay, key, num_updates, utility
        )

    def _rl_update(self, agent, optimizer, replay, key, num_updates):
        self.learner_input = (replay, key, int(num_updates), self.prior_utility)
        return super()._rl_update(agent, optimizer, replay, key, num_updates)

    def step(self, state):
        self.prior_utility = state.utility_replay_state
        self.learner_input = None
        started = time.perf_counter()
        metrics, after = super().step(state)
        jax.block_until_ready((metrics, after))
        self.step_seconds = time.perf_counter() - started
        self.baseline_key_ok = bool(
            np.array_equal(np.asarray(after.key), np.asarray(jax.random.split(state.key, 4)[0]))
        )
        self.prior_utility = None
        return metrics, after


def create_bootstrap_checkpoint(segment, config, state, runtime, settings):
    with PilotCheckpointStore(
        segment / "checkpoints",
        config,
        interval=settings["checkpoint_interval"],
        keep=settings["keep"],
    ) as store:
        checkpoint = store.save(state, runtime, force=True)
    if checkpoint is None or checkpoint.name != "10":
        raise ValueError("failed to seal CP10 bootstrap")


def find_resume_checkpoint(output):
    candidates = [
        path
        for path in (Path(output) / "segments").glob("*/checkpoints/*")
        if path.is_dir()
        and path.name.isdigit()
        and (path / "runtime/metadata").exists()
        and (path.parent.parent / "checkpoint_checksums" / f"{path.name}.json").exists()
    ]
    return max(candidates, key=lambda path: (int(path.name), path.stat().st_mtime), default=None)


def run_rows(output):
    latest = {}
    for path in sorted(Path(output).glob("segments/*/temporal.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            latest[row["generation"]] = row
    return [latest[generation] for generation in sorted(latest)]


class TemporalObserver:
    workflow_type = ObservedTemporalWorkflow
    budget_audit_type = TemporalBudgetAudit
    audit_before_checkpoint = True
    reconstruct_sample_evidence = False

    def __init__(self, profile, label, parent=None):
        self.profile = profile
        self.label = label
        self.mode = ARMS[label]
        self.parent = None if parent is None else Path(parent)
        plan = json.loads((PROFILES[profile]["root"] / "preregistration.json").read_text())
        self.identity = {
            "experiment": "temporal-stratified-replay",
            "profile": profile,
            "arm": label,
            "mode": self.mode,
            "seed": SEED,
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        }

    def validate_config(self, config):
        TemporalStratifiedReplayWorkflow._validate_replay_config(config)
        if str(config.temporal_mode) != self.mode or int(config.seed) != SEED:
            raise ValueError("observer/config arm mismatch")
        manifest = json.loads(Path(config.arm_manifest_path).read_text())
        if manifest.get("schema") != 1 or set(manifest.get("arm_hashes", {})) != set(ARMS):
            raise ValueError("invalid temporal arm manifest")
        if manifest["arm_hashes"][self.label] != config_digest(config):
            raise ValueError("resolved temporal config differs from manifest")
        if manifest["common_hash"] != config_digest(config, common=True):
            raise ValueError("temporal common config mismatch")

    def configure(self, workflow, segment):
        self.segment = Path(segment)
        self.rows = run_rows(segment.parent.parent)
        self.stream = (self.segment / "temporal.jsonl").open("x", encoding="utf-8")

    def fork_state(self, workflow, run_id):
        if self.parent is None:
            raise ValueError("fork arm requires A CP10 parent")
        parent_config = OmegaConf.load(self.parent.parents[3] / "resolved_config.yaml")
        parent_common = canonical_config(parent_config, common=True)
        child_common = canonical_config(workflow.config, common=True)
        if parent_common != child_common:
            raise ValueError("child differs from A beyond temporal_mode")
        with PilotCheckpointStore(self.parent.parent, parent_config, read_only=True) as store:
            state, runtime = store.restore(10, restore_template(workflow))
        metadata = json.loads((self.parent / "runtime/metadata").read_text())
        if (
            metadata["state_sha256"] != initialization_fingerprint(state)
            or runtime["evaluations"]
            or runtime["budget"]["steps"]
            or int(state.metrics.iterations) != 10
        ):
            raise ValueError("parent is not untouched original A CP10")
        runtime = copy.deepcopy(runtime)
        runtime.update(
            run_id=run_id,
            target_generation=PROFILES[self.profile]["final_generation"],
        )
        create_bootstrap_checkpoint(
            self.segment, workflow.config, state, runtime, PROFILES[self.profile]
        )
        self.parent_state_sha256 = metadata["state_sha256"]
        return state, runtime

    def before_generation(self, state):
        generation = int(state.metrics.iterations)
        final = PROFILES[self.profile]["final_generation"]
        if not 10 <= generation < final:
            raise ValueError("generation outside temporal experiment contract")
        replay = state.replay_buffer_state
        self.before_index = int(replay.current_index)
        self.before_size = int(replay.buffer_size)

    def record(self, workflow, state, checkpoint, runtime):
        generation = int(state.metrics.iterations)
        replay = state.replay_buffer_state
        capacity = int(workflow.replay_buffer.capacity)
        after_index = int(replay.current_index)
        write_count = (after_index - self.before_index) % capacity
        if write_count != 33_000:
            raise ValueError(f"expected 33000 writes, observed {write_count}")
        indices = (self.before_index + np.arange(33_000, dtype=np.int64)) % capacity
        device_indices = jnp.asarray(indices, dtype=jnp.int32)
        origin = np.asarray(jax.device_get(replay.origin_type[device_indices]))
        source_generation = np.asarray(
            jax.device_get(replay.source_generation[device_indices])
        )
        if not np.all(origin[:32_000] == EA_ORIGIN):
            raise ValueError("not all 32000 EA transitions were written")
        if not np.all(source_generation[:32_000] == generation):
            raise ValueError("EA generation provenance mismatch")
        if not np.all(origin[32_000:] == RL_ORIGIN):
            raise ValueError("not all 1000 RL transitions were written")
        if not workflow.baseline_key_ok or workflow.learner_input is None:
            raise ValueError("baseline PRNG or learner-input audit failed")

        learner_replay, learner_key, updates, utility = workflow.learner_input
        started = time.perf_counter()
        trace = jax.device_get(
            reconstruct_replay_draws(
                learner_replay,
                learner_key,
                utility,
                num_updates=min(8, updates),
                num_draws=int(workflow.config.actor_update_interval * workflow.config.num_rl_agents),
                batch_size=int(workflow.config.batch_size),
                uniform=self.label == "A",
            )
        )
        reconstruction_seconds = time.perf_counter() - started
        baseline = np.asarray(trace.baseline_indices)
        final = np.asarray(trace.final_indices)
        ea = np.asarray(trace.ea_slot_mask, dtype=bool)
        rl = np.asarray(trace.rl_slot_mask, dtype=bool)
        if not np.array_equal(final[rl], baseline[rl]):
            raise ValueError("RL replay indices changed")
        final_origin = np.asarray(
            jax.device_get(learner_replay.origin_type[jnp.asarray(final)])
        )
        if not np.all(np.where(ea, final_origin == EA_ORIGIN, final_origin == RL_ORIGIN)):
            raise ValueError("overlay sampled an invalid origin")
        sampled_generations = np.asarray(
            jax.device_get(
                learner_replay.source_generation[jnp.asarray(final[ea])]
            )
        )
        ages = generation - sampled_generations
        age_counts = {
            "recent_0_4": int(np.count_nonzero((ages >= 0) & (ages <= 4))),
            "history_5_29": int(np.count_nonzero((ages >= 5) & (ages <= 29))),
            "age_30_plus": int(np.count_nonzero(ages >= 30)),
            "negative": int(np.count_nonzero(ages < 0)),
        }
        lifecycle = getattr(workflow, "_latest_score_lifecycle_audit", {})
        recent_count = int(lifecycle.get("temporal_recent_count", 0))
        history_count = int(lifecycle.get("temporal_history_count", 0))
        window_count = recent_count + history_count
        if bool(utility.distribution_prepared):
            if self.mode == MODE_STRATIFIED_5_25:
                target_recent_mass, target_history_mass = 0.5, 0.5
            else:
                target_recent_mass = recent_count / window_count
                target_history_mass = history_count / window_count
        else:
            target_recent_mass = target_history_mass = 0.0
        row = {
            "generation": generation,
            "arm": self.label,
            "mode": self.mode,
            "writes": {"ea": 32_000, "rl": 1_000, "total": write_count},
            "buffer_size": int(replay.buffer_size),
            "buffer_current_index": after_index,
            "learner_updates": updates,
            "sampled_batches": int(baseline.shape[0] * baseline.shape[1]),
            "sampled_ea_slots": int(ea.sum()),
            "sampled_rl_slots": int(rl.sum()),
            "changed_ea_indices": int(np.count_nonzero((baseline != final) & ea)),
            "changed_rl_indices": int(np.count_nonzero((baseline != final) & rl)),
            "sampled_ea_age_counts": age_counts,
            "temporal_prepared": bool(utility.distribution_prepared),
            "learner_intervention_key": np.asarray(
                jax.random.key_data(utility.intervention_key)
            ).tolist(),
            "post_intervention_key": np.asarray(
                jax.random.key_data(state.utility_replay_state.intervention_key)
            ).tolist(),
            "target_recent_mass": target_recent_mass,
            "target_history_mass": target_history_mass,
            "step_seconds": float(workflow.step_seconds),
            "distribution_refresh_seconds": float(lifecycle.get("distribution_refresh_seconds", 0.0)),
            "weighted_learner_phase_seconds": float(lifecycle.get("weighted_learner_phase_seconds", 0.0)),
            "audit_reconstruction_seconds": reconstruction_seconds,
            "checkpoint": checkpoint,
        }
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        self.rows.append(row)
        workflow.learner_input = None

    def summary(self):
        rows = self.rows
        active = [row for row in rows if row["temporal_prepared"]]
        sampled = sum(row["sampled_ea_slots"] for row in active)
        recent = sum(row["sampled_ea_age_counts"]["recent_0_4"] for row in active)
        history = sum(row["sampled_ea_age_counts"]["history_5_29"] for row in active)
        return {
            "generations": len(rows),
            "active_generations": len(active),
            "ea_transitions_written": 32_000 * len(rows),
            "rl_transitions_written": 1_000 * len(rows),
            "sampled_ea_slots_active": sampled,
            "actual_recent_fraction_active": recent / sampled if sampled else None,
            "actual_history_fraction_active": history / sampled if sampled else None,
            "changed_ea_indices": sum(row["changed_ea_indices"] for row in rows),
            "changed_rl_indices": sum(row["changed_rl_indices"] for row in rows),
            "training_step_seconds": sum(row["step_seconds"] for row in rows),
            "distribution_refresh_seconds": sum(row["distribution_refresh_seconds"] for row in rows),
            "weighted_learner_phase_seconds": sum(row["weighted_learner_phase_seconds"] for row in rows),
            "audit_reconstruction_seconds": sum(row["audit_reconstruction_seconds"] for row in rows),
        }


def train(profile, label, *, stop_after_generation=None):
    if jax.default_backend() != "gpu":
        raise ValueError("GPU backend required")
    plan = load_plan(profile)
    root = PROFILES[profile]["root"]
    output = root / "runs" / label
    summary_path = root / f"summary-{label}.json"
    if summary_path.exists():
        print(f"ALREADY COMPLETE {profile} {label}", flush=True)
        return json.loads(summary_path.read_text())
    parent = None
    if label != "A":
        parent_info = json.loads((root / "shared-parent.json").read_text())
        parent = Path(parent_info["checkpoint"])
    observer = TemporalObserver(profile, label, parent)
    resume = find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:
        raise ValueError("existing run has no sealed checkpoint")
    started = time.perf_counter()
    state, report = run_training(
        load_config(profile, label),
        output,
        checkpoint_interval=PROFILES[profile]["checkpoint_interval"],
        keep=PROFILES[profile]["keep"],
        ablation=observer,
        fork=label != "A" and resume is None,
        resume_checkpoint=resume,
        stop_after_generation=stop_after_generation,
    )
    result = {
        "profile": profile,
        "algorithm": label,
        "mode": ARMS[label],
        "seed": SEED,
        "performance": report,
        "mechanism": observer.summary(),
        "process_wall_seconds": time.perf_counter() - started,
        "final_training_key": np.asarray(state.key).tolist(),
        "final_ga_key": np.asarray(state.ec_opt_state.key).tolist(),
        "state_sha256": initialization_fingerprint(state),
        "parent_state_sha256": getattr(observer, "parent_state_sha256", None),
        "plan_commit": plan["commit"],
    }
    if report["status"] != "COMPLETE":
        print(
            "PAUSED", profile, label, "generation", report["final"]["generation"],
            flush=True,
        )
        return result
    write_json(summary_path, result)
    if label == "A":
        checkpoint = Path(report["segment_directory"]) / "checkpoints" / "10"
        metadata = json.loads((checkpoint / "runtime/metadata").read_text())
        write_json(
            root / "shared-parent.json",
            {"checkpoint": str(checkpoint), "state_sha256": metadata["state_sha256"], "seed": SEED},
        )
    print(
        "COMPLETE", profile, label, "AUC", report["auc"],
        "Return", report["final_return"], flush=True,
    )
    return result


def first_threshold(rows, threshold=350.0):
    for row in rows:
        if row["return_mean"] >= threshold:
            return {
                "generation": row["generation"],
                "environment_timesteps": row["environment_timesteps"],
                "return_mean": row["return_mean"],
            }
    return None


def report(profile):
    root = PROFILES[profile]["root"]
    summaries = {label: json.loads((root / f"summary-{label}.json").read_text()) for label in ARMS}
    rows = {
        label: summaries[label]["performance"]["evaluations"] for label in ARMS
    }
    comparison = {
        "status": "COMPLETE",
        "profile": profile,
        "seed": SEED,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arms": {
            label: {
                "auc": summaries[label]["performance"]["auc"],
                "return_final": summaries[label]["performance"]["final_return"],
                "first350": first_threshold(rows[label]),
                "process_wall_seconds": summaries[label]["process_wall_seconds"],
                "mechanism": summaries[label]["mechanism"],
            }
            for label in ARMS
        },
    }
    if profile == "pilot":
        auc_b = comparison["arms"]["B"]["auc"]
        auc_c = comparison["arms"]["C"]["auc"]
        auc_a = comparison["arms"]["A"]["auc"]
        comparison["contrasts_percent"] = {
            "C_vs_B": (auc_c / auc_b - 1.0) * 100.0,
            "B_vs_A": (auc_b / auc_a - 1.0) * 100.0,
            "C_vs_A": (auc_c / auc_a - 1.0) * 100.0,
        }
    write_json(root / "comparison.json", comparison)
    with (root / "learning_curves.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", *rows["A"][0].keys()])
        writer.writeheader()
        for label in ARMS:
            for row in rows[label]:
                writer.writerow({"arm": label, **row})
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "report"))
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--stop-after-generation", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.profile)
    elif args.stage == "train":
        if args.arm is None:
            raise ValueError("train requires --arm")
        train(
            args.profile,
            args.arm,
            stop_after_generation=args.stop_after_generation,
        )
    else:
        report(args.profile)


if __name__ == "__main__":
    main()
