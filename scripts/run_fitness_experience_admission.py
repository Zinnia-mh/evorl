"""Run Fitness Top16 and Random16 admission from frozen Uniform A CP10."""

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

from evorl.algorithms.erl.erl_td3.erl_utility_replay import _optimizer_step_count
from evorl.algorithms.erl.erl_td3.source_replay import EA_ORIGIN, RL_ORIGIN
from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest
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
from scripts.fitness_experience_admission import (
    FitnessExperienceAdmissionWorkflow,
    select_admitted_individuals,
)


ROOT = Path(
    "/disk1/zmh_projects/evorl_artifacts/"
    "fitness_experience_admission/cp100-v1"
)
FINAL_GENERATION = 100
FINAL_EPISODES = 3290
SEEDS = (3083, 3079)
ARMS = {
    "F-Top16": "fitness_top",
    "F-Random16": "random",
}
PARENTS = {
    3083: Path(
        "/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/"
        "algorithm/s-utility-replication-seed3083-v1/runs/A/segments/"
        "eacf2ffb25574622990fa3e1fe56a8ed/checkpoints/10"
    ),
    3079: Path(
        "/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/"
        "algorithm/budgeted-scoring-v2/pilot/runs/A/segments/"
        "a3adee0a62d344dc8e26be04c6012805/checkpoints/10"
    ),
}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_rows(output):
    files = sorted(
        Path(output).glob("segments/*/admission.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    latest = {}
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            latest[row["generation"]] = row
    return [latest[generation] for generation in sorted(latest)]


def find_resume_checkpoint(output):
    candidates = [
        path
        for path in (Path(output) / "segments").glob("*/checkpoints/*")
        if path.is_dir()
        and path.name.isdigit()
        and (path / "runtime/metadata").exists()
        and (
            path.parent.parent
            / "checkpoint_checksums"
            / f"{path.name}.json"
        ).exists()
    ]
    return max(
        candidates,
        key=lambda path: (int(path.name), path.stat().st_mtime),
        default=None,
    )


def create_bootstrap_checkpoint(segment, config, state, runtime):
    with PilotCheckpointStore(
        segment / "checkpoints", config, interval=25, keep=5
    ) as store:
        checkpoint = store.save(state, runtime, force=True)
    if checkpoint is None or checkpoint.name != "10":
        raise ValueError("failed to seal CP10 bootstrap")


def validate_parent_child(parent, child):
    ignored = {
        "output_dir",
        "arm_manifest_path",
        "total_episodes",
        "admission_mode",
        "admission_count",
    }
    values = []
    for config in (parent, child):
        data = OmegaConf.to_container(config, resolve=True)
        values.append({key: value for key, value in data.items() if key not in ignored})
    if values[0] != values[1]:
        raise ValueError("child configuration differs from frozen A beyond admission controls")


def scalar(value):
    return int(jax.device_get(value))


def tree_equal(left, right):
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


class AdmissionObserver:
    workflow_type = FitnessExperienceAdmissionWorkflow
    audit_before_checkpoint = True
    reconstruct_sample_evidence = False

    def __init__(self, plan, seed, label):
        self.plan = plan
        self.seed = seed
        self.label = label
        self.mode = ARMS[label]
        self.parent = PARENTS[seed]
        self.rows = []
        self.cumulative_writes = 0
        self.identity = {
            "algorithm": label,
            "seed": seed,
            "mode": self.mode,
            "plan_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode()
            ).hexdigest(),
        }

    def configure(self, workflow, segment):
        self.segment = segment
        if (
            int(workflow.config.seed) != self.seed
            or str(workflow.config.admission_mode) != self.mode
        ):
            raise ValueError("observer and workflow configuration differ")

        existing = run_rows(segment.parent.parent)
        if existing:
            self.rows = existing
            self.cumulative_writes = sum(
                row["written_ea_transitions"] + row["written_rl_transitions"]
                for row in existing
            )
            first_writes = (
                existing[0]["written_ea_transitions"]
                + existing[0]["written_rl_transitions"]
            )
            self.parent_index = (
                existing[0]["buffer_current_index"] - first_writes
            ) % existing[0]["buffer_capacity"]
            self.parent_size = existing[0]["buffer_size"] - first_writes

    def fork_state(self, workflow, run_id):
        parent_config = OmegaConf.load(self.parent.parents[3] / "resolved_config.yaml")
        validate_parent_child(parent_config, workflow.config)
        with PilotCheckpointStore(
            self.parent.parent, parent_config, read_only=True
        ) as store:
            state, runtime = store.restore(10, restore_template(workflow))
        metadata = json.loads((self.parent / "runtime/metadata").read_text())
        if (
            metadata["state_sha256"] != initialization_fingerprint(state)
            or runtime["evaluations"]
            or runtime["budget"]["steps"]
            or scalar(state.metrics.iterations) != 10
        ):
            raise ValueError("parent is not the frozen untouched A CP10 state")
        runtime = copy.deepcopy(runtime)
        runtime.update(run_id=run_id, target_generation=FINAL_GENERATION)
        create_bootstrap_checkpoint(self.segment, workflow.config, state, runtime)
        self.parent_state_sha256 = metadata["state_sha256"]
        self.parent_index = scalar(state.replay_buffer_state.current_index)
        self.parent_size = scalar(state.replay_buffer_state.buffer_size)
        return state, runtime

    def before_generation(self, state):
        generation = scalar(state.metrics.iterations)
        if not 10 <= generation < FINAL_GENERATION:
            raise ValueError("generation outside pilot contract")
        if generation == 10:
            self.parent_index = scalar(state.replay_buffer_state.current_index)
            self.parent_size = scalar(state.replay_buffer_state.buffer_size)

    def _sampling_evidence(self, workflow, replay, learner_key, utility, updates):
        trace = jax.device_get(
            reconstruct_replay_draws(
                replay,
                learner_key,
                utility,
                num_updates=min(32, updates),
                num_draws=int(
                    workflow.config.actor_update_interval
                    * workflow.config.num_rl_agents
                ),
                batch_size=int(workflow.config.batch_size),
                uniform=True,
            )
        )
        baseline = np.asarray(trace.baseline_indices)
        final = np.asarray(trace.final_indices)
        if not np.array_equal(baseline, final):
            raise ValueError("Uniform Replay unexpectedly overlaid indices")
        origin = np.asarray(replay.origin_type)
        sampled_origin = origin[baseline]
        if not np.all((sampled_origin == EA_ORIGIN) | (sampled_origin == RL_ORIGIN)):
            raise ValueError("sampled unset Replay provenance")
        return {
            "rounds": int(baseline.shape[0]),
            "ea_samples": int(np.count_nonzero(sampled_origin == EA_ORIGIN)),
            "rl_samples": int(np.count_nonzero(sampled_origin == RL_ORIGIN)),
            "indices_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        }

    def record(self, workflow, state, checkpoint, runtime):
        generation = scalar(state.metrics.iterations)
        admission = workflow.latest_admission
        rl_write = workflow.latest_rl_write
        fitnesses = np.asarray(jax.device_get(admission["fitnesses"]))
        selected = np.asarray(jax.device_get(admission["selected_ids"]))
        if fitnesses.shape != (32,) or not np.all(np.isfinite(fitnesses)):
            raise ValueError("all 32 finite Fitness values were not evaluated")
        expected = np.asarray(jax.device_get(select_admitted_individuals(
            admission["fitnesses"],
            mode=self.mode,
            admission_count=16,
            seed=self.seed,
            generation=generation,
        )))
        np.testing.assert_array_equal(selected, expected)

        all_valid = scalar(admission["all_valid_transitions"])
        ea_written = scalar(admission["written_ea_transitions"])
        rl_written = scalar(rl_write["written_rl_transitions"])
        if all_valid != 32000 or ea_written != 16000 or rl_written != 1000:
            raise ValueError("unexpected real transition count")
        if scalar(admission["after_index"]) != scalar(rl_write["before_index"]):
            raise ValueError("RL insertion did not immediately follow admitted EA insertion")

        self.cumulative_writes += ea_written + rl_written
        replay = state.replay_buffer_state
        capacity = int(workflow.replay_buffer.capacity)
        expected_index = (self.parent_index + self.cumulative_writes) % capacity
        expected_size = min(self.parent_size + self.cumulative_writes, capacity)
        if (
            scalar(replay.current_index) != expected_index
            or scalar(replay.buffer_size) != expected_size
        ):
            raise ValueError("Replay ring pointer or size differs from admitted writes")

        ea_start = scalar(admission["before_index"])
        ea_indices = (ea_start + np.arange(ea_written)) % capacity
        written_tokens = np.asarray(jax.device_get(replay.source_token[ea_indices]))
        written_origins = np.asarray(jax.device_get(replay.origin_type[ea_indices]))
        np.testing.assert_array_equal(written_origins, EA_ORIGIN)
        written_ids = np.unique(written_tokens % 32)
        np.testing.assert_array_equal(written_ids, selected)
        if len(written_tokens) != ea_written or len(written_ids) != 16:
            raise ValueError("admitted EA provenance count differs")

        rl_start = scalar(rl_write["before_index"])
        rl_indices = (rl_start + np.arange(rl_written)) % capacity
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(replay.origin_type[rl_indices])), RL_ORIGIN
        )

        resident = np.arange(capacity) < scalar(replay.buffer_size)
        origin = np.asarray(jax.device_get(replay.origin_type))
        generations = np.asarray(jax.device_get(replay.source_generation))
        resident_ea = resident & (origin == EA_ORIGIN)
        resident_rl = resident & (origin == RL_ORIGIN)
        ea_count = int(resident_ea.sum())
        rl_count = int(resident_rl.sum())
        ages = generation - generations[resident_ea]
        if np.any(ages < 0):
            raise ValueError("future EA generation is resident")

        learner_replay, learner_key, updates, utility = workflow.learner_input
        if updates != 16000 or not workflow.baseline_key_ok:
            raise ValueError("TD3 update budget or training PRNG changed")
        evidence = None
        if generation in (20, FINAL_GENERATION):
            evidence = self._sampling_evidence(
                workflow, learner_replay, learner_key, utility, updates
            )
            with (self.segment / f"uniform-sampling-{generation}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump(evidence, stream, indent=2)

        rejected = np.setdiff1d(np.arange(32), selected)
        selected_fitness = fitnesses[selected]
        rejected_fitness = fitnesses[rejected]
        if self.mode == "fitness_top" and selected_fitness.min() < rejected_fitness.max():
            raise ValueError("Top16 admission differs from Fitness ranking")

        evaluation = runtime["evaluations"][-1] if runtime["evaluations"] else None
        if evaluation is not None and evaluation["generation"] != generation:
            evaluation = None
        row = {
            "generation": generation,
            "mode": self.mode,
            "population_evaluated": 32,
            "selected_ids": selected.tolist(),
            "population_fitness": fitnesses.tolist(),
            "selected_fitness_mean": float(selected_fitness.mean()),
            "selected_fitness_median": float(np.median(selected_fitness)),
            "selected_fitness_min": float(selected_fitness.min()),
            "selected_fitness_max": float(selected_fitness.max()),
            "rejected_fitness_mean": float(rejected_fitness.mean()),
            "rejected_fitness_median": float(np.median(rejected_fitness)),
            "all_valid_ea_transitions": all_valid,
            "written_ea_transitions": ea_written,
            "written_rl_transitions": rl_written,
            "ea_admission_fraction": ea_written / all_valid,
            "buffer_size": scalar(replay.buffer_size),
            "buffer_capacity": capacity,
            "buffer_coverage": scalar(replay.buffer_size) / capacity,
            "buffer_current_index": scalar(replay.current_index),
            "ring_wraps_since_start": (
                self.parent_index + self.cumulative_writes
            ) // capacity,
            "resident_ea_transitions": ea_count,
            "resident_rl_transitions": rl_count,
            "resident_ea_fraction": ea_count / max(1, ea_count + rl_count),
            "resident_rl_fraction": rl_count / max(1, ea_count + rl_count),
            "resident_ea_age_mean": float(ages.mean()),
            "resident_ea_age_median": float(np.median(ages)),
            "resident_ea_age_max": int(ages.max()),
            "actor_optimizer_count": _optimizer_step_count(state.opt_state.actor),
            "critic_optimizer_count": _optimizer_step_count(state.opt_state.critic),
            "training_seconds": workflow.step_seconds,
            "evaluation": evaluation,
            "uniform_sampling_evidence": evidence,
            "rl_injection_generation": generation
            % int(workflow.config.rl_injection_interval)
            == 0,
        }
        with (self.segment / "admission.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        workflow.learner_input = None

    def summary(self):
        return {
            "generations": len(self.rows),
            "ea_transitions_evaluated": sum(
                row["all_valid_ea_transitions"] for row in self.rows
            ),
            "ea_transitions_written": sum(
                row["written_ea_transitions"] for row in self.rows
            ),
            "rl_transitions_written": sum(
                row["written_rl_transitions"] for row in self.rows
            ),
            "mean_admission_fraction": float(np.mean(
                [row["ea_admission_fraction"] for row in self.rows]
            )),
            "final_buffer": {
                key: self.rows[-1][key]
                for key in (
                    "buffer_size",
                    "buffer_capacity",
                    "buffer_current_index",
                    "ring_wraps_since_start",
                    "resident_ea_transitions",
                    "resident_rl_transitions",
                    "resident_ea_fraction",
                    "resident_rl_fraction",
                    "resident_ea_age_mean",
                    "resident_ea_age_median",
                    "resident_ea_age_max",
                )
            },
            "training_step_seconds": sum(
                row["training_seconds"] for row in self.rows
            ),
        }


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    seed_configs = {}
    parent_states = {}
    for seed in SEEDS:
        parent = PARENTS[seed]
        parent_config = OmegaConf.load(parent.parents[3] / "resolved_config.yaml")
        parent_metadata = json.loads((parent / "runtime/metadata").read_text())
        parent_states[str(seed)] = {
            "checkpoint": str(parent),
            "state_sha256": parent_metadata["state_sha256"],
            "mechanism_digest": parent_metadata["mechanism_digest"],
        }
        seed_root = ROOT / f"seed_{seed}"
        protocol = seed_root / "protocol"
        protocol.mkdir(parents=True)
        configs = []
        for arm in ("uniform", "fitness", "td", "utility"):
            config = copy.deepcopy(parent_config)
            config.arm = arm
            config.score_kind = arm
            config.alpha = 0.0 if arm == "uniform" else 0.5
            config.diagnostic_enabled = arm != "uniform"
            config.total_episodes = FINAL_EPISODES
            config.eval_interval = 10
            config.eval_episodes = 128
            config.num_eval_envs = 128
            config.admission_count = 16
            config.admission_mode = "fitness_top"
            config.output_dir = str(seed_root / "runs" / arm)
            config.arm_manifest_path = str(protocol / "arm_manifest.json")
            OmegaConf.save(config, protocol / f"arm_{arm}.yaml", resolve=True)
            configs.append(config)
        prepare_arm_manifest(configs, protocol / "arm_manifest.json")
        seed_configs[str(seed)] = {
            path.name: file_sha(path) for path in protocol.iterdir()
        }

    source_files = [
        Path(__file__),
        Path(__file__).with_name("fitness_experience_admission.py"),
    ]
    plan = {
        "algorithm": "Fitness-Based Experience Admission",
        "base_commit": "6d5b8578fb22f144951327ece9f2c025b052cb2c",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            path.name: file_sha(path) for path in source_files
        },
        "mechanism_digest": mechanism_digest(),
        "parents": parent_states,
        "configuration_hashes": seed_configs,
        "seeds": list(SEEDS),
        "arms": ARMS,
        "primary": "Return AUC CP20-100 over 650000-3290000 environment steps",
        "final_generation": FINAL_GENERATION,
        "evaluation": "every 10 generations, 128 episodes",
        "threshold": "first scheduled mean Return >= 350",
        "admission": "16 of 32 EA individuals; all 32 evaluated",
        "random_stream": (
            "fixed seed/generation fold-in namespace, independent of state.key"
        ),
        "replay": "original Uniform sampling, no weighted replay intervention",
        "no_extra_rollout": True,
        "no_interim_selection": True,
    }
    write_json(ROOT / "preregistration.json", plan)
    return plan


def validate_sources(plan):
    if plan["mechanism_digest"] != mechanism_digest():
        raise ValueError("baseline evorl mechanism changed")
    for name, expected in plan["source_hashes"].items():
        if file_sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"preregistered source changed: {name}")


def smoke():
    import evorl.algorithms.erl.erl_td3.erl_learner_utility as utility_module
    from evorl.algorithms.erl.erl_td3.d0_placebo import ObservedD0

    def forbidden(*args, **kwargs):
        raise AssertionError("Utility path invoked by admission workflow")

    utility_module.build_source_batches = forbidden
    utility_module.virtual_critic_utility = forbidden

    config = OmegaConf.load(ROOT / "seed_3083/protocol/arm_uniform.yaml")
    config.env.max_episode_steps = 8
    config.replay_buffer_capacity = 4096
    config.batch_size = 8
    config.source_batch_size = 4
    config.eval_episodes = 2
    config.num_eval_envs = 2
    config.agent_network.actor_hidden_layer_sizes = [8, 8]
    config.agent_network.critic_hidden_layer_sizes = [8, 8]
    baseline = type("AdmissionSmokeParent", (ObservedD0,), {}).build_from_config(
        config, enable_jit=True
    )
    initial = baseline.init(jax.random.PRNGKey(3083))

    results = {}
    states = {}
    for label, mode in ARMS.items():
        arm_config = copy.deepcopy(config)
        arm_config.admission_mode = mode
        arm_config.admission_count = 16
        workflow = type(
            f"AdmissionSmoke{label}",
            (FitnessExperienceAdmissionWorkflow,),
            {},
        ).build_from_config(arm_config, enable_jit=True)
        metrics, state = workflow.step(initial)
        jax.block_until_ready((metrics, state))
        admission = workflow.latest_admission
        selected = np.asarray(admission["selected_ids"])
        fitnesses = np.asarray(admission["fitnesses"])
        assert fitnesses.shape == (32,)
        expected = np.asarray(select_admitted_individuals(
            admission["fitnesses"],
            mode=mode,
            admission_count=16,
            seed=3083,
            generation=11,
        ))
        np.testing.assert_array_equal(selected, expected)
        assert scalar(admission["all_valid_transitions"]) == 256
        assert scalar(admission["written_ea_transitions"]) == 128
        assert scalar(workflow.latest_rl_write["written_rl_transitions"]) == 8
        replay, learner_key, updates, utility = workflow.learner_input
        trace = jax.device_get(reconstruct_replay_draws(
            replay,
            learner_key,
            utility,
            num_updates=min(8, updates),
            num_draws=2,
            batch_size=8,
            uniform=True,
        ))
        np.testing.assert_array_equal(trace.baseline_indices, trace.final_indices)
        origin = np.asarray(state.replay_buffer_state.origin_type)
        resident = np.arange(len(origin)) < scalar(state.replay_buffer_state.buffer_size)
        assert np.count_nonzero(origin[resident] == EA_ORIGIN) == 2688
        assert np.count_nonzero(origin[resident] == RL_ORIGIN) == 8
        assert workflow.baseline_key_ok
        results[label] = {
            "selected_ids": selected.tolist(),
            "population_fitness": fitnesses.tolist(),
            "all_population_evaluated": len(fitnesses) == 32,
            "written_ea": 128,
            "written_rl": 8,
            "uniform_sample_indices_unchanged": True,
            "training_key": np.asarray(state.key).tolist(),
            "ga_key": np.asarray(state.ec_opt_state.key).tolist(),
        }
        states[label] = state
        workflow.close()

    assert tree_equal(
        states["F-Top16"].ec_opt_state, states["F-Random16"].ec_opt_state
    )
    assert results["F-Top16"]["training_key"] == results["F-Random16"]["training_key"]
    write_json(ROOT / "smoke.json", {
        "status": "PASS",
        "real_brax_gpu": True,
        "utility_calls": 0,
        "all_32_evaluated": True,
        "ga_state_identical_after_generation_11": True,
        "results": results,
    })
    baseline.close()
    print("FITNESS ADMISSION GPU SMOKE PASS", flush=True)


def train(seed, label):
    plan = json.loads((ROOT / "preregistration.json").read_text())
    validate_sources(plan)
    if not (ROOT / "smoke.json").exists():
        raise ValueError("GPU smoke must pass before formal pilot")
    seed_root = ROOT / f"seed_{seed}"
    output = seed_root / "runs" / label
    config = OmegaConf.load(seed_root / "protocol" / "arm_uniform.yaml")
    config.admission_mode = ARMS[label]
    observer = AdmissionObserver(plan, seed, label)
    resume = find_resume_checkpoint(output) if output.exists() else None
    if output.exists() and resume is None:
        raise ValueError("existing run has no sealed resume checkpoint")
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
        raise ValueError("incomplete generation journal")
    result = {
        "algorithm": label,
        "seed": seed,
        "performance": report,
        "mechanism": observer.summary(),
        "process_wall_seconds": time.perf_counter() - started,
        "final_training_key": np.asarray(state.key).tolist(),
        "final_ga_key": np.asarray(state.ec_opt_state.key).tolist(),
        "parent_state_sha256": observer.parent_state_sha256,
    }
    write_json(seed_root / f"summary-{label}.json", result)
    print(
        "COMPLETE",
        seed,
        label,
        "AUC",
        report["auc"],
        "Return100",
        report["final_return"],
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    if jax.default_backend() != "gpu":
        raise ValueError("GPU backend required")
    if args.stage == "smoke":
        smoke()
        return
    if args.seed is None or args.arm is None:
        raise ValueError("train requires --seed and --arm")
    train(args.seed, args.arm)


if __name__ == "__main__":
    main()
