"""Read-only CP100 Replay coverage analysis for Fitness admission arms."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
    ERLUtilityReplayWorkflow,
)
from evorl.algorithms.erl.erl_td3.source_replay import EA_ORIGIN, RL_ORIGIN
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import (
    PilotCheckpointStore,
    restore_template,
)


ROOT = Path(
    "/disk1/zmh_projects/evorl_artifacts/fitness_experience_admission/"
    "seed4099-cp100-v1"
)
CHECKPOINTS = {
    "A": ROOT / "runs/A/segments/21933761900d47f9ba7af615a8dcd59a/checkpoints/100",
    "F-Top16": ROOT / "runs/F-Top16/segments/920ee3ff638c46078bbb06fc497e86b3/checkpoints/100",
    "F-Random16": ROOT / "runs/F-Random16/segments/84511a8f3ba148f8a9ca1a6c2dd76591/checkpoints/100",
}
SAMPLE_SEED = 20261127
SAMPLE_SIZE = 100_000
PAIR_COUNT = 50_000


def _array(value):
    return np.asarray(jax.device_get(value))


def _scalar(value):
    return int(jax.device_get(value))


def _restore(label):
    checkpoint = CHECKPOINTS[label]
    config = OmegaConf.load(checkpoint.parents[3] / "resolved_config.yaml")
    workflow = ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=False)
    with PilotCheckpointStore(checkpoint.parent, config, read_only=True) as store:
        state, _ = store.restore(100, restore_template(workflow))
    return workflow, state


def _age_summary(ages):
    values, counts = np.unique(ages, return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return {
        "mean": float(np.mean(ages)),
        "median": float(np.median(ages)),
        "q10": float(np.quantile(ages, 0.10)),
        "q25": float(np.quantile(ages, 0.25)),
        "q75": float(np.quantile(ages, 0.75)),
        "q90": float(np.quantile(ages, 0.90)),
        "max": int(np.max(ages)),
        "age_0_4": int(np.count_nonzero((ages >= 0) & (ages <= 4))),
        "age_5_29": int(np.count_nonzero((ages >= 5) & (ages <= 29))),
        "age_30_plus": int(np.count_nonzero(ages >= 30)),
        "histogram": histogram,
    }


def _load_arm(label, seed):
    workflow, state = _restore(label)
    replay = state.replay_buffer_state
    capacity = int(workflow.replay_buffer.capacity)
    size = _scalar(replay.buffer_size)
    generation = _scalar(state.metrics.iterations)
    resident = np.arange(capacity) < size
    origin = _array(replay.origin_type)
    source_generation = _array(replay.source_generation)
    source_token = _array(replay.source_token)
    ea = resident & (origin == EA_ORIGIN)
    rl = resident & (origin == RL_ORIGIN)
    ea_indices = np.flatnonzero(ea)
    sample_count = min(SAMPLE_SIZE, ea_indices.size)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(ea_indices, size=sample_count, replace=False)

    data = replay.data
    obs = _array(data.obs)[sample_indices]
    actions = _array(data.actions)[sample_indices]
    rewards = _array(data.rewards)[sample_indices, None]
    next_obs = _array(data.extras.env_extras.ori_obs)[sample_indices]
    termination = _array(data.extras.env_extras.termination)[sample_indices, None]
    ages = generation - source_generation[ea]
    result = {
        "generation": generation,
        "buffer_size": size,
        "buffer_capacity": capacity,
        "current_index": _scalar(replay.current_index),
        "resident_ea": int(np.count_nonzero(ea)),
        "resident_rl": int(np.count_nonzero(rl)),
        "ea_fraction": float(np.count_nonzero(ea) / size),
        "rl_fraction": float(np.count_nonzero(rl) / size),
        "ea_unique_generations": int(np.unique(source_generation[ea]).size),
        "ea_unique_sources": int(np.unique(source_token[ea]).size),
        "ea_age": _age_summary(ages),
        "coverage_sample_size": int(sample_count),
    }
    workflow.close()
    return result, {
        "state": obs,
        "action": actions,
        "transition": np.concatenate(
            [obs, actions, rewards, next_obs - obs, termination], axis=1
        ),
    }


def _standardize(samples):
    pooled = np.concatenate(samples, axis=0).astype(np.float64)
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return [(sample - mean) / scale for sample in samples]


def _effective_rank(sample):
    covariance = np.cov(sample, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total == 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _pair_distance(sample, rng):
    left = rng.integers(0, sample.shape[0], size=PAIR_COUNT)
    right = rng.integers(0, sample.shape[0], size=PAIR_COUNT)
    distances = np.linalg.norm(sample[left] - sample[right], axis=1)
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "q10": float(np.quantile(distances, 0.10)),
        "q90": float(np.quantile(distances, 0.90)),
    }


def _projection_occupancy(samples, rng):
    dimension = samples[0].shape[1]
    projection = rng.normal(size=(dimension, 4))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    projected = [sample @ projection for sample in samples]
    pooled = np.concatenate(projected, axis=0)
    edges = [
        np.quantile(pooled[:, column], np.linspace(0.1, 0.9, 9))
        for column in range(4)
    ]
    results = []
    for sample in projected:
        bins = [np.digitize(sample[:, column], edges[column]) for column in range(4)]
        cell = bins[0] + 10 * bins[1] + 100 * bins[2] + 1000 * bins[3]
        counts = np.bincount(cell, minlength=10_000)
        nonzero = counts[counts > 0]
        probabilities = nonzero / nonzero.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        results.append(
            {
                "occupied_cells_of_10000": int(nonzero.size),
                "normalized_entropy": float(entropy / np.log(10_000)),
            }
        )
    return results


def analyze(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")
    labels = list(CHECKPOINTS)
    summaries = {}
    samples = {}
    for offset, label in enumerate(labels):
        summaries[label], samples[label] = _load_arm(label, SAMPLE_SEED + offset)

    for feature in ("state", "action", "transition"):
        standardized = _standardize([samples[label][feature] for label in labels])
        occupancy = _projection_occupancy(
            standardized, np.random.default_rng(SAMPLE_SEED + 100 + len(feature))
        )
        for offset, (label, sample, occupied) in enumerate(
            zip(labels, standardized, occupancy)
        ):
            summaries[label].setdefault("coverage", {})[feature] = {
                "effective_rank": _effective_rank(sample),
                "pair_distance": _pair_distance(
                    sample,
                    np.random.default_rng(SAMPLE_SEED + 200 + 10 * offset + len(feature)),
                ),
                **occupied,
            }

    result = {
        "contract": {
            "source_root": str(ROOT),
            "read_only_checkpoints": {key: str(value) for key, value in CHECKPOINTS.items()},
            "sample_seed": SAMPLE_SEED,
            "equal_ea_sample_size_per_arm": SAMPLE_SIZE,
            "pair_count_per_metric": PAIR_COUNT,
            "transition_feature": "standardized [obs, action, reward, next_obs-obs, termination]",
            "projection_occupancy": "fixed 4D Gaussian projection with pooled decile bins",
        },
        "arms": summaries,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "coverage_analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.output)


if __name__ == "__main__":
    main()
