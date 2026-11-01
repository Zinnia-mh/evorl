"""Pilot-only arm and matched-budget validation."""

import math
import hashlib
import json
from pathlib import Path


ARM_SCORE_KIND = {
    "uniform": "uniform",
    "fitness": "fitness",
    "td": "td",
    "utility": "utility",
}

COMMON_FIELDS = (
    "replay_temperature", "freshness_window", "min_resident",
    "diagnostic_interval", "total_episodes", "episodes_for_fitness",
    "pop_size", "rollout_episodes", "num_rl_agents", "batch_size",
    "actor_update_interval", "rl_updates_frac", "rl_updates_mode",
    "eval_interval", "eval_episodes", "num_eval_envs", "num_elites",
    "tournament_size", "rl_injection_interval", "replay_buffer_capacity",
)


def validate_arm_config(config):
    arm = str(config.arm)
    if arm not in ARM_SCORE_KIND:
        raise ValueError(f"unknown utility replay arm: {arm}")
    if str(config.score_kind) != ARM_SCORE_KIND[arm]:
        raise ValueError(f"{arm} arm requires score_kind={ARM_SCORE_KIND[arm]}")
    alpha = float(config.alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite within [0, 1]")
    if arm == "uniform":
        if alpha != 0 or bool(config.diagnostic_enabled):
            raise ValueError("uniform arm requires alpha=0 and diagnostics disabled")
    elif alpha == 0 or not bool(config.diagnostic_enabled):
        raise ValueError("weighted arms require alpha>0 and diagnostics enabled")
    if int(config.freshness_window) < 0 or int(config.min_resident) < 1:
        raise ValueError("invalid common freshness or resident threshold")
    if float(config.replay_temperature) <= 0:
        raise ValueError("replay temperature must be positive")
    return arm


def validate_arm_set(configs):
    by_arm = {validate_arm_config(config): config for config in configs}
    if len(by_arm) != 4 or set(by_arm) != set(ARM_SCORE_KIND):
        raise ValueError("exactly one config for each A/B/C/D arm is required")
    baseline = by_arm["fitness"]
    for arm in ("td", "utility"):
        config = by_arm[arm]
        if float(config.alpha) != float(baseline.alpha):
            raise ValueError("weighted arms must share alpha")
    for arm, config in by_arm.items():
        for field in COMMON_FIELDS:
            if getattr(config, field) != getattr(baseline, field):
                raise ValueError(f"arm-specific override forbidden: {arm}.{field}")
    common = _canonical_config(baseline, exclude_arm=True)
    for arm, config in by_arm.items():
        if _canonical_config(config, exclude_arm=True) != common:
            raise ValueError(f"arm-specific configuration mismatch: {arm}")
    return by_arm


def _canonical_config(config, *, exclude_arm=False):
    try:
        from omegaconf import OmegaConf
        data = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else vars(config).copy()
    except ImportError:
        data = vars(config).copy()
    data = dict(data)
    for key in ("output_dir", "arm_manifest_path"):
        data.pop(key, None)
    if exclude_arm:
        for key in ("arm", "score_kind", "alpha", "diagnostic_enabled"):
            data.pop(key, None)
    return data


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def prepare_arm_manifest(configs, path):
    by_arm = validate_arm_set(configs)
    manifest = {"schema": 1, "common_hash": _digest(_canonical_config(by_arm["fitness"], exclude_arm=True)),
                "arm_hashes": {arm: _digest(_canonical_config(config)) for arm, config in by_arm.items()}}
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
    return manifest


def validate_arm_manifest(config):
    validate_arm_config(config)
    path = getattr(config, "arm_manifest_path", None)
    if not path:
        raise ValueError("formal Pilot launch requires arm_manifest_path")
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != 1 or set(manifest.get("arm_hashes", {})) != set(ARM_SCORE_KIND):
        raise ValueError("invalid four-arm manifest")
    arm = str(config.arm)
    if manifest["arm_hashes"][arm] != _digest(_canonical_config(config)):
        raise ValueError("resolved arm config does not match frozen manifest")
    if manifest["common_hash"] != _digest(_canonical_config(config, exclude_arm=True)):
        raise ValueError("common arm config does not match frozen manifest")
    return manifest


class BudgetAudit:
    """Host-side counters; never enters learner state or training PRNG."""

    def __init__(self, config, *, initial_episodes, initial_iteration, initial_timesteps=0):
        validate_arm_config(config)
        self.config = config
        self.initial_episodes = int(initial_episodes)
        self.initial_iteration = int(initial_iteration)
        self.initial_timesteps = int(initial_timesteps)
        self.episodes = 0
        self.timesteps = 0
        self.updates = 0
        self.steps = 0
        self.ask = 0
        self.tell = 0
        self.checkpoints = []
        self.eval_episodes = 0

    def observe_step(self, *, episodes, timesteps, rl_updates, batch_size, ea_ask, ea_tell,
                     actor_optimizer_steps=None, critic_optimizer_steps=None):
        if int(batch_size) != int(self.config.batch_size):
            raise ValueError("replay batch size mismatch")
        if (int(ea_ask), int(ea_tell)) != (1, 1):
            raise ValueError("EA population operation mismatch")
        expected_episodes = (int(self.config.episodes_for_fitness) * int(self.config.pop_size)
                             + int(self.config.rollout_episodes) * int(self.config.num_rl_agents))
        if int(episodes) != expected_episodes or int(rl_updates) < 0:
            raise ValueError("environment or learner budget mismatch")
        if actor_optimizer_steps is not None and int(actor_optimizer_steps) != int(rl_updates):
            raise ValueError("actual actor optimizer update mismatch")
        if critic_optimizer_steps is not None and int(critic_optimizer_steps) != int(rl_updates) * int(self.config.actor_update_interval):
            raise ValueError("actual critic optimizer update mismatch")
        self.episodes += int(episodes)
        self.timesteps += int(timesteps)
        self.updates += int(rl_updates)
        self.ask += int(ea_ask)
        self.tell += int(ea_tell)
        self.steps += 1

    def observe_eval(self, iteration, *, episodes):
        expected = (math.ceil(int(self.config.eval_episodes) / int(self.config.num_eval_envs))
                    * int(self.config.num_eval_envs) * int(self.config.num_rl_agents))
        if int(episodes) != expected:
            raise ValueError("evaluation episode mismatch")
        self.checkpoints.append(int(iteration))
        self.eval_episodes += int(episodes)

    def finalize(self, *, final_episodes, final_iteration, final_timesteps):
        c = self.config
        per_step = int(c.episodes_for_fitness) * int(c.pop_size) + int(c.rollout_episodes) * int(c.num_rl_agents)
        expected_steps = math.ceil(max(0, int(c.total_episodes) - self.initial_episodes) / per_step)
        expected_checkpoints = [i for i in range(self.initial_iteration + 1, self.initial_iteration + expected_steps + 1)
                                if i % int(c.eval_interval) == 0 or i == self.initial_iteration + expected_steps]
        if self.steps != expected_steps or int(final_iteration) != self.initial_iteration + self.steps:
            raise ValueError("training iteration budget mismatch")
        if int(final_episodes) != self.initial_episodes + self.episodes:
            raise ValueError("actual environment episode counter mismatch")
        if int(final_timesteps) != self.initial_timesteps + self.timesteps:
            raise ValueError("actual environment timestep counter mismatch")
        if self.checkpoints != expected_checkpoints:
            raise ValueError("evaluation checkpoint schedule mismatch")
        if self.ask != self.steps or self.tell != self.steps:
            raise ValueError("EA population operation counter mismatch")
        expected_eval = (math.ceil(int(c.eval_episodes) / int(c.num_eval_envs))
                         * int(c.num_eval_envs) * int(c.num_rl_agents))
        if self.eval_episodes != len(self.checkpoints) * expected_eval:
            raise ValueError("evaluation budget mismatch")
        return {
            "arm": str(c.arm), "score_kind": str(c.score_kind),
            "configured_total_episodes": int(c.total_episodes),
            "environment_episodes": self.episodes,
            "environment_timesteps": self.timesteps,
            "rl_updates": self.updates, "batch_size": int(c.batch_size),
            "ea_ask": self.ask, "ea_tell": self.tell,
            "evaluation_checkpoints": tuple(self.checkpoints),
            "evaluation_episodes": self.eval_episodes,
        }
