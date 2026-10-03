"""Exact-equivalent alpha-zero utility replay pilot.

Phase 3 installs provenance sidecars and the replay sampling seam, but it
deliberately contains no scoring, weighting, eligibility, or replacement.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    ReplayProvenance,
    SourceAwareReplayBuffer,
    make_rl_provenance,
)
from evorl.utils.jax_utils import (
    right_shift_with_padding,
    tree_get,
    tree_stop_gradient,
)
from evorl.utils.rl_toolkits import (
    flatten_pop_rollout_episode,
    flatten_rollout_trajectory,
)

from .erl_origin import ERLWorkflow
from .source_replay import draw_baseline_indices, overlay_ea_indices


ALLOWED_PILOT_ONLY_PATHS = (
    "replay_buffer_state.origin_type",
    "replay_buffer_state.source_token",
    "replay_buffer_state.source_generation",
    "replay_buffer_state.score_version",
    "workflow.intervention_key",
    "train_metrics.time_cost_per_iter",
)


class AlphaZeroSample(NamedTuple):
    batch: chex.ArrayTree
    baseline_indices: chex.Array
    final_indices: chex.Array
    intervention_key: chex.PRNGKey
    baseline_draw_count: chex.Array


def alpha_zero_sample(
    replay_buffer_state,
    replay_key: chex.PRNGKey,
    intervention_key: chex.PRNGKey,
    sample_batch_size: int,
) -> AlphaZeroSample:
    """Perform exactly one baseline draw and no intervention random operation."""
    baseline_indices = draw_baseline_indices(
        replay_buffer_state, replay_key, sample_batch_size
    )
    overlay = overlay_ea_indices(
        baseline_indices, replay_buffer_state.origin_type
    )
    batch = tree_get(replay_buffer_state.data, overlay.final_indices)
    return AlphaZeroSample(
        batch=batch,
        baseline_indices=overlay.baseline_indices,
        final_indices=overlay.final_indices,
        intervention_key=intervention_key,
        baseline_draw_count=jnp.asarray(1, dtype=jnp.uint32),
    )


class UtilityReplayBuffer(SourceAwareReplayBuffer):
    """Source-aware replay whose Phase-3 sampler is statically alpha zero."""

    def sample(self, buffer_state, key):
        return alpha_zero_sample(
            buffer_state,
            key,
            jax.random.PRNGKey(0),
            self.sample_batch_size,
        ).batch


def _rollout_episode_with_provenance(
    agent_state,
    replay_buffer_state,
    key,
    *,
    collector,
    replay_buffer,
    agent_state_vmap_axes,
    num_episodes,
    num_agents,
    origin,
):
    """Baseline rollout/flatten path plus index-aligned pilot-only sidecars."""
    eval_metrics, raw_trajectory = jax.vmap(
        collector.rollout,
        in_axes=(agent_state_vmap_axes, 0, None),
    )(
        agent_state,
        jax.random.split(key, num_agents),
        num_episodes,
    )

    raw_rewards = raw_trajectory.rewards
    time_size, episode_batch = raw_rewards.shape[1:3]
    trajectory = raw_trajectory.replace(next_obs=None)
    trajectory = flatten_pop_rollout_episode(trajectory)
    mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
    trajectory = trajectory.replace(dones=None)
    trajectory, mask = tree_stop_gradient(
        flatten_rollout_trajectory((trajectory, mask))
    )

    flat_size = time_size * num_agents * episode_batch
    if origin == "ea":
        individual_ids = jnp.tile(
            jnp.repeat(jnp.arange(num_agents, dtype=jnp.int32), episode_batch),
            time_size,
        )
        provenance = ReplayProvenance(
            origin_type=jnp.full((flat_size,), EA_ORIGIN, dtype=jnp.int8),
            source_token=individual_ids,
            source_generation=jnp.zeros((flat_size,), dtype=jnp.int32),
            score_version=jnp.zeros((flat_size,), dtype=jnp.int32),
        )
    else:
        provenance = make_rl_provenance(batch_size=flat_size)

    replay_buffer_state = replay_buffer.add(
        replay_buffer_state, trajectory, provenance, mask
    )
    return eval_metrics, trajectory, replay_buffer_state


class ERLUtilityReplayWorkflow(ERLWorkflow):
    """Original ERL with a provenance-aware alpha-zero replay seam."""

    @classmethod
    def name(cls):
        return "ERL-Utility-Replay"

    @classmethod
    def _build_from_config(cls, config):
        if float(config.alpha) != 0.0:
            raise NotImplementedError("Phase 3 supports only static alpha=0")
        workflow = super()._build_from_config(config)
        baseline = workflow.replay_buffer
        workflow.replay_buffer = UtilityReplayBuffer(
            capacity=baseline.capacity,
            min_sample_timesteps=baseline.min_sample_timesteps,
            sample_batch_size=baseline.sample_batch_size,
        )
        workflow.intervention_key = jax.random.PRNGKey(0)
        return workflow

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        return _rollout_episode_with_provenance(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.ec_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.pop_size,
            num_episodes=self.config.episodes_for_fitness,
            origin="ea",
        )

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        return _rollout_episode_with_provenance(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.rl_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.num_rl_agents,
            num_episodes=self.config.rollout_episodes,
            origin="rl",
        )
