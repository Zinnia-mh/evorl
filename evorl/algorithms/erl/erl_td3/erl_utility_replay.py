"""Exact-equivalent alpha-zero utility replay pilot.

Phase 3 installs provenance sidecars and the replay sampling seam, but it
deliberately contains no scoring, weighting, eligibility, or replacement.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    NO_SCORE_VERSION,
    ReplayProvenance,
    SourceAwareReplayBuffer,
    encode_source_token,
    make_rl_provenance,
)
from evorl.utils.jax_utils import tree_get

from .erl_origin import ERLWorkflow
from .erl_td3_workflow import ERLTD3WorkflowTemplate
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

    def add(self, buffer_state, xs, provenance_or_mask=None, mask=None):
        """Accept both the baseline and provenance-aware add signatures."""
        if isinstance(provenance_or_mask, ReplayProvenance):
            return super().add(buffer_state, xs, provenance_or_mask, mask)
        if mask is not None:
            raise TypeError("baseline add accepts its mask as the third argument")
        return ReplayBuffer.add(self, buffer_state, xs, provenance_or_mask)


def make_ea_rollout_provenance(
    *, generation: int, population_size: int, time_size: int, episode_batch: int
) -> ReplayProvenance:
    """Encode authoritative generation and individual identity for an EA rollout."""
    generation = int(generation)
    population_size = int(population_size)
    time_size = int(time_size)
    episode_batch = int(episode_batch)
    base_token = encode_source_token(generation, 0, population_size)
    encode_source_token(generation, population_size - 1, population_size)
    individual_ids = jnp.tile(
        jnp.repeat(jnp.arange(population_size, dtype=jnp.int32), episode_batch),
        time_size,
    )
    flat_size = time_size * population_size * episode_batch
    return ReplayProvenance(
        origin_type=jnp.full((flat_size,), EA_ORIGIN, dtype=jnp.int8),
        source_token=jnp.asarray(base_token, dtype=jnp.int32) + individual_ids,
        source_generation=jnp.full(
            (flat_size,), generation, dtype=jnp.int32
        ),
        score_version=jnp.full(
            (flat_size,), NO_SCORE_VERSION, dtype=jnp.int32
        ),
    )


class ERLUtilityReplayWorkflow(ERLWorkflow):
    """Original ERL with a provenance-aware alpha-zero replay seam."""

    _baseline_ec_rollout = ERLTD3WorkflowTemplate._ec_rollout
    _baseline_rl_rollout = ERLTD3WorkflowTemplate._rl_rollout

    @classmethod
    def name(cls):
        return "ERL-Utility-Replay"

    @classmethod
    def enable_jit(cls) -> None:
        """JIT the unchanged baseline cores; keep sidecar writes host-scoped."""
        cls._baseline_rl_rollout = jax.jit(
            ERLTD3WorkflowTemplate._rl_rollout, static_argnums=(0,)
        )
        cls._baseline_ec_rollout = jax.jit(
            ERLTD3WorkflowTemplate._ec_rollout, static_argnums=(0,)
        )
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))
        cls._ec_update_with_rl_injection = jax.jit(
            cls._ec_update_with_rl_injection, static_argnums=(0,)
        )
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )

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

    def _generation_from_state(self, state) -> int:
        """Use the frozen diagnostic-v1 definition of the active generation."""
        return int(jax.device_get(state.metrics.iterations)) + 1

    def _with_generation_context(self, state, operation):
        self._provenance_generation = self._generation_from_state(state)
        try:
            return operation(state)
        finally:
            del self._provenance_generation

    def step(self, state):
        return self._with_generation_context(state, super().step)

    def warmup_step(self, state):
        return self._with_generation_context(state, super().warmup_step)

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        start_index = replay_buffer_state.current_index
        metrics, trajectory, replay_buffer_state = self._baseline_ec_rollout(
            agent_state, replay_buffer_state, key
        )
        replay_buffer_state = self._write_rollout_provenance(
            replay_buffer_state,
            metrics,
            trajectory,
            start_index=start_index,
            num_agents=self.config.pop_size,
            generation=self._provenance_generation,
            origin="ea",
        )
        return metrics, trajectory, replay_buffer_state

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        start_index = replay_buffer_state.current_index
        metrics, trajectory, replay_buffer_state = self._baseline_rl_rollout(
            agent_state, replay_buffer_state, key
        )
        replay_buffer_state = self._write_rollout_provenance(
            replay_buffer_state,
            metrics,
            trajectory,
            start_index=start_index,
            num_agents=self.config.num_rl_agents,
            generation=0,
            origin="rl",
        )
        return metrics, trajectory, replay_buffer_state

    def _write_rollout_provenance(
        self,
        replay_buffer_state,
        metrics,
        trajectory,
        *,
        start_index,
        num_agents,
        generation,
        origin,
    ):
        """Align sidecars to the baseline core's already-written transitions."""
        flat_size = jtu.tree_leaves(trajectory)[0].shape[0]
        episode_batch = metrics.episode_lengths.shape[-1]
        time_size = flat_size // (num_agents * episode_batch)
        mask = (
            jnp.arange(time_size)[:, None, None]
            < metrics.episode_lengths[None, :, :]
        ).reshape(-1)
        provenance = (
            make_ea_rollout_provenance(
                generation=generation,
                population_size=num_agents,
                time_size=time_size,
                episode_batch=episode_batch,
            )
            if origin == "ea"
            else make_rl_provenance(batch_size=flat_size)
        )
        return self.replay_buffer.add_provenance(
            replay_buffer_state,
            provenance,
            mask,
            current_index=start_index,
        )
