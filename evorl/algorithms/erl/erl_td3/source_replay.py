"""Pure baseline-draw overlay primitives for the replay pilot.

Phase 2 only identifies realized EA/RL slots.  It deliberately performs no
replacement, weighting, score lookup, or intervention random draw.
"""

from typing import NamedTuple

import chex
import jax

from evorl.replay_buffers.source_aware_replay_buffer import EA_ORIGIN, RL_ORIGIN


class ReplayOverlay(NamedTuple):
    """The unchanged baseline draw and its exact origin slot masks."""

    baseline_indices: chex.Array
    final_indices: chex.Array
    ea_slot_mask: chex.Array
    rl_slot_mask: chex.Array


def draw_baseline_indices(
    replay_buffer_state, key: chex.PRNGKey, sample_batch_size: int
) -> chex.Array:
    """Reproduce ``ReplayBuffer.sample`` index generation exactly."""
    return jax.random.randint(
        key,
        (sample_batch_size,),
        minval=0,
        maxval=replay_buffer_state.buffer_size,
    )


def build_origin_slot_masks(
    origin_type: chex.Array, baseline_indices: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Look up the exact realized EA and RL positions in a baseline draw."""
    sampled_origins = origin_type[baseline_indices]
    return sampled_origins == EA_ORIGIN, sampled_origins == RL_ORIGIN


def overlay_ea_indices(
    baseline_indices: chex.Array, origin_type: chex.Array
) -> ReplayOverlay:
    """Build the Phase-2 overlay seam without replacing any transition."""
    ea_slot_mask, rl_slot_mask = build_origin_slot_masks(
        origin_type, baseline_indices
    )
    return ReplayOverlay(
        baseline_indices=baseline_indices,
        final_indices=baseline_indices,
        ea_slot_mask=ea_slot_mask,
        rl_slot_mask=rl_slot_mask,
    )
