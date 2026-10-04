"""Pilot-only replay provenance sidecars.

This module preserves the payload and ring-buffer behavior of ``ReplayBuffer``
while writing fixed-shape provenance arrays at the exact same resident indices.
It intentionally contains no source scores, eligibility, or sampling policy.
"""

import chex
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from evorl.replay_buffers.replay_buffer import ReplayBuffer, ReplayBufferState
from evorl.types import PyTreeData
from evorl.utils.jax_utils import tree_set


UNSET_ORIGIN = -1
EA_ORIGIN = 0
RL_ORIGIN = 1

UNSET_SOURCE_TOKEN = -1
RL_SOURCE_TOKEN = -1
UNSET_SOURCE_GENERATION = -1
RL_SOURCE_GENERATION = -1
UNSET_SCORE_VERSION = -1
NO_SCORE_VERSION = -1
RL_SCORE_VERSION = -1


class ReplayProvenance(PyTreeData):
    """Per-transition provenance values supplied alongside a payload batch."""

    origin_type: chex.Array
    source_token: chex.Array
    source_generation: chex.Array
    score_version: chex.Array


class SourceAwareReplayBufferState(ReplayBufferState):
    """Baseline replay state plus index-aligned, fixed-shape sidecars."""

    origin_type: chex.Array
    source_token: chex.Array
    source_generation: chex.Array
    score_version: chex.Array


def encode_source_token(generation: int, individual_id: int, population_size: int) -> int:
    """Encode an EA source identity, rejecting invalid or int32-overflowing inputs."""
    generation = int(generation)
    individual_id = int(individual_id)
    population_size = int(population_size)
    if generation < 0:
        raise ValueError("generation must be non-negative")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if individual_id < 0 or individual_id >= population_size:
        raise ValueError("individual_id must be in [0, population_size)")

    token = generation * population_size + individual_id
    if token > np.iinfo(np.int32).max:
        raise ValueError("source_token exceeds signed int32 range")
    return token


def make_ea_provenance(
    *,
    generation: int,
    individual_id: int,
    population_size: int,
    score_version: int,
    batch_size: int,
) -> ReplayProvenance:
    """Build constant EA provenance for one source batch."""
    token = encode_source_token(generation, individual_id, population_size)
    batch_size = int(batch_size)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if score_version < 0 or score_version > np.iinfo(np.int32).max:
        raise ValueError("score_version must be in signed int32 non-negative range")
    return ReplayProvenance(
        origin_type=jnp.full((batch_size,), EA_ORIGIN, dtype=jnp.int8),
        source_token=jnp.full((batch_size,), token, dtype=jnp.int32),
        source_generation=jnp.full((batch_size,), generation, dtype=jnp.int32),
        score_version=jnp.full((batch_size,), score_version, dtype=jnp.int32),
    )


def make_rl_provenance(*, batch_size: int) -> ReplayProvenance:
    """Build explicit neutral provenance for RL-origin transitions."""
    batch_size = int(batch_size)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    return ReplayProvenance(
        origin_type=jnp.full((batch_size,), RL_ORIGIN, dtype=jnp.int8),
        source_token=jnp.full((batch_size,), RL_SOURCE_TOKEN, dtype=jnp.int32),
        source_generation=jnp.full(
            (batch_size,), RL_SOURCE_GENERATION, dtype=jnp.int32
        ),
        score_version=jnp.full((batch_size,), RL_SCORE_VERSION, dtype=jnp.int32),
    )


def sidecar_nbytes(capacity: int) -> int:
    """Return theoretical raw bytes for the four registered provenance arrays."""
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    bytes_per_slot = sum(
        np.dtype(dtype).itemsize for dtype in (np.int8, np.int32, np.int32, np.int32)
    )
    return capacity * bytes_per_slot


class SourceAwareReplayBuffer(ReplayBuffer):
    """ReplayBuffer variant that writes payload and provenance in lockstep."""

    def init(self, spec: chex.ArrayTree) -> SourceAwareReplayBufferState:
        baseline = ReplayBuffer.init(self, spec)
        return SourceAwareReplayBufferState(
            data=baseline.data,
            current_index=baseline.current_index,
            buffer_size=baseline.buffer_size,
            origin_type=jnp.full((self.capacity,), UNSET_ORIGIN, dtype=jnp.int8),
            source_token=jnp.full(
                (self.capacity,), UNSET_SOURCE_TOKEN, dtype=jnp.int32
            ),
            source_generation=jnp.full(
                (self.capacity,), UNSET_SOURCE_GENERATION, dtype=jnp.int32
            ),
            score_version=jnp.full(
                (self.capacity,), UNSET_SCORE_VERSION, dtype=jnp.int32
            ),
        )

    def add(
        self,
        buffer_state: SourceAwareReplayBufferState,
        xs: chex.ArrayTree,
        provenance: ReplayProvenance,
        mask: chex.Array | None = None,
    ) -> SourceAwareReplayBufferState:
        """Add payload and provenance using one baseline ring-index calculation."""
        first_leaf = jtu.tree_leaves(xs)[0]
        batch_shape = (first_leaf.shape[0],)
        chex.assert_tree_shape_prefix(provenance, batch_shape)

        if mask is not None:
            assert mask.ndim == 1
            chex.assert_shape(mask, batch_shape)
            batch_size = mask.sum()
            cumsum_mask = jnp.cumsum(mask, axis=0, dtype=jnp.int32)
            indices = (buffer_state.current_index + cumsum_mask - 1) % self.capacity
            indices = jnp.where(mask, indices, self.capacity)
        else:
            batch_size = first_leaf.shape[0]
            indices = (
                buffer_state.current_index
                + jnp.arange(batch_size, dtype=jnp.int32)
            ) % self.capacity

        sidecars = ReplayProvenance(
            origin_type=buffer_state.origin_type,
            source_token=buffer_state.source_token,
            source_generation=buffer_state.source_generation,
            score_version=buffer_state.score_version,
        )
        sidecars = tree_set(sidecars, provenance, indices, unique_indices=False)

        payload_state = ReplayBuffer.add(self, buffer_state, xs, mask)
        return payload_state.replace(
            origin_type=sidecars.origin_type,
            source_token=sidecars.source_token,
            source_generation=sidecars.source_generation,
            score_version=sidecars.score_version,
            current_index=(buffer_state.current_index + batch_size) % self.capacity,
            buffer_size=jnp.minimum(
                buffer_state.buffer_size + batch_size, self.capacity
            ),
        )

    def add_provenance(
        self,
        buffer_state: SourceAwareReplayBufferState,
        provenance: ReplayProvenance,
        mask: chex.Array | None = None,
        *,
        current_index: chex.Array | None = None,
    ) -> SourceAwareReplayBufferState:
        """Write only provenance sidecars for an already-added payload batch."""
        first_leaf = jtu.tree_leaves(provenance)[0]
        batch_shape = (first_leaf.shape[0],)
        start_index = (
            buffer_state.current_index if current_index is None else current_index
        )

        if mask is not None:
            assert mask.ndim == 1
            chex.assert_shape(mask, batch_shape)
            cumsum_mask = jnp.cumsum(mask, axis=0, dtype=jnp.int32)
            indices = (start_index + cumsum_mask - 1) % self.capacity
            indices = jnp.where(mask, indices, self.capacity)
        else:
            indices = (
                start_index + jnp.arange(first_leaf.shape[0], dtype=jnp.int32)
            ) % self.capacity

        sidecars = ReplayProvenance(
            origin_type=buffer_state.origin_type,
            source_token=buffer_state.source_token,
            source_generation=buffer_state.source_generation,
            score_version=buffer_state.score_version,
        )
        sidecars = tree_set(sidecars, provenance, indices, unique_indices=False)
        return buffer_state.replace(
            origin_type=sidecars.origin_type,
            source_token=sidecars.source_token,
            source_generation=sidecars.source_generation,
            score_version=sidecars.score_version,
        )
