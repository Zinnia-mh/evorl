"""Fixed-window temporal EA replay distributions without source scoring."""

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from evorl.algorithms.erl.erl_td3.source_replay import (
    WeightedSourceDistribution,
    build_sampling_cdf,
)
from evorl.replay_buffers.source_aware_replay_buffer import EA_ORIGIN


MODE_RECENT_30 = "recent30"
MODE_STRATIFIED_5_25 = "stratified5_25"
TEMPORAL_MODES = (MODE_RECENT_30, MODE_STRATIFIED_5_25)


class TemporalDistributionResult(NamedTuple):
    distribution: WeightedSourceDistribution
    prepared: chex.Array
    recent_count: chex.Array
    history_count: chex.Array
    window_count: chex.Array


@partial(
    jax.jit,
    static_argnames=("mode", "recent_generations", "history_generations"),
)
def build_temporal_ea_distribution(
    replay_buffer_state,
    *,
    current_generation,
    mode: str,
    recent_generations: int = 5,
    history_generations: int = 25,
) -> TemporalDistributionResult:
    """Build a transition distribution conditioned on a realized EA slot.

    The intervention remains inactive until a complete recent+history window
    exists. Empty required strata also fail closed to the original draw.
    """
    if mode not in TEMPORAL_MODES:
        raise ValueError(f"unknown temporal replay mode: {mode}")
    recent_generations = int(recent_generations)
    history_generations = int(history_generations)
    if recent_generations <= 0 or history_generations <= 0:
        raise ValueError("temporal replay windows must be positive")

    capacity = replay_buffer_state.origin_type.shape[0]
    indices = jnp.arange(capacity, dtype=jnp.int32)
    resident = indices < replay_buffer_state.buffer_size
    ea = resident & (replay_buffer_state.origin_type == EA_ORIGIN)
    generation = jnp.asarray(current_generation, dtype=jnp.int32)
    age = generation - replay_buffer_state.source_generation.astype(jnp.int32)
    valid_age = age >= 0
    recent = ea & valid_age & (age < recent_generations)
    history = (
        ea
        & valid_age
        & (age >= recent_generations)
        & (age < recent_generations + history_generations)
    )
    window = recent | history

    recent_count_i = jnp.sum(recent, dtype=jnp.int32)
    history_count_i = jnp.sum(history, dtype=jnp.int32)
    window_count_i = recent_count_i + history_count_i
    n_ea_i = jnp.sum(ea, dtype=jnp.int32)
    recent_count = recent_count_i.astype(jnp.float32)
    history_count = history_count_i.astype(jnp.float32)
    window_count = window_count_i.astype(jnp.float32)
    complete_window = generation >= recent_generations + history_generations

    if mode == MODE_RECENT_30:
        strata_ready = window_count_i > 0
        probability = jnp.where(
            window,
            1.0 / jnp.maximum(window_count, 1.0),
            0.0,
        )
    else:
        strata_ready = (recent_count_i > 0) & (history_count_i > 0)
        probability = jnp.where(
            recent,
            0.5 / jnp.maximum(recent_count, 1.0),
            jnp.where(
                history,
                0.5 / jnp.maximum(history_count, 1.0),
                0.0,
            ),
        )

    prepared = complete_window & strata_ready & (n_ea_i > 0)
    probability = jnp.where(prepared, probability, 0.0).astype(jnp.float32)
    cdf = build_sampling_cdf(probability)
    squared_mass = jnp.sum(probability * probability)
    effective_count = jnp.where(
        prepared,
        1.0 / jnp.maximum(squared_mass, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    max_probability = jnp.where(prepared, jnp.max(probability), 0.0)
    n_ea = n_ea_i.astype(jnp.float32)
    eligible_mass = window_count / jnp.maximum(n_ea, 1.0)
    one = jnp.asarray([1.0], dtype=jnp.float32)
    zero = jnp.asarray([0.0], dtype=jnp.float32)
    source_summary = jnp.where(prepared, one, zero)
    distribution = WeightedSourceDistribution(
        baseline_source_probability=source_summary,
        weighted_source_probability=source_summary,
        source_probability=source_summary,
        transition_probability=probability,
        transition_cdf=cdf,
        n_ea=n_ea_i,
        eligible_source_count=jnp.where(
            prepared,
            jnp.asarray(1 if mode == MODE_RECENT_30 else 2, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        resident_source_count=jnp.where(n_ea_i > 0, 1, 0).astype(jnp.int32),
        eligible_baseline_mass=jnp.where(prepared, eligible_mass, 0.0),
        effective_source_count=effective_count,
        ess_fraction=jnp.where(
            n_ea_i > 0, effective_count / jnp.maximum(n_ea, 1.0), 0.0
        ),
        max_source_probability=max_probability,
        uniform_floor_per_transition=jnp.asarray(0.0, dtype=jnp.float32),
        uniform_component_total_mass=jnp.asarray(0.0, dtype=jnp.float32),
        has_resident_ea=n_ea_i > 0,
    )
    return TemporalDistributionResult(
        distribution=distribution,
        prepared=prepared,
        recent_count=recent_count_i,
        history_count=history_count_i,
        window_count=window_count_i,
    )
