from typing import Any, Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from evorl.sample_batch import SampleBatch
from evorl.utils.jax_utils import right_shift_with_padding


LossFn = Callable[[chex.ArrayTree, Any, chex.PRNGKey], chex.Array]


class UtilityResult(NamedTuple):
    utility: chex.Array
    loss_ref_before: chex.Array
    loss_ref_after: chex.Array


class SourceBatch(NamedTuple):
    source_id: str
    individual_id: int
    batch: SampleBatch | None
    eligible: bool
    reason: str | None
    valid_transition_count: int


def td3_critic_loss(agent: Any, agent_state: Any, batch: Any, key: chex.PRNGKey) -> chex.Array:
    """Return the exact critic loss exposed by the active TD3 agent."""
    return agent.critic_loss(agent_state, batch, key).critic_loss


def build_source_batches(
    raw_trajectory: SampleBatch,
    *,
    generation: int,
    source_batch_size: int,
    key: chex.PRNGKey,
) -> tuple[SourceBatch, ...]:
    """Create provenance-preserving source batches before replay flattening."""
    if raw_trajectory.dones is None:
        raise ValueError("raw population trajectory must include dones")
    if raw_trajectory.dones.ndim < 3:
        raise ValueError("raw population trajectory must have [population, time, env] dones")

    population_size = raw_trajectory.dones.shape[0]
    results = []
    for individual_id in range(population_size):
        individual = jtu.tree_map(
            lambda leaf: None if leaf is None else leaf[individual_id], raw_trajectory
        )
        valid_mask = jnp.logical_not(right_shift_with_padding(individual.dones, 1))
        valid_indices = np.flatnonzero(np.asarray(jax.device_get(valid_mask)).reshape(-1))
        source_id = f"gen_{generation}_ind_{individual_id}"
        valid_count = int(valid_indices.size)
        if valid_count < source_batch_size:
            results.append(
                SourceBatch(
                    source_id=source_id,
                    individual_id=individual_id,
                    batch=None,
                    eligible=False,
                    reason="insufficient_valid_transitions",
                    valid_transition_count=valid_count,
                )
            )
            continue

        source_key = jax.random.fold_in(key, jnp.uint32(individual_id))
        chosen = jax.random.choice(
            source_key, jnp.asarray(valid_indices), (source_batch_size,), replace=False
        )

        def _select(leaf):
            if leaf is None:
                return None
            flattened = leaf.reshape((-1, *leaf.shape[2:]))
            return flattened[chosen]

        results.append(
            SourceBatch(
                source_id=source_id,
                individual_id=individual_id,
                batch=jtu.tree_map(_select, individual),
                eligible=True,
                reason=None,
                valid_transition_count=valid_count,
            )
        )
    return tuple(results)


def derive_diagnostic_key(
    baseline_key: chex.PRNGKey, generation: int, tag: int
) -> chex.PRNGKey:
    """Derive diagnostic randomness without returning or mutating a new baseline key."""
    key = jax.random.fold_in(baseline_key, jnp.uint32(tag))
    return jax.random.fold_in(key, jnp.uint32(generation))


def virtual_critic_utility(
    *,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    params: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    source_batch: Any,
    reference_batch: Any,
    key: chex.PRNGKey,
) -> UtilityResult:
    """Measure one optimizer-faithful critic update without writing training state."""
    source_key, reference_key = jax.random.split(key)
    loss_ref_before = loss_fn(params, reference_batch, reference_key)
    grads = jax.grad(loss_fn)(params, source_batch, source_key)
    updates, _ = optimizer.update(grads, opt_state, params)
    virtual_params = optax.apply_updates(params, updates)
    loss_ref_after = loss_fn(virtual_params, reference_batch, reference_key)
    return UtilityResult(
        utility=loss_ref_before - loss_ref_after,
        loss_ref_before=loss_ref_before,
        loss_ref_after=loss_ref_after,
    )
