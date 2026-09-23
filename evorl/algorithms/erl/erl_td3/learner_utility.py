from typing import Any, Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


LossFn = Callable[[chex.ArrayTree, Any, chex.PRNGKey], chex.Array]


class UtilityResult(NamedTuple):
    utility: chex.Array
    loss_ref_before: chex.Array
    loss_ref_after: chex.Array


def td3_critic_loss(agent: Any, agent_state: Any, batch: Any, key: chex.PRNGKey) -> chex.Array:
    """Return the exact critic loss exposed by the active TD3 agent."""
    return agent.critic_loss(agent_state, batch, key).critic_loss


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
