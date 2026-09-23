import chex
import jax
import jax.numpy as jnp
import optax

from evorl.algorithms.erl.erl_td3.learner_utility import (
    derive_diagnostic_key,
    td3_critic_loss,
    virtual_critic_utility,
)


def _squared_loss(params, batch, key):
    del key
    prediction = params["w"] * batch["x"]
    return jnp.mean(jnp.square(prediction - batch["y"]))


def test_virtual_utility_uses_functional_optimizer_update_without_mutation():
    params = {"w": jnp.array(0.0)}
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(params)
    source_batch = {"x": jnp.array([1.0, 1.0]), "y": jnp.array([1.0, 1.0])}
    reference_batch = {
        "x": jnp.array([1.0, 1.0]),
        "y": jnp.array([1.0, 1.0]),
    }
    protected_before = (params, opt_state, jax.random.key(0))

    result = virtual_critic_utility(
        loss_fn=_squared_loss,
        optimizer=optimizer,
        params=params,
        opt_state=opt_state,
        source_batch=source_batch,
        reference_batch=reference_batch,
        key=protected_before[2],
    )

    assert result.utility > 0
    assert result.loss_ref_after < result.loss_ref_before
    chex.assert_trees_all_equal(protected_before, (params, opt_state, protected_before[2]))


def test_diagnostic_key_is_deterministic_without_advancing_baseline_key():
    baseline_key = jax.random.key(7)

    first = derive_diagnostic_key(baseline_key, generation=3, tag=41)
    repeated = derive_diagnostic_key(baseline_key, generation=3, tag=41)
    later_generation = derive_diagnostic_key(baseline_key, generation=4, tag=41)

    chex.assert_trees_all_equal(first, repeated)
    assert not bool(jnp.all(first == later_generation))
    chex.assert_trees_all_equal(baseline_key, jax.random.key(7))


def test_td3_critic_loss_adapter_uses_agent_critic_loss_directly():
    class Loss:
        critic_loss = jnp.array(3.5)

    class Agent:
        def __init__(self):
            self.received = None

        def critic_loss(self, agent_state, batch, key):
            self.received = (agent_state, batch, key)
            return Loss()

    agent = Agent()
    agent_state = object()
    batch = object()
    key = jax.random.key(1)

    assert float(td3_critic_loss(agent, agent_state, batch, key)) == 3.5
    assert agent.received == (agent_state, batch, key)
