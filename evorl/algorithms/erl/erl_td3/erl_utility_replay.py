"""Utility-replay pilot with an exact alpha-zero arm and weighted EA overlay.

Phase 7 consumes only explicitly prepared distributions.  Diagnostic score
lifecycle updates remain intentionally outside this module until Phase 8.
"""

import time
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.replay_buffers.replay_buffer import ReplayBuffer
from evorl.algorithms.td3 import TD3TrainMetric
from evorl.types import PyTreeDict
from evorl.replay_buffers.source_aware_replay_buffer import (
    EA_ORIGIN,
    NO_SCORE_VERSION,
    ReplayProvenance,
    SourceAwareReplayBuffer,
    encode_source_token,
    make_rl_provenance,
)
from evorl.utils.jax_utils import is_jitted, tree_get

from .erl_origin import ERLTrainMetric, ERLWorkflow
from .erl_td3_workflow import (
    ERLTD3WorkflowTemplate,
    create_dummy_td3_trainmetric,
    erl_replace_td3_actor_params,
)
from .source_replay import (
    WeightedReplayOverlay,
    WeightedSourceDistribution,
    apply_weighted_ea_overlay,
    draw_baseline_indices,
    overlay_ea_indices,
)


ALLOWED_PILOT_ONLY_PATHS = (
    "replay_buffer_state.origin_type",
    "replay_buffer_state.source_token",
    "replay_buffer_state.source_generation",
    "replay_buffer_state.score_version",
    "utility_replay_state",
    "train_metrics.time_cost_per_iter",
)


class AlphaZeroSample(NamedTuple):
    batch: chex.ArrayTree
    baseline_indices: chex.Array
    final_indices: chex.Array
    intervention_key: chex.PRNGKey
    baseline_draw_count: chex.Array


class UtilityReplayState(NamedTuple):
    intervention_key: chex.PRNGKey
    distribution: WeightedSourceDistribution
    distribution_prepared: chex.Array
    distribution_version: chex.Array


class WeightedReplaySample(NamedTuple):
    batch: chex.ArrayTree
    baseline_indices: chex.Array
    final_indices: chex.Array
    ea_slot_mask: chex.Array
    rl_slot_mask: chex.Array
    next_intervention_key: chex.PRNGKey
    replacement_count: chex.Array


class WeightedReplayDraws(NamedTuple):
    batches: chex.ArrayTree
    baseline_indices: chex.Array
    final_indices: chex.Array
    ea_slot_mask: chex.Array
    rl_slot_mask: chex.Array
    intervention_subkeys: chex.Array
    replacement_count: chex.Array
    utility_replay_state: UtilityReplayState


def init_utility_replay_state(
    replay_capacity: int, intervention_key: chex.PRNGKey
) -> UtilityReplayState:
    """Allocate a fixed-shape inactive distribution for pre-Phase-8 startup."""
    replay_capacity = int(replay_capacity)
    zero_source = jnp.zeros((1,), dtype=jnp.float32)
    zero_transition = jnp.zeros((replay_capacity,), dtype=jnp.float32)
    distribution = WeightedSourceDistribution(
        baseline_source_probability=zero_source,
        weighted_source_probability=zero_source,
        source_probability=zero_source,
        transition_probability=zero_transition,
        transition_cdf=zero_transition,
        n_ea=jnp.asarray(0, dtype=jnp.int32),
        eligible_source_count=jnp.asarray(0, dtype=jnp.int32),
        resident_source_count=jnp.asarray(0, dtype=jnp.int32),
        eligible_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
        effective_source_count=jnp.asarray(0.0, dtype=jnp.float32),
        ess_fraction=jnp.asarray(0.0, dtype=jnp.float32),
        max_source_probability=jnp.asarray(0.0, dtype=jnp.float32),
        uniform_floor_per_transition=jnp.asarray(0.0, dtype=jnp.float32),
        uniform_component_total_mass=jnp.asarray(0.0, dtype=jnp.float32),
        has_resident_ea=jnp.asarray(False),
    )
    return UtilityReplayState(
        intervention_key=intervention_key,
        distribution=distribution,
        distribution_prepared=jnp.asarray(False),
        distribution_version=jnp.asarray(-1, dtype=jnp.int32),
    )


def weighted_replay_sample(
    replay_buffer_state,
    replay_key: chex.PRNGKey,
    intervention_key: chex.PRNGKey,
    distribution: WeightedSourceDistribution,
    distribution_prepared: chex.Array,
    sample_batch_size: int,
) -> WeightedReplaySample:
    """Draw baseline indices once, overlay them, then gather payload once."""
    baseline_indices = draw_baseline_indices(
        replay_buffer_state, replay_key, sample_batch_size
    )

    def weighted(_):
        return apply_weighted_ea_overlay(
            baseline_indices,
            replay_buffer_state.origin_type,
            distribution,
            intervention_key,
        )

    def inactive(_):
        overlay = overlay_ea_indices(
            baseline_indices, replay_buffer_state.origin_type
        )
        return WeightedReplayOverlay(
            baseline_indices=overlay.baseline_indices,
            final_indices=overlay.final_indices,
            ea_slot_mask=overlay.ea_slot_mask,
            rl_slot_mask=overlay.rl_slot_mask,
            next_intervention_key=intervention_key,
            replacement_count=jnp.asarray(0, dtype=jnp.int32),
        )

    overlay = jax.lax.cond(
        distribution_prepared, weighted, inactive, operand=None
    )
    batch = tree_get(replay_buffer_state.data, overlay.final_indices)
    return WeightedReplaySample(
        batch=batch,
        baseline_indices=overlay.baseline_indices,
        final_indices=overlay.final_indices,
        ea_slot_mask=overlay.ea_slot_mask,
        rl_slot_mask=overlay.rl_slot_mask,
        next_intervention_key=overlay.next_intervention_key,
        replacement_count=overlay.replacement_count,
    )


def weighted_replay_draws(
    replay_buffer_state,
    replay_keys: chex.Array,
    utility_replay_state: UtilityReplayState,
    sample_batch_size: int,
) -> WeightedReplayDraws:
    """Vectorize draws with one unique pilot-local key per replay operation."""
    split_keys = jax.random.split(
        utility_replay_state.intervention_key, replay_keys.shape[0] + 1
    )
    candidate_next_key = split_keys[0]
    intervention_subkeys = split_keys[1:]

    def one_draw(replay_key, intervention_subkey):
        return weighted_replay_sample(
            replay_buffer_state,
            replay_key,
            intervention_subkey,
            utility_replay_state.distribution,
            utility_replay_state.distribution_prepared,
            sample_batch_size,
        )

    samples = jax.vmap(one_draw)(replay_keys, intervention_subkeys)
    replacement_possible = (
        utility_replay_state.distribution_prepared
        & utility_replay_state.distribution.has_resident_ea
        & jnp.any(samples.ea_slot_mask)
    )
    next_key = jnp.where(
        replacement_possible,
        candidate_next_key,
        utility_replay_state.intervention_key,
    )
    return WeightedReplayDraws(
        batches=samples.batch,
        baseline_indices=samples.baseline_indices,
        final_indices=samples.final_indices,
        ea_slot_mask=samples.ea_slot_mask,
        rl_slot_mask=samples.rl_slot_mask,
        intervention_subkeys=intervention_subkeys,
        replacement_count=jnp.sum(samples.replacement_count, dtype=jnp.int32),
        utility_replay_state=utility_replay_state._replace(
            intervention_key=next_key
        ),
    )


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

    def setup(self, key):
        state = super().setup(key)
        intervention_key = jax.random.fold_in(
            key, jnp.asarray(0x55525037, dtype=jnp.uint32)
        )
        return state.replace(
            utility_replay_state=init_utility_replay_state(
                self.replay_buffer.capacity, intervention_key
            )
        )

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
        alpha = float(config.alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        workflow = super()._build_from_config(config)
        baseline = workflow.replay_buffer
        workflow.replay_buffer = UtilityReplayBuffer(
            capacity=baseline.capacity,
            min_sample_timesteps=baseline.min_sample_timesteps,
            sample_batch_size=baseline.sample_batch_size,
        )

        if alpha > 0.0:
            def _weighted_rl_sample_and_update_fn(carry, unused_t):
                del unused_t
                (
                    key,
                    agent_state,
                    opt_state,
                    replay_buffer_state,
                    _,
                    utility_replay_state,
                ) = carry
                key, rb_key, learn_key = jax.random.split(key, 3)
                num_draws = (
                    config.actor_update_interval * config.num_rl_agents
                )
                rb_keys = jax.random.split(rb_key, num_draws)
                sampled = weighted_replay_draws(
                    replay_buffer_state,
                    rb_keys,
                    utility_replay_state,
                    workflow.replay_buffer.sample_batch_size,
                )
                sample_batches = jtu.tree_map(
                    lambda x: x.reshape(
                        (
                            config.actor_update_interval,
                            config.num_rl_agents,
                            *x.shape[1:],
                        )
                    ),
                    sampled.batches,
                )
                (
                    (agent_state, opt_state),
                    (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
                ) = workflow._rl_update_fn(
                    agent_state, opt_state, sample_batches, learn_key
                )
                td3_metrics = TD3TrainMetric(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    raw_loss_dict=PyTreeDict(
                        {**critic_loss_dict, **actor_loss_dict}
                    ),
                )
                return (
                    key,
                    agent_state,
                    opt_state,
                    replay_buffer_state,
                    td3_metrics,
                    sampled.utility_replay_state,
                ), None

            if is_jitted(cls.evaluate):
                _weighted_rl_sample_and_update_fn = jax.jit(
                    _weighted_rl_sample_and_update_fn
                )
            workflow._rl_sample_and_update_fn = (
                _weighted_rl_sample_and_update_fn
            )
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

    def _weighted_rl_update(
        self,
        agent_state,
        opt_state,
        replay_buffer_state,
        key,
        num_updates,
        utility_replay_state,
    ):
        init_td3_metrics = create_dummy_td3_trainmetric(
            self.config.num_rl_agents
        )
        (
            (
                _,
                agent_state,
                opt_state,
                _,
                td3_metrics,
                utility_replay_state,
            ),
            _,
        ) = jax.lax.scan(
            self._rl_sample_and_update_fn,
            (
                key,
                agent_state,
                opt_state,
                replay_buffer_state,
                init_td3_metrics,
                utility_replay_state,
            ),
            (),
            length=num_updates,
        )
        return td3_metrics, agent_state, opt_state, utility_replay_state

    def _weighted_step(self, state):
        """Mirror Original ERL while carrying only pilot intervention state."""
        start_t = time.perf_counter()
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        utility_replay_state = state.utility_replay_state
        iterations = state.metrics.iterations + 1
        sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)
        sampled_episodes = jnp.zeros((), dtype=jnp.uint32)
        key, ec_rollout_key, rl_rollout_key, learn_key = jax.random.split(
            state.key, num=4
        )

        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        pop_agent_state = erl_replace_td3_actor_params(
            agent_state, pop_actor_params
        )
        ec_eval_metrics, _, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(
            jnp.uint32
        )
        sampled_episodes += jnp.uint32(
            self.config.episodes_for_fitness * pop_size
        )
        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        rl_eval_metrics, _, replay_buffer_state = self._rl_rollout(
            agent_state, replay_buffer_state, rl_rollout_key
        )
        if self.config.rl_updates_mode == "global":
            total_timesteps = (
                state.metrics.sampled_timesteps + sampled_timesteps
            )
            num_updates = (
                jnp.ceil(total_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        elif self.config.rl_updates_mode == "iter":
            num_updates = (
                jnp.ceil(sampled_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        else:
            raise ValueError(
                f"Unknown rl_updates_mode: {self.config.rl_updates_mode}"
            )
        (
            td3_metrics,
            agent_state,
            opt_state,
            utility_replay_state,
        ) = self._weighted_rl_update(
            agent_state,
            opt_state,
            replay_buffer_state,
            learn_key,
            num_updates,
            utility_replay_state,
        )
        td3_metrics = td3_metrics.replace(
            actor_loss=td3_metrics.actor_loss / self.config.num_rl_agents,
            critic_loss=td3_metrics.critic_loss / self.config.num_rl_agents,
        )
        train_metrics = train_metrics.replace(
            num_updates_per_iter=num_updates,
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
        )
        rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
            jnp.uint32
        )
        sampled_timesteps += rl_sampled_timesteps
        sampled_episodes += jnp.uint32(
            self.config.num_rl_agents * self.config.rollout_episodes
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        if iterations % self.config.rl_injection_interval == 0:
            ec_metrics, ec_opt_state = self._ec_update_with_rl_injection(
                ec_opt_state, agent_state, fitnesses
            )
        else:
            ec_metrics, ec_opt_state = self._ec_update(
                ec_opt_state, fitnesses
            )
        train_metrics = train_metrics.replace(
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
            time_cost_per_iter=time.perf_counter() - start_t,
        )
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=(
                state.metrics.sampled_timesteps + sampled_timesteps
            ),
            sampled_episodes=(
                state.metrics.sampled_episodes + sampled_episodes
            ),
            rl_sampled_timesteps=(
                state.metrics.rl_sampled_timesteps + rl_sampled_timesteps
            ),
            iterations=iterations,
        )
        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
            opt_state=opt_state,
            utility_replay_state=utility_replay_state,
        )
        return train_metrics, state

    def step(self, state):
        operation = (
            super().step if float(self.config.alpha) == 0.0
            else self._weighted_step
        )
        return self._with_generation_context(state, operation)

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
