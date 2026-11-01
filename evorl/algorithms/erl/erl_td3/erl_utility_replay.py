"""Utility-replay pilot with pre-exposure diagnostic score lifecycle."""

import time
from numbers import Integral
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
from evorl.utils.jax_utils import right_shift_with_padding, tree_stop_gradient
from evorl.utils.rl_toolkits import (
    flatten_pop_rollout_episode,
    flatten_rollout_trajectory,
)

from .erl_origin import ERLTrainMetric, ERLWorkflow
from .erl_ga import EvaluateMetric
from .erl_learner_utility import ERLLearnerUtilityWorkflow
from .learner_utility import derive_diagnostic_key, td3_critic_loss
from .erl_td3_workflow import (
    ERLTD3WorkflowTemplate,
    create_dummy_td3_trainmetric,
    erl_replace_td3_actor_params,
)
from .source_replay import (
    SCORE_KIND_FITNESS,
    SCORE_KIND_TD,
    SCORE_KIND_UTILITY,
    SCORE_TABLE_OK,
    SourceScoreTable,
    WeightedReplayOverlay,
    WeightedSourceDistribution,
    apply_checkpoint_score_update,
    apply_weighted_ea_overlay,
    build_checkpoint_score_update,
    build_weighted_source_distribution,
    compute_source_eligibility,
    draw_baseline_indices,
    init_source_score_table,
    overlay_ea_indices,
    raise_for_score_table_status,
    update_resident_counts,
)
from .utility_replay_budget import BudgetAudit, validate_arm_config


def _optimizer_step_count(opt_state):
    """Read Optax Adam's actual step counter from its state tree."""
    count = getattr(opt_state, "count", None)
    if count is not None and not callable(count):
        return int(jax.device_get(count))
    if isinstance(opt_state, (tuple, list)):
        for item in opt_state:
            try:
                return _optimizer_step_count(item)
            except ValueError:
                pass
    raise ValueError("optimizer state has no observable step counter")


_SCORE_KIND_BY_NAME = {
    "fitness": SCORE_KIND_FITNESS,
    "td": SCORE_KIND_TD,
    "utility": SCORE_KIND_UTILITY,
}

EVAL_LOSS_TAG = 0x4556414C


def derive_eval_batch_key(baseline_key, generation):
    """Derive the held-out replay key outside the frozen reference namespace."""
    root = derive_diagnostic_key(baseline_key, generation, EVAL_LOSS_TAG)
    return jax.random.split(root)[0]


class HeldOutEvalLoss(NamedTuple):
    loss: float
    eval_fingerprint: str
    reference_fingerprint: str
    generation: int


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
    score_table: SourceScoreTable
    distribution: WeightedSourceDistribution
    distribution_prepared: chex.Array
    distribution_version: chex.Array
    distribution_generation: chex.Array


class DiagnosticCheckpointResult(NamedTuple):
    utility_replay_state: UtilityReplayState
    score_version: chex.Array
    applied: bool


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
    replay_capacity: int,
    intervention_key: chex.PRNGKey,
    score_table_capacity: int | None = None,
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
        score_table=init_source_score_table(
            replay_capacity
            if score_table_capacity is None
            else score_table_capacity
        ),
        distribution=distribution,
        distribution_prepared=jnp.asarray(False),
        distribution_version=jnp.asarray(-1, dtype=jnp.int32),
        distribution_generation=jnp.asarray(-1, dtype=jnp.int32),
    )


def apply_diagnostic_checkpoint(
    utility_replay_state: UtilityReplayState,
    records,
    *,
    score_kind: int,
    population_size: int,
) -> DiagnosticCheckpointResult:
    """Install one usable frozen checkpoint before matching EA insertion."""
    records = tuple(records)
    if not records:
        return DiagnosticCheckpointResult(
            utility_replay_state=utility_replay_state,
            score_version=jnp.asarray(-1, dtype=jnp.int32),
            applied=False,
        )
    next_version = int(
        jax.device_get(utility_replay_state.score_table.current_version)
    ) + 1
    update = build_checkpoint_score_update(
        records,
        score_kind=score_kind,
        population_size=population_size,
        version=next_version,
    )
    if not bool(jax.device_get(jnp.any(update.validity))):
        return DiagnosticCheckpointResult(
            utility_replay_state=utility_replay_state,
            score_version=jnp.asarray(-1, dtype=jnp.int32),
            applied=False,
        )
    applied = apply_checkpoint_score_update(
        utility_replay_state.score_table, update
    )
    raise_for_score_table_status(applied.status)
    return DiagnosticCheckpointResult(
        utility_replay_state=utility_replay_state._replace(
            score_table=applied.table
        ),
        score_version=update.version,
        applied=True,
    )


def refresh_replay_distribution(
    utility_replay_state: UtilityReplayState,
    replay_buffer_state,
    *,
    current_generation: int,
    freshness_window: int,
    min_resident: int,
    alpha: float,
    replay_temperature: float,
    timing_sink: dict | None = None,
) -> UtilityReplayState:
    """Refresh residency, eligibility and distribution after all insertions."""
    capacity = replay_buffer_state.origin_type.shape[0]
    indices = jnp.arange(capacity, dtype=jnp.int32)
    resident = indices < replay_buffer_state.buffer_size
    resident_ea = resident & (replay_buffer_state.origin_type == EA_ORIGIN)
    final_tokens = jnp.where(
        resident_ea, replay_buffer_state.source_token, -1
    ).astype(jnp.int32)
    reset_counts = utility_replay_state.score_table._replace(
        resident_count=jnp.zeros_like(
            utility_replay_state.score_table.resident_count
        )
    )
    recounted = update_resident_counts(
        reset_counts,
        resident_source_tokens=jnp.full((capacity,), -1, dtype=jnp.int32),
        write_indices=indices,
        new_source_tokens=final_tokens,
        mask=resident_ea,
    )
    raise_for_score_table_status(recounted.status)
    eligibility_start = time.perf_counter()
    eligibility = compute_source_eligibility(
        recounted.table,
        current_generation=jnp.asarray(current_generation, dtype=jnp.uint32),
        freshness_window=freshness_window,
        min_resident=min_resident,
    )
    if timing_sink is not None:
        timing_sink["eligibility_seconds"] = (
            time.perf_counter() - eligibility_start
        )
    distribution_start = time.perf_counter()
    distribution = build_weighted_source_distribution(
        eligibility.table,
        resident_source_tokens=recounted.resident_source_tokens,
        resident_mask=resident_ea,
        alpha=alpha,
        replay_temperature=replay_temperature,
    )
    if timing_sink is not None:
        timing_sink["distribution_refresh_seconds"] = (
            time.perf_counter() - distribution_start
        )
    return utility_replay_state._replace(
        score_table=eligibility.table,
        distribution=distribution,
        distribution_prepared=(
            distribution.has_resident_ea
            & (distribution.eligible_source_count > 0)
        ),
        distribution_version=eligibility.table.current_version,
        distribution_generation=jnp.asarray(
            current_generation, dtype=jnp.int32
        ),
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
    *,
    generation: int,
    population_size: int,
    time_size: int,
    episode_batch: int,
    score_version: int | chex.Array = NO_SCORE_VERSION,
) -> ReplayProvenance:
    """Encode authoritative generation and individual identity for an EA rollout."""
    population_size = int(population_size)
    time_size = int(time_size)
    episode_batch = int(episode_batch)
    if isinstance(generation, Integral):
        base_token = encode_source_token(generation, 0, population_size)
        encode_source_token(generation, population_size - 1, population_size)
    else:
        # The warmup scan carries generation as a JAX scalar. Its upper bound
        # is checked from the static warmup configuration at workflow build.
        base_token = jnp.asarray(generation, dtype=jnp.int32) * population_size
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
            (flat_size,), score_version, dtype=jnp.int32
        ),
    )


class ERLUtilityReplayWorkflow(ERLLearnerUtilityWorkflow):
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

    def _evaluate_optional_loss(self, replay_buffer_state, *, reference_fingerprint):
        """Read an independently sampled replay batch for a secondary endpoint."""
        if not bool(getattr(self.config, "enable_eval_loss_endpoint", False)):
            return None
        if not reference_fingerprint or not bool(
            jax.device_get(self.replay_buffer.can_sample(replay_buffer_state))
        ):
            return None

        context = self._diagnostic_context
        batch_key = derive_eval_batch_key(context.key, context.generation)
        root = derive_diagnostic_key(
            context.key, context.generation, EVAL_LOSS_TAG
        )
        loss_key = jax.random.split(root)[1]
        for attempt in range(16):
            eval_batch = self.replay_buffer.sample(
                replay_buffer_state, jax.random.fold_in(batch_key, attempt)
            )
            eval_fingerprint = self._reference_batch_fingerprint(eval_batch)
            if eval_fingerprint != reference_fingerprint:
                break
        else:
            raise ValueError("held-out eval batch is identical to B_ref")

        loss = td3_critic_loss(
            self.agent, context.learner_agent_state, eval_batch, loss_key
        )
        return HeldOutEvalLoss(
            loss=float(jax.device_get(loss)),
            eval_fingerprint=eval_fingerprint,
            reference_fingerprint=reference_fingerprint,
            generation=context.generation,
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
        cls._evaluate_with_count = jax.jit(
            cls._evaluate_with_count, static_argnums=(0,)
        )
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )

    @classmethod
    def _build_from_config(cls, config):
        validate_arm_config(config)
        encode_source_token(
            int(config.warmup_iters), int(config.pop_size) - 1,
            int(config.pop_size),
        )
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

            if is_jitted(cls._evaluate_with_count):
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
        self._active_utility_replay_state = utility_replay_state
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
        utility_replay_state = self._active_utility_replay_state
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
        refresh_start = time.perf_counter()
        utility_replay_state = refresh_replay_distribution(
            utility_replay_state,
            replay_buffer_state,
            current_generation=int(jax.device_get(iterations)),
            freshness_window=int(getattr(self.config, "freshness_window", 3)),
            min_resident=int(getattr(self.config, "min_resident", 1)),
            alpha=float(self.config.alpha),
            replay_temperature=float(self.config.replay_temperature),
            timing_sink=self._latest_score_lifecycle_audit,
        )
        self._active_utility_replay_state = utility_replay_state
        self._latest_score_lifecycle_audit["resident_and_refresh_seconds"] = (
            time.perf_counter() - refresh_start
        )
        self._latest_score_lifecycle_audit["distribution_generation"] = int(
            jax.device_get(utility_replay_state.distribution_generation)
        )
        distribution = utility_replay_state.distribution
        self._latest_score_lifecycle_audit.update(
            eligible_source_count=int(
                jax.device_get(distribution.eligible_source_count)
            ),
            n_ea=int(jax.device_get(distribution.n_ea)),
            max_source_probability=float(
                jax.device_get(distribution.max_source_probability)
            ),
            ess=float(jax.device_get(distribution.effective_source_count)),
        )
        self._latest_score_lifecycle_audit["ordering"].append(
            "distribution_refresh_complete"
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
        self._latest_score_lifecycle_audit["ordering"].append(
            "learner_sampling_start"
        )
        learner_start = time.perf_counter()
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
        self._latest_score_lifecycle_audit["weighted_learner_phase_seconds"] = (
            time.perf_counter() - learner_start
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
        del self._active_utility_replay_state
        return train_metrics, state

    def step(self, state):
        if float(self.config.alpha) == 0.0:
            observer = getattr(self, "_calibration_observer", None)
            if observer is not None:
                observer.begin(self, state)
            result = self._with_generation_context(state, super().step)
            self._record_budget_step(state, result)
            if observer is not None:
                observer.finish(self, result[1])
            return result

        def operation(current_state):
            self._latest_score_lifecycle_audit = {
                "generation": self._generation_from_state(current_state),
                "ordering": [],
                "diagnostic_seconds": 0.0,
                "score_update_seconds": 0.0,
                "eligibility_seconds": 0.0,
                "distribution_refresh_seconds": 0.0,
                "replacement_sampling_seconds": None,
            }
            if not self._diagnostic_due(current_state):
                return self._weighted_step(current_state)
            self._diagnostic_context = self._make_pre_exposure_context(
                current_state
            )
            self._diagnostic_active = True
            try:
                return self._weighted_step(current_state)
            finally:
                del self._diagnostic_context
                del self._diagnostic_active

        result = self._with_generation_context(state, operation)
        self._record_budget_step(state, result)
        return result

    def _record_budget_step(self, before, result):
        audit = getattr(self, "_budget_audit", None)
        if audit is None:
            return
        train_metrics, after = result
        get_int = lambda value: int(jax.device_get(value))
        expected_ec_key = jax.random.split(before.ec_opt_state.key, 4)[0]
        observed_tell = int(bool(jax.device_get(jnp.array_equal(
            after.ec_opt_state.key, expected_ec_key
        ))))
        audit.observe_step(
            episodes=get_int(after.metrics.sampled_episodes - before.metrics.sampled_episodes),
            timesteps=get_int(after.metrics.sampled_timesteps - before.metrics.sampled_timesteps),
            rl_updates=get_int(train_metrics.num_updates_per_iter),
            batch_size=int(self.replay_buffer.sample_batch_size),
            ea_ask=self._observed_ec_rollouts,
            ea_tell=observed_tell,
            actor_optimizer_steps=(
                _optimizer_step_count(after.opt_state.actor)
                - _optimizer_step_count(before.opt_state.actor)
            ),
            critic_optimizer_steps=(
                _optimizer_step_count(after.opt_state.critic)
                - _optimizer_step_count(before.opt_state.critic)
            ),
        )
        self._observed_ec_rollouts = 0

    def learn(self, state):
        """Validate pilot budget from observed state deltas at run completion."""
        get_int = lambda value: int(jax.device_get(value))
        audit = BudgetAudit(
            self.config,
            initial_episodes=get_int(state.metrics.sampled_episodes),
            initial_iteration=get_int(state.metrics.iterations),
            initial_timesteps=get_int(state.metrics.sampled_timesteps),
        )
        self._budget_audit = audit
        self._observed_ec_rollouts = 0
        try:
            final_state = super().learn(state)
            self._last_budget_audit = audit.finalize(
                final_episodes=get_int(final_state.metrics.sampled_episodes),
                final_iteration=get_int(final_state.metrics.iterations),
                final_timesteps=get_int(final_state.metrics.sampled_timesteps),
            )
            return final_state
        finally:
            del self._budget_audit
            del self._observed_ec_rollouts

    def _evaluate_with_count(self, state):
        """Baseline evaluation computation plus the raw result cardinality."""
        key, eval_key = jax.random.split(state.key, num=2)
        raw = jax.vmap(
            self.evaluator.evaluate,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            state.agent_state,
            jax.random.split(eval_key, self.config.num_rl_agents),
            self.config.eval_episodes,
        )
        metrics = EvaluateMetric(
            rl_episode_returns=raw.episode_returns.mean(-1),
            rl_episode_lengths=raw.episode_lengths.mean(-1),
        )
        return metrics, state.replace(key=key), raw.episode_returns.size

    def evaluate(self, state):
        metrics, next_state, actual_episodes = self._evaluate_with_count(state)
        audit = getattr(self, "_budget_audit", None)
        if audit is not None:
            audit.observe_eval(
                int(jax.device_get(state.metrics.iterations)),
                episodes=int(actual_episodes),
            )
        return metrics, next_state

    def warmup_step(self, state):
        self._provenance_generation = state.metrics.iterations + jnp.uint32(1)
        try:
            return super().warmup_step(state)
        finally:
            del self._provenance_generation

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        pre_insertion_replay = replay_buffer_state
        if getattr(self, "_budget_audit", None) is not None:
            self._observed_ec_rollouts += 1
        if getattr(self, "_diagnostic_active", False):
            return ERLLearnerUtilityWorkflow._ec_rollout(
                self, agent_state, replay_buffer_state, key
            )
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
        observer = getattr(self, "_calibration_observer", None)
        if observer is not None and hasattr(observer, "context"):
            observer.score(self, pre_insertion_replay, metrics, trajectory)
        return metrics, trajectory, replay_buffer_state

    def attach_calibration_observer(self, observer):
        """Attach only to an isolated, validated uniform calibration workflow."""
        if self.config.arm != "uniform" or float(self.config.alpha) != 0 or self._diagnostic_enabled:
            raise ValueError("calibration observer requires baseline uniform arm")
        if hasattr(self, "_calibration_observer"):
            raise ValueError("calibration observer already attached")
        self._calibration_observer = observer

    def _process_pre_exposure_ec_trajectory(
        self,
        raw_trajectory,
        replay_buffer_state,
        diagnostic,
        *,
        return_trajectory=False,
    ):
        """Score the raw population, then insert it with the matching version."""
        audit = self._latest_score_lifecycle_audit
        audit["pre_exposure_buffer_size"] = int(
            jax.device_get(replay_buffer_state.buffer_size)
        )
        audit["pre_exposure_current_index"] = int(
            jax.device_get(replay_buffer_state.current_index)
        )
        audit["ordering"].append("diagnostic_start")
        diagnostic_start = time.perf_counter()
        records = diagnostic(raw_trajectory, replay_buffer_state)
        audit["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
        audit["ordering"].append("diagnostic_complete")

        score_start = time.perf_counter()
        score_kind_name = str(getattr(self.config, "score_kind", "utility"))
        if score_kind_name not in _SCORE_KIND_BY_NAME:
            raise ValueError(f"unsupported score_kind: {score_kind_name}")
        checkpoint = apply_diagnostic_checkpoint(
            self._active_utility_replay_state,
            records,
            score_kind=_SCORE_KIND_BY_NAME[score_kind_name],
            population_size=int(self.config.pop_size),
        )
        self._active_utility_replay_state = checkpoint.utility_replay_state
        audit["score_update_seconds"] = time.perf_counter() - score_start
        audit["score_version"] = int(jax.device_get(checkpoint.score_version))
        audit["ordering"].append("score_update_complete")

        self._latest_eval_loss_endpoint = self._evaluate_optional_loss(
            replay_buffer_state,
            reference_fingerprint=(records[0].reference_fingerprint if records else ""),
        )
        if self._latest_eval_loss_endpoint is not None:
            audit["eval_fingerprint"] = (
                self._latest_eval_loss_endpoint.eval_fingerprint
            )
            audit["ordering"].append("eval_loss_complete")

        trajectory = raw_trajectory.replace(next_obs=None)
        trajectory = flatten_pop_rollout_episode(trajectory)
        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = trajectory.replace(dones=None)
        trajectory, mask = tree_stop_gradient(
            flatten_rollout_trajectory((trajectory, mask))
        )
        flat_size = jtu.tree_leaves(trajectory)[0].shape[0]
        episode_batch = int(self.config.episodes_for_fitness)
        time_size = flat_size // (int(self.config.pop_size) * episode_batch)
        provenance = make_ea_rollout_provenance(
            generation=self._provenance_generation,
            population_size=int(self.config.pop_size),
            time_size=time_size,
            episode_batch=episode_batch,
            score_version=checkpoint.score_version,
        )
        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory, provenance, mask
        )
        audit["post_ea_buffer_size"] = int(
            jax.device_get(replay_buffer_state.buffer_size)
        )
        audit["post_ea_current_index"] = int(
            jax.device_get(replay_buffer_state.current_index)
        )
        audit["ordering"].append("ea_insertion_complete")
        audit["reference_fingerprints"] = tuple(
            sorted({record.reference_fingerprint for record in records})
        )
        self._latest_diagnostic_records = records
        if return_trajectory:
            return records, trajectory, replay_buffer_state
        return records, replay_buffer_state

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
        if hasattr(self, "_latest_score_lifecycle_audit"):
            self._latest_score_lifecycle_audit["ordering"].append(
                "rl_insertion_complete"
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
