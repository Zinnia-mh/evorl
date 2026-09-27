import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.utils.jax_utils import right_shift_with_padding, tree_stop_gradient
from evorl.utils.rl_toolkits import (
    flatten_pop_rollout_episode,
    flatten_rollout_trajectory,
)

from .erl_origin import ERLWorkflow
from .learner_utility import (
    SourceBatch,
    build_source_batches,
    derive_diagnostic_key,
    td3_critic_loss,
    virtual_critic_utility,
)


UTILITY_TAG = 0x4C55


class TDStats(NamedTuple):
    mean: jax.Array
    median: jax.Array
    std: jax.Array


@dataclass(frozen=True)
class DiagnosticRecord:
    source_id: str
    fitness: float
    td_error_mean: float | None
    td_error_median: float | None
    td_error_std: float | None
    utility: float | None
    loss_before: float | None
    loss_after: float | None
    source_batch_size: int
    reference_batch_size: int
    reference_fingerprint: str
    eligible: bool
    reason: str | None
    generation: int
    individual_id: int
    measurement_phase: str = "pre_exposure"
    checkpoint: int | None = None
    reference_setting: str = "current_replay_pre_exposure"

    @property
    def td_error(self):
        return self.td_error_mean


def build_checkpoint_records(
    *,
    sources,
    fitnesses,
    reference_fingerprint,
    source_batch_size,
    evaluate,
    reference_batch_size=0,
    generation=0,
    checkpoint=None,
):
    """Build host-only records; callers provide the read-only utility evaluation."""
    records = []
    for source in sources:
        fitness = float(fitnesses[source.individual_id])
        if source.eligible:
            td_error, result = evaluate(source.batch)
            if hasattr(td_error, "mean") and not isinstance(td_error, jax.Array):
                td_mean = float(td_error.mean)
                td_median = float(td_error.median)
                td_std = float(td_error.std)
            else:
                td_mean = td_median = float(td_error)
                td_std = 0.0
            records.append(
                DiagnosticRecord(
                    source_id=source.source_id,
                    fitness=fitness,
                    td_error_mean=td_mean,
                    td_error_median=td_median,
                    td_error_std=td_std,
                    utility=float(result.utility),
                    loss_before=float(result.loss_ref_before),
                    loss_after=float(result.loss_ref_after),
                    source_batch_size=source_batch_size,
                    reference_batch_size=reference_batch_size,
                    reference_fingerprint=reference_fingerprint,
                    eligible=True,
                    reason=None,
                    generation=generation,
                    individual_id=source.individual_id,
                    checkpoint=checkpoint,
                )
            )
        else:
            records.append(
                DiagnosticRecord(
                    source_id=source.source_id,
                    fitness=fitness,
                    td_error_mean=None,
                    td_error_median=None,
                    td_error_std=None,
                    utility=None,
                    loss_before=None,
                    loss_after=None,
                    source_batch_size=source_batch_size,
                    reference_batch_size=reference_batch_size,
                    reference_fingerprint=reference_fingerprint,
                    eligible=False,
                    reason=source.reason,
                    generation=generation,
                    individual_id=source.individual_id,
                    checkpoint=checkpoint,
                )
            )
    return tuple(records)


class ERLLearnerUtilityWorkflow(ERLWorkflow):
    """Original ERL with an optional, read-only learner-utility diagnostic."""

    @classmethod
    def name(cls):
        return "ERL-Learner-Utility"

    @classmethod
    def enable_jit(cls) -> None:
        """JIT baseline kernels while keeping host-side diagnostics eager."""
        cls._rl_rollout = jax.jit(cls._rl_rollout, static_argnums=(0,))
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
        workflow = super()._build_from_config(config)
        workflow._diagnostic_enabled = bool(
            getattr(config, "diagnostic_enabled", False)
        )
        return workflow

    def step(self, state):
        """Scope pre-exposure diagnostic context around the unchanged parent step."""
        if not self._diagnostic_due(state):
            return super().step(state)
        self._diagnostic_context = self._make_pre_exposure_context(state)
        self._diagnostic_active = True
        try:
            return super().step(state)
        finally:
            del self._diagnostic_context
            del self._diagnostic_active

    def _diagnostic_due(self, state):
        if not getattr(self, "_diagnostic_enabled", False):
            return False
        interval = int(getattr(self.config, "diagnostic_interval", 1))
        generation = int(jax.device_get(state.metrics.iterations)) + 1
        return generation % interval == 0

    @staticmethod
    def _reference_batch_fingerprint(batch):
        """Stable host-side identity for the one shared checkpoint reference batch."""
        digest = hashlib.sha256()
        for leaf in jtu.tree_leaves(batch):
            if leaf is not None:
                digest.update(jax.device_get(leaf).tobytes())
        return digest.hexdigest()

    def _make_pre_exposure_context(self, state):
        """Read the current first learner and critic Optax state without write-back."""
        learner_params = jtu.tree_map(
            lambda leaf: leaf[0], state.agent_state.params
        )
        learner_agent_state = state.agent_state.replace(params=learner_params)
        critic_opt_state = jtu.tree_map(
            lambda leaf: leaf if leaf.ndim == 0 else leaf[0],
            state.opt_state.critic,
        )
        return SimpleNamespace(
            generation=int(jax.device_get(state.metrics.iterations)) + 1,
            key=state.key,
            learner_agent_state=learner_agent_state,
            critic_opt_state=critic_opt_state,
            fitnesses=None,
        )

    def _sample_current_reference(self, state, key):
        """Read one Setting-A reference batch without changing replay state."""
        replay_buffer_state = getattr(state, "replay_buffer_state", state)
        reference = self.replay_buffer.sample(replay_buffer_state, key)
        return reference, self._reference_batch_fingerprint(reference)

    def _run_pre_exposure_diagnostic(self, raw_trajectory, replay_buffer_state):
        """Measure every eligible raw population source against one pre-add reference."""
        context = self._diagnostic_context
        diagnostic_key = derive_diagnostic_key(
            context.key, context.generation, UTILITY_TAG
        )
        source_key, reference_key, evaluation_key = jax.random.split(
            diagnostic_key, 3
        )
        sources = build_source_batches(
            raw_trajectory,
            generation=context.generation,
            source_batch_size=self.config.source_batch_size,
            key=source_key,
        )
        if not bool(jax.device_get(self.replay_buffer.can_sample(replay_buffer_state))):
            return tuple(
                DiagnosticRecord(
                    source_id=source.source_id,
                    fitness=float(context.fitnesses[source.individual_id]),
                    td_error_mean=None,
                    td_error_median=None,
                    td_error_std=None,
                    utility=None,
                    loss_before=None,
                    loss_after=None,
                    source_batch_size=self.config.source_batch_size,
                    reference_batch_size=0,
                    reference_fingerprint="",
                    eligible=False,
                    reason="insufficient_reference_replay",
                    generation=context.generation,
                    individual_id=source.individual_id,
                    checkpoint=context.generation,
                )
                for source in sources
            )
        reference, fingerprint = self._sample_current_reference(
            replay_buffer_state, reference_key
        )
        reference_batch_size = int(jtu.tree_leaves(reference)[0].shape[0])
        individual_by_batch = {
            id(source.batch): source.individual_id
            for source in sources
            if source.eligible
        }

        def evaluate(batch):
            individual_id = individual_by_batch[id(batch)]
            key = jax.random.fold_in(evaluation_key, individual_id)
            return self._evaluate_source_utility(batch, reference, key)

        return build_checkpoint_records(
            sources=sources,
            fitnesses=context.fitnesses,
            reference_fingerprint=fingerprint,
            source_batch_size=self.config.source_batch_size,
            reference_batch_size=reference_batch_size,
            generation=context.generation,
            checkpoint=context.generation,
            evaluate=evaluate,
        )

    def _evaluate_source_utility(self, source_batch, reference_batch, key):
        """Use the active TD3 objective and a discarded functional Optax update."""
        context = self._diagnostic_context
        learner_state = context.learner_agent_state
        critic_params = learner_state.params.critic_params
        td_key, utility_key = jax.random.split(key)

        def loss_fn(params, batch, loss_key):
            virtual_state = learner_state.replace(
                params=learner_state.params.replace(critic_params=params)
            )
            return td3_critic_loss(self.agent, virtual_state, batch, loss_key)

        batch_size = jtu.tree_leaves(source_batch)[0].shape[0]

        def per_transition_loss(single_transition, loss_key):
            singleton_batch = jtu.tree_map(
                lambda leaf: leaf[None, ...], single_transition
            )
            return td3_critic_loss(
                self.agent, learner_state, singleton_batch, loss_key
            )

        td_losses = jax.vmap(per_transition_loss)(
            source_batch, jax.random.split(td_key, batch_size)
        )
        td_stats = TDStats(
            mean=jnp.mean(td_losses),
            median=jnp.median(td_losses),
            std=jnp.std(td_losses),
        )
        result = virtual_critic_utility(
            loss_fn=loss_fn,
            optimizer=self.optimizer,
            params=critic_params,
            opt_state=context.critic_opt_state,
            source_batch=source_batch,
            reference_batch=reference_batch,
            key=utility_key,
        )
        return td_stats, result

    def _process_pre_exposure_ec_trajectory(
        self,
        raw_trajectory,
        replay_buffer_state,
        diagnostic,
        *,
        return_trajectory=False,
    ):
        """Measure raw population data before reproducing baseline replay add."""
        records = diagnostic(raw_trajectory, replay_buffer_state)

        trajectory = raw_trajectory.replace(next_obs=None)
        trajectory = flatten_pop_rollout_episode(trajectory)
        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = trajectory.replace(dones=None)
        trajectory, mask = tree_stop_gradient(
            flatten_rollout_trajectory((trajectory, mask))
        )
        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory, mask
        )
        if return_trajectory:
            return records, trajectory, replay_buffer_state
        return records, replay_buffer_state

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        """Baseline EC rollout with an optional pre-exposure read-only hook."""
        if not getattr(self, "_diagnostic_active", False):
            return super()._ec_rollout(agent_state, replay_buffer_state, key)

        eval_metrics, raw_trajectory = jax.vmap(
            self.ec_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, self.config.pop_size),
            self.config.episodes_for_fitness,
        )
        if hasattr(self, "_diagnostic_context"):
            self._diagnostic_context.fitnesses = (
                eval_metrics.episode_returns.mean(axis=-1)
            )
        records, trajectory, replay_buffer_state = (
            self._process_pre_exposure_ec_trajectory(
                raw_trajectory,
                replay_buffer_state,
                self._run_pre_exposure_diagnostic,
                return_trajectory=True,
            )
        )
        self._latest_diagnostic_records = records
        return eval_metrics, trajectory, replay_buffer_state

    def _run_read_only_hook(
        self,
        state,
        *,
        sources,
        fitnesses,
        reference_fingerprint,
        source_batch_size,
        evaluate,
    ):
        """Build checkpoint diagnostics without replacing or updating ``state``.

        All mutable-looking objects needed by a real diagnostic (virtual critic
        parameters, virtual optimizer state, and diagnostic keys) belong to the
        supplied evaluator.  This orchestration layer only reads ``state`` and
        returns host-side records.
        """
        del state
        return build_checkpoint_records(
            sources=sources,
            fitnesses=fitnesses,
            reference_fingerprint=reference_fingerprint,
            source_batch_size=source_batch_size,
            evaluate=evaluate,
        )
