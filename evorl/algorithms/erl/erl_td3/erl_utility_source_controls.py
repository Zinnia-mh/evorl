"""Observer-only learner-utility source-control workflow."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import jax

from .erl_learner_utility import ERLLearnerUtilityWorkflow
from .learner_utility import build_source_batches, derive_diagnostic_key
from .utility_source_controls import (
    FrozenDiagnosticSnapshot,
    SourceControlBatch,
    tree_fingerprint,
)


SOURCE_CONTROL_TAG = 0x5343
SOURCE_CONTROL_REFERENCE_TAG = 0x53435246
SOURCE_CONTROL_REPLAY_RANDOM_TAG = 0x53435252
SOURCE_CONTROL_EA_TAG = 0x53434541


def _replay_metadata_fingerprint(replay_state) -> str:
    """Fingerprint replay cursors and structure without hashing capacity data."""
    digest = hashlib.sha256()
    found = False
    for name in ("current_size", "current_position", "size", "position"):
        if hasattr(replay_state, name):
            digest.update(name.encode())
            digest.update(tree_fingerprint(getattr(replay_state, name)).encode())
            found = True
    if not found:
        for leaf in jax.tree_util.tree_leaves(replay_state):
            digest.update(str(getattr(leaf, "shape", None)).encode())
            digest.update(str(getattr(leaf, "dtype", type(leaf))).encode())
    return digest.hexdigest()


class ERLUtilitySourceControlsWorkflow(ERLLearnerUtilityWorkflow):
    """Compare frozen source controls without changing baseline training."""

    @classmethod
    def name(cls):
        return "ERL-Utility-Source-Controls"

    def _make_pre_exposure_context(self, state):
        context = super()._make_pre_exposure_context(state)
        counters = {
            name: getattr(state.metrics, name)
            for name in (
                "iterations",
                "sampled_episodes",
                "sampled_timesteps",
                "rl_sampled_timesteps",
            )
            if hasattr(state.metrics, name)
        }
        params = context.learner_agent_state.params
        context.snapshot = FrozenDiagnosticSnapshot(
            generation=context.generation,
            learner_agent_state=context.learner_agent_state,
            critic_opt_state=context.critic_opt_state,
            baseline_key=context.key,
            baseline_key_fingerprint=tree_fingerprint(context.key),
            critic_fingerprint=tree_fingerprint(params.critic_params),
            target_critic_fingerprint=tree_fingerprint(params.target_critic_params),
            optimizer_fingerprint=tree_fingerprint(context.critic_opt_state),
            replay_metadata_fingerprint=_replay_metadata_fingerprint(
                state.replay_buffer_state
            ),
            counter_fingerprint=tree_fingerprint(counters),
        )
        context.controls_eligible = False
        context.controls_reason = None

        expected_batch_size = int(self.config.source_batch_size)
        replay_batch_size = int(self.replay_buffer.sample_batch_size)
        if replay_batch_size != expected_batch_size:
            raise ValueError(
                "replay sample_batch_size must equal diagnostic source_batch_size: "
                f"{replay_batch_size} != {expected_batch_size}"
            )

        if not bool(
            jax.device_get(
                self.replay_buffer.can_sample(state.replay_buffer_state)
            )
        ):
            context.controls_reason = "replay_not_ready"
            return context

        reference_namespace = derive_diagnostic_key(
            context.key,
            context.generation,
            SOURCE_CONTROL_REFERENCE_TAG,
        )
        replay_namespace = derive_diagnostic_key(
            context.key,
            context.generation,
            SOURCE_CONTROL_REPLAY_RANDOM_TAG,
        )
        reference_batches = tuple(
            self.replay_buffer.sample(
                state.replay_buffer_state,
                jax.random.fold_in(reference_namespace, index),
            )
            for index in range(
                int(getattr(self.config, "num_reference_batches", 4))
            )
        )
        replay_control_batches = tuple(
            self.replay_buffer.sample(
                state.replay_buffer_state,
                jax.random.fold_in(replay_namespace, index),
            )
            for index in range(
                int(getattr(self.config, "num_random_control_batches", 8))
            )
        )
        context.snapshot = replace(
            context.snapshot,
            reference_batches=reference_batches,
            reference_fingerprints=tuple(
                tree_fingerprint(batch) for batch in reference_batches
            ),
            replay_control_batches=replay_control_batches,
            replay_control_fingerprints=tuple(
                tree_fingerprint(batch) for batch in replay_control_batches
            ),
        )
        context.controls_eligible = True
        return context

    def _stage_ea_sources(self, raw_trajectory, replay_buffer_state):
        """Retain matched EA batches before baseline replay insertion."""
        del replay_buffer_state
        context = self._diagnostic_context
        source_key = derive_diagnostic_key(
            context.key,
            context.generation,
            SOURCE_CONTROL_EA_TAG,
        )
        sources = build_source_batches(
            raw_trajectory,
            generation=context.generation,
            source_batch_size=int(self.config.source_batch_size),
            key=source_key,
        )
        context.ea_sources = tuple(
            SourceControlBatch(
                source_type="EA",
                source_id=source.source_id,
                batch=source.batch,
                eligible=source.eligible,
                reason=source.reason,
                valid_transition_count=source.valid_transition_count,
                evolutionary_fitness=float(
                    jax.device_get(context.ea_fitnesses[source.individual_id])
                ),
            )
            for source in sources
        )
        return ()

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        """Stage raw EA sources, then reproduce the baseline flatten/add path."""
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
        context = self._diagnostic_context
        context.ea_fitnesses = eval_metrics.episode_returns.mean(axis=-1)
        _, trajectory, replay_buffer_state = (
            self._process_pre_exposure_ec_trajectory(
                raw_trajectory,
                replay_buffer_state,
                self._stage_ea_sources,
                return_trajectory=True,
            )
        )
        return eval_metrics, trajectory, replay_buffer_state
