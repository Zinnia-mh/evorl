"""Observer-only learner-utility source-control workflow."""

from __future__ import annotations

import hashlib

import jax

from .erl_learner_utility import ERLLearnerUtilityWorkflow
from .utility_source_controls import FrozenDiagnosticSnapshot, tree_fingerprint


SOURCE_CONTROL_TAG = 0x5343


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
        return context
