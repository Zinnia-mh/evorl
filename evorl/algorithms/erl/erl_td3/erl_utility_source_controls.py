"""Observer-only learner-utility source-control workflow."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.tree_util as jtu

from .erl_learner_utility import ERLLearnerUtilityWorkflow
from .learner_utility import (
    build_source_batches,
    derive_diagnostic_key,
    td3_critic_loss,
)
from .utility_source_controls import (
    CheckpointDiagnostic,
    FrozenDiagnosticSnapshot,
    SourceControlBatch,
    build_rl_own_source_batch,
    compute_source_td_stats,
    derive_source_update_key,
    evaluate_frozen_source_controls,
    tree_fingerprint,
)
from .utility_source_control_artifacts import UtilitySourceControlArtifactWriter


SOURCE_CONTROL_TAG = 0x5343
SOURCE_CONTROL_REFERENCE_TAG = 0x53435246
SOURCE_CONTROL_REPLAY_RANDOM_TAG = 0x53435252
SOURCE_CONTROL_EA_TAG = 0x53434541
SOURCE_CONTROL_RL_OWN_TAG = 0x5343524C
SOURCE_CONTROL_ALLOWED_ARTIFACT_ROOT = Path(
    "/disk1/zmh_projects/evorl_utility_source_controls_outputs"
)
SOURCE_CONTROL_FORBIDDEN_ROOTS = (
    Path("/disk1/zmh_projects/evorl"),
    Path("/disk1/zmh_projects/evorl_utility_source_controls"),
)


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

    @classmethod
    def enable_jit(cls) -> None:
        """JIT inherited baseline kernels while leaving host hooks eager."""
        ERLLearnerUtilityWorkflow.enable_jit()

    @classmethod
    def _build_from_config(cls, config):
        workflow = super()._build_from_config(config)
        workflow._diagnostic_enabled = bool(
            getattr(config, "diagnostic_enabled", False)
        )
        workflow._source_control_writer = None
        workflow._source_control_writer_finalized = False
        if not workflow._diagnostic_enabled:
            return workflow

        artifact_root = getattr(config, "artifact_root", None)
        if artifact_root is None:
            raise ValueError(
                "artifact_root is required when source-control diagnostic is enabled"
            )
        workflow._source_control_writer = UtilitySourceControlArtifactWriter(
            Path(artifact_root),
            allowed_root=SOURCE_CONTROL_ALLOWED_ARTIFACT_ROOT,
            forbidden_roots=SOURCE_CONTROL_FORBIDDEN_ROOTS,
            run_config=cls._source_control_run_config(config),
        )
        return workflow

    @staticmethod
    def _source_control_numerical_tolerance(config):
        numerical_tolerance = getattr(
            config, "source_control_numerical_tolerance", None
        )
        if numerical_tolerance is None:
            numerical_tolerance = 1e-6
        return float(numerical_tolerance)

    @classmethod
    def _source_control_run_config(cls, config):
        numerical_tolerance = cls._source_control_numerical_tolerance(config)
        zero_step_tolerance = getattr(config, "zero_step_tolerance", None)
        if zero_step_tolerance is None:
            zero_step_tolerance = numerical_tolerance
        run_config = {
            "source_batch_size": int(config.source_batch_size),
            "num_reference_batches": int(config.num_reference_batches),
            "num_random_control_batches": int(config.num_random_control_batches),
            "step_scales": list(config.source_control_step_scales),
            "numerical_tolerance": float(numerical_tolerance),
            "zero_step_tolerance": float(zero_step_tolerance),
            "reference_self_warning_threshold": float(
                config.reference_self_warning_threshold
            ),
            "reference_self_systemic_negative_threshold": float(
                config.reference_self_systemic_negative_threshold
            ),
            "ea_mismatch_negative_rate_margin": float(
                config.ea_mismatch_negative_rate_margin
            ),
            "step_sensitivity_negative_rate_drop": float(
                config.step_sensitivity_negative_rate_drop
            ),
            "case_b_negative_rate_min": float(config.case_b_negative_rate_min),
            "case_b_negative_rate_max": float(config.case_b_negative_rate_max),
        }
        worktree_path = Path(
            subprocess.run(
                ("git", "rev-parse", "--show-toplevel"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        common_dir = Path(
            subprocess.run(
                ("git", "rev-parse", "--git-common-dir"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if not common_dir.is_absolute():
            common_dir = (worktree_path / common_dir).resolve()
        repository_path = common_dir.parent if common_dir.name == ".git" else common_dir
        gpu_model = None
        gpu_driver_version = None
        try:
            gpu_line = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0]
            gpu_model, gpu_driver_version = (
                value.strip() for value in gpu_line.split(",", maxsplit=1)
            )
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
            pass
        run_config.update(
            {
                "repository_path": str(repository_path),
                "worktree_path": str(worktree_path),
                "branch": subprocess.run(
                    ("git", "branch", "--show-current"),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "commit": subprocess.run(
                    ("git", "rev-parse", "HEAD"),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "artifact_root": str(Path(config.artifact_root).resolve()),
                "tmux_session": getattr(config, "tmux_session", None),
                "tmux_window": getattr(config, "tmux_window", None),
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "python_version": sys.version.split()[0],
                "jax_version": version("jax"),
                "jaxlib_version": version("jaxlib"),
                "flax_version": version("flax"),
                "optax_version": version("optax"),
                "device": ", ".join(str(device) for device in jax.devices()),
                "backend": jax.default_backend(),
                "gpu_model": gpu_model,
                "gpu_driver_version": gpu_driver_version,
                "environment": getattr(
                    config, "env_name", getattr(config, "environment", None)
                ),
                "seed": (
                    int(config.seed) if getattr(config, "seed", None) is not None else None
                ),
                "population": (
                    int(config.pop_size)
                    if getattr(config, "pop_size", None) is not None
                    else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return run_config

    def step(self, state):
        """Persist a due host diagnostic after the unchanged baseline step."""
        diagnostic_due = self._diagnostic_due(state)
        if diagnostic_due and hasattr(self, "_latest_source_control_diagnostic"):
            del self._latest_source_control_diagnostic
        if diagnostic_due and hasattr(self, "_latest_source_control_provenance"):
            del self._latest_source_control_provenance
        result = super().step(state)
        writer = getattr(self, "_source_control_writer", None)
        if diagnostic_due and writer is not None:
            diagnostic = getattr(self, "_latest_source_control_diagnostic", None)
            if diagnostic is not None:
                writer.append_checkpoint(
                    diagnostic,
                    checkpoint_provenance=getattr(
                        self, "_latest_source_control_provenance", None
                    ),
                )
                finalize_checkpoint = getattr(
                    writer, "finalize_checkpoint", None
                )
                if finalize_checkpoint is not None:
                    finalize_checkpoint()
        return result

    def close(self) -> None:
        """Finalize source-control artifacts once during normal shutdown."""
        try:
            super().close()
        finally:
            writer = getattr(self, "_source_control_writer", None)
            if writer is not None and not getattr(
                self, "_source_control_writer_finalized", False
            ):
                writer.finalize_run()
                self._source_control_writer_finalized = True

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

    def _stage_rl_own_source(self, raw_trajectory, replay_buffer_state):
        """Retain one matched source from the real baseline RL rollout."""
        del replay_buffer_state
        context = self._diagnostic_context
        source_key = derive_diagnostic_key(
            context.key,
            context.generation,
            SOURCE_CONTROL_RL_OWN_TAG,
        )
        context.rl_source = build_rl_own_source_batch(
            raw_trajectory,
            generation=context.generation,
            source_batch_size=int(self.config.source_batch_size),
            key=source_key,
        )
        return ()

    def _run_deferred_source_control_diagnostic(
        self, context, rl_source, replay_after_rl
    ):
        """Evaluate all controls from the one frozen pre-exposure state."""
        snapshot = context.snapshot
        self._latest_source_control_provenance = {
            "baseline_prng_fingerprint": snapshot.baseline_key_fingerprint,
            "replay_metadata_fingerprint": snapshot.replay_metadata_fingerprint,
            "temporal_alignment": context.source_temporal_alignment,
        }
        if not context.controls_eligible:
            diagnostic = CheckpointDiagnostic.from_records(
                checkpoint=context.generation,
                temporal_alignment=context.source_temporal_alignment,
                records=(),
                measurement_eligible=False,
                comparison_eligible=False,
                eligibility_reasons=(
                    context.controls_reason or "controls_ineligible",
                ),
            )
            context.checkpoint_diagnostic = diagnostic
            self._latest_source_control_diagnostic = diagnostic
            return diagnostic

        learner_state = snapshot.learner_agent_state
        critic_params = learner_state.params.critic_params
        replay_before = _replay_metadata_fingerprint(replay_after_rl)
        frozen_before = (
            tree_fingerprint(learner_state),
            tree_fingerprint(snapshot.critic_opt_state),
        )

        def loss_fn(params, batch, loss_key):
            virtual_state = learner_state.replace(
                params=learner_state.params.replace(critic_params=params)
            )
            return td3_critic_loss(self.agent, virtual_state, batch, loss_key)

        replay_sources = tuple(
            SourceControlBatch(
                source_type="Replay-random",
                source_id=f"replay_random_{index}",
                batch=batch,
            )
            for index, batch in enumerate(snapshot.replay_control_batches)
        )
        ordinary_sources = (
            tuple(context.ea_sources) + (rl_source,) + replay_sources
        )
        update_key = derive_source_update_key(
            snapshot.baseline_key, snapshot.generation
        )
        td_sources = ordinary_sources + tuple(
            SourceControlBatch(
                source_type="Reference-self",
                source_id=f"reference_self_{index}",
                batch=batch,
            )
            for index, batch in enumerate(snapshot.reference_batches)
        )
        td_stats = {
            source.source_id: compute_source_td_stats(
                agent=self.agent,
                frozen_agent_state=learner_state,
                source_batch=source.batch,
                source_update_key=update_key,
            )
            for source in td_sources
            if source.eligible
        }
        diagnostic = evaluate_frozen_source_controls(
            checkpoint=context.generation,
            generation=snapshot.generation,
            sources=ordinary_sources,
            reference_batches=snapshot.reference_batches,
            reference_fingerprints=snapshot.reference_fingerprints,
            loss_fn=loss_fn,
            optimizer=self.optimizer,
            params=critic_params,
            opt_state=snapshot.critic_opt_state,
            baseline_key=snapshot.baseline_key,
            step_scales=tuple(
                getattr(
                    self.config,
                    "source_control_step_scales",
                    (1.0, 0.5, 0.1),
                )
            ),
            numerical_tolerance=self._source_control_numerical_tolerance(
                self.config
            ),
            temporal_alignment=context.source_temporal_alignment,
            td_stats_by_source=td_stats,
        )
        reasons = list(diagnostic.reasons)
        frozen_after = (
            tree_fingerprint(learner_state),
            tree_fingerprint(snapshot.critic_opt_state),
        )
        if frozen_after != frozen_before:
            reasons.append("frozen_state_mutation")
        if _replay_metadata_fingerprint(replay_after_rl) != replay_before:
            reasons.append("replay_state_mutation")
        if reasons != list(diagnostic.reasons):
            diagnostic = replace(
                diagnostic,
                valid=False,
                comparison_eligible=False,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        context.checkpoint_diagnostic = diagnostic
        self._latest_source_control_diagnostic = diagnostic
        return diagnostic

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        """Stage the real RL rollout and invoke the pre-update deferred boundary."""
        if not getattr(self, "_diagnostic_active", False):
            return super()._rl_rollout(agent_state, replay_buffer_state, key)

        eval_metrics, raw_trajectory = jax.vmap(
            self.rl_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, self.config.num_rl_agents),
            self.config.rollout_episodes,
        )
        _, trajectory, replay_buffer_state = (
            self._process_pre_exposure_ec_trajectory(
                raw_trajectory,
                replay_buffer_state,
                self._stage_rl_own_source,
                return_trajectory=True,
            )
        )
        context = self._diagnostic_context
        current_critic = jtu.tree_map(
            lambda leaf: leaf[0], agent_state.params.critic_params
        )
        context.source_temporal_alignment = (
            "exact"
            if tree_fingerprint(current_critic)
            == context.snapshot.critic_fingerprint
            else "delayed"
        )
        self._run_deferred_source_control_diagnostic(
            context,
            context.rl_source,
            replay_buffer_state,
        )
        return eval_metrics, trajectory, replay_buffer_state
