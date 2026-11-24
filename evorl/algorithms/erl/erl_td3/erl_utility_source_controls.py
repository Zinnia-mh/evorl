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
import uuid

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .erl_learner_utility import ERLLearnerUtilityWorkflow
from .erl_td3_workflow import erl_replace_td3_actor_params
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
from .pilot_transition_identity import (
    CollectionOrdering, check_disjointness, derive_pilot_sampling_key,
    freeze_replay_snapshot, sample_disjoint_replay_groups, sample_fresh_source,
)
from .pilot_dual_reference import (
    choose_ea_candidates_and_donors, evaluate_paired_utility,
    paired_aggregates, sample_rl_heldout,
)
from .pilot_artifacts import PILOT_ROOT, PilotArtifactWriter
from .pilot_gradient_mechanisms import ReferenceGradientCache
from .utility_source_controls import derive_reference_eval_key


SOURCE_CONTROL_TAG = 0x5343
SOURCE_CONTROL_REFERENCE_TAG = 0x53435246
SOURCE_CONTROL_REPLAY_RANDOM_TAG = 0x53435252
SOURCE_CONTROL_EA_TAG = 0x53434541
SOURCE_CONTROL_RL_OWN_TAG = 0x5343524C
SOURCE_CONTROL_DERIVED_TAG = 0x53434452
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
        workflow._pilot_enabled = bool(getattr(config, "reference_target_bias_enabled", False))
        workflow._pilot_writer = None
        if workflow._pilot_enabled and not workflow._diagnostic_enabled:
            raise ValueError("Pilot requires diagnostic_enabled")
        if workflow._pilot_enabled:
            workflow._pilot_run_id = uuid.uuid4().hex
            workflow._pilot_writer = PilotArtifactWriter(
                getattr(config, "pilot_artifact_root", None) or PILOT_ROOT,
                workflow._pilot_run_id,
            )
            return workflow
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
        python_executable = Path(sys.executable).resolve()
        conda_prefix = Path(sys.prefix).resolve()
        conda_env = conda_prefix.name
        if (
            conda_prefix.parent.name != "envs"
            and os.environ.get("CONDA_PREFIX") == str(conda_prefix)
        ):
            conda_env = os.environ.get("CONDA_DEFAULT_ENV") or conda_env
        hydra_env = getattr(config, "env", None)
        env_type = getattr(hydra_env, "env_type", None)
        env_name = getattr(hydra_env, "env_name", None)
        environment = (
            f"{env_type}/{env_name}"
            if env_type is not None and env_name is not None
            else getattr(config, "env_name", getattr(config, "environment", None))
        )
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
                "python_executable": str(python_executable),
                "conda_prefix": str(conda_prefix),
                "conda_env": conda_env,
                "python_version": sys.version.split()[0],
                "jax_version": version("jax"),
                "jaxlib_version": version("jaxlib"),
                "flax_version": version("flax"),
                "optax_version": version("optax"),
                "device": ", ".join(str(device) for device in jax.devices()),
                "backend": jax.default_backend(),
                "gpu_model": gpu_model,
                "gpu_driver_version": gpu_driver_version,
                "environment": environment,
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
        if getattr(self, "_pilot_enabled", False):
            context.pilot_run_id = self._pilot_run_id
            context.pilot_state_anchor = state
            context.pilot_event_counter = 1
            context.pilot_replay_capture_event = 1
            context.pilot_snapshot = freeze_replay_snapshot(
                state.replay_buffer_state,
                run_id=context.pilot_run_id, generation=context.generation,
            )
            batch_size = int(self.config.source_batch_size)
            if batch_size != int(self.replay_buffer.sample_batch_size):
                raise ValueError("Pilot replay and source batch sizes must match")
            if (int(getattr(self.config, "num_reference_batches", 4)) != 4
                    or int(getattr(self.config, "num_random_control_batches", 8)) != 8):
                raise ValueError("Pilot requires exactly four Replay references and eight controls")
            groups = sample_disjoint_replay_groups(
                self.replay_buffer, context.pilot_snapshot,
                derive_pilot_sampling_key(context.key, generation=context.generation),
                batch_size=batch_size,
                num_references=int(getattr(self.config, "num_reference_batches", 4)),
                num_controls=int(getattr(self.config, "num_random_control_batches", 8)),
            )
            context.pilot_replay_groups = groups
            context.controls_eligible = groups.eligible
            context.controls_reason = groups.ineligible_reason
            if groups.eligible:
                references = tuple(group.batch for group in groups.references)
                controls = tuple(group.batch for group in groups.controls)
                context.snapshot = replace(
                    context.snapshot,
                    reference_batches=references,
                    reference_fingerprints=tuple(tree_fingerprint(x) for x in references),
                    replay_control_batches=controls,
                    replay_control_fingerprints=tuple(tree_fingerprint(x) for x in controls),
                )
            return context

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
        if getattr(self, "_pilot_enabled", False):
            context.pilot_event_counter += 1
            context.pilot_ea_capture_event = context.pilot_event_counter
            context.pilot_ea_raw = raw_trajectory
            candidate_pairs, donor_pairs, reason = choose_ea_candidates_and_donors(
                raw_trajectory, context.ea_fitnesses,
                run_id=context.pilot_run_id, generation=context.generation,
                batch_size=int(self.config.source_batch_size),
                source_key=source_key,
                donor_key=jax.random.fold_in(
                    derive_pilot_sampling_key(context.key, generation=context.generation),
                    0x4541444f,
                ),
            )
            context.pilot_ea_reason = reason
            context.pilot_ea_candidates = candidate_pairs
            context.pilot_ea_donors = donor_pairs
            context.ea_sources = tuple(
                SourceControlBatch(
                    source_type="EA", source_id=f"gen_{context.generation}_ind_{actor}",
                    batch=group.batch, eligible=group.eligible,
                    reason=group.ineligible_reason,
                    valid_transition_count=group.valid_sample_count,
                    evolutionary_fitness=float(jax.device_get(context.ea_fitnesses[actor])),
                )
                for actor, group in candidate_pairs
            )
            return ()
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
        if getattr(self, "_pilot_enabled", False):
            context.pilot_event_counter += 1
            context.pilot_ea_insert_event = context.pilot_event_counter
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
        if getattr(self, "_pilot_enabled", False):
            context.pilot_event_counter += 1
            context.pilot_rl_capture_event = context.pilot_event_counter
            context.pilot_rl_raw = raw_trajectory
            context.pilot_rl_source = sample_fresh_source(
                raw_trajectory, run_id=context.pilot_run_id,
                generation=context.generation, producer="RL", actor_id=None,
                batch_size=int(self.config.source_batch_size), key=source_key,
            )
            context.pilot_rl_references = sample_rl_heldout(
                raw_trajectory, context.pilot_rl_source,
                run_id=context.pilot_run_id, generation=context.generation,
                batch_size=int(self.config.source_batch_size),
                key=jax.random.fold_in(
                    derive_pilot_sampling_key(context.key, generation=context.generation),
                    0x524c5246,
                ),
            )
            context.rl_source = SourceControlBatch(
                source_type="RL-own", source_id=f"gen_{context.generation}_rl_own",
                batch=context.pilot_rl_source.batch,
                eligible=context.pilot_rl_source.eligible,
                reason=context.pilot_rl_source.ineligible_reason,
                valid_transition_count=context.pilot_rl_source.valid_sample_count,
            )
            return ()
        context.rl_source = build_rl_own_source_batch(
            raw_trajectory,
            generation=context.generation,
            source_batch_size=int(self.config.source_batch_size),
            key=source_key,
        )
        return ()

    def _stage_derived_sources(self, agent_state, context):
        if not bool(getattr(self.config, "derived_enabled", False)):
            context.derived_sources, context.derived_timesteps = (), 0
            return
        actor = jtu.tree_map(lambda x: x[0], agent_state.params.actor_params)
        root = derive_diagnostic_key(context.key, context.generation, SOURCE_CONTROL_DERIVED_TAG)
        sources, steps = [], 0
        distances = {}
        for i, (name, strength) in enumerate((
            ("RL-derived-small", float(self.config.derived_small_strength)),
            ("RL-derived-large", float(self.config.derived_large_strength)),
        )):
            mutation_key, rollout_key, sample_key = jax.random.split(jax.random.fold_in(root, i), 3)
            leaves, treedef = jtu.tree_flatten(actor)
            keys = jax.random.split(mutation_key, len(leaves))
            mutated = jtu.tree_unflatten(treedef, [
                x + strength * jax.random.normal(k, x.shape, x.dtype)
                for x, k in zip(leaves, keys)
            ])
            batched = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), mutated)
            derived_state = erl_replace_td3_actor_params(agent_state, batched)
            metrics, trajectory = jax.vmap(
                self.ec_collector.rollout, in_axes=(self.agent_state_vmap_axes, 0, None)
            )(derived_state, jax.random.split(rollout_key, 1), self.config.episodes_for_fitness)
            obs = trajectory.obs.reshape((-1, *trajectory.obs.shape[3:]))
            if self.agent.normalize_obs:
                obs = self.agent.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            base_actions = self.agent.actor_network.apply(actor, obs)
            derived_actions = self.agent.actor_network.apply(mutated, obs)
            distances[name] = float(jax.device_get(
                jnp.mean(jnp.linalg.norm(derived_actions - base_actions, axis=-1))
            ))
            batch = build_rl_own_source_batch(
                trajectory, generation=context.generation,
                source_batch_size=int(self.config.source_batch_size), key=sample_key
            )
            sources.append(replace(batch, source_type=name,
                source_id=f"gen_{context.generation}_{name.lower()}",
                evolutionary_fitness=float(jax.device_get(metrics.episode_returns.mean()))))
            steps += int(jax.device_get(metrics.episode_lengths.sum()))
        context.derived_sources, context.derived_timesteps = tuple(sources), steps
        context.derived_behavior_distances = distances

    def _run_deferred_source_control_diagnostic(
        self, context, rl_source, replay_after_rl
    ):
        """Evaluate all controls from the one frozen pre-exposure state."""
        if getattr(self, "_pilot_enabled", False):
            return self._run_deferred_pilot_diagnostic(context, rl_source, replay_after_rl)
        snapshot = context.snapshot
        self._latest_source_control_provenance = {
            "baseline_prng_fingerprint": snapshot.baseline_key_fingerprint,
            "replay_metadata_fingerprint": snapshot.replay_metadata_fingerprint,
            "temporal_alignment": context.source_temporal_alignment,
        }
        if bool(getattr(self.config, "derived_enabled", False)):
            self._latest_source_control_provenance["derived_extra_timesteps"] = int(getattr(context, "derived_timesteps", 0))
            self._latest_source_control_provenance["derived_behavior_distances"] = dict(getattr(context, "derived_behavior_distances", {}))
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
            tuple(context.ea_sources) + (rl_source,) + tuple(getattr(context, 'derived_sources', ())) + replay_sources
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
        if getattr(self, "_pilot_enabled", False):
            context.pilot_event_counter += 1
            context.pilot_rl_insert_event = context.pilot_event_counter
        current_critic = jtu.tree_map(
            lambda leaf: leaf[0], agent_state.params.critic_params
        )
        context.source_temporal_alignment = (
            "exact"
            if tree_fingerprint(current_critic)
            == context.snapshot.critic_fingerprint
            else "delayed"
        )
        self._stage_derived_sources(agent_state, context)
        self._run_deferred_source_control_diagnostic(
            context,
            context.rl_source,
            replay_buffer_state,
        )
        return eval_metrics, trajectory, replay_buffer_state

    def _run_deferred_pilot_diagnostic(self, context, rl_source, replay_after_rl):
        """Validate physical captures before any paired virtual update."""
        snapshot = context.snapshot
        batch_size = int(self.config.source_batch_size)
        groups = context.pilot_replay_groups
        reasons = []
        if not groups.eligible:
            reasons.append(groups.ineligible_reason)
        reasons.extend([context.pilot_ea_reason] if context.pilot_ea_reason else [])
        ea = tuple(group for _, group in context.pilot_ea_candidates)
        donors = tuple(group for _, group in context.pilot_ea_donors)
        rl = context.pilot_rl_source
        rl_refs = tuple(context.pilot_rl_references)
        if not rl.eligible:
            reasons.append(rl.ineligible_reason)
        reasons.extend(group.ineligible_reason for group in rl_refs if not group.eligible)
        provenance = dict(groups.provenance)
        provenance.update({
            "ea_candidate_actor_ids": [actor for actor, _ in context.pilot_ea_candidates],
            "ea_candidate_roles": dict(zip(
                ("Top", "Mid", "Low"),
                (actor for actor, _ in context.pilot_ea_candidates),
            )),
            "ea_donor_actor_ids": [actor for actor, _ in context.pilot_ea_donors],
            "ea_capture_id": ea[0].fresh_capture.capture_id if ea and ea[0].fresh_capture else None,
            "rl_capture_id": rl.fresh_capture.capture_id if rl.fresh_capture else None,
            "rl_own_indices": list(rl.selected_indices),
            "rl_fresh_indices": [list(group.selected_indices) for group in rl_refs],
            "ea_candidate_indices": [list(group.selected_indices) for group in ea],
            "ea_donor_indices": [list(group.selected_indices) for group in donors],
            "event_ordinals": {
                "replay_capture": context.pilot_replay_capture_event,
                "ea_capture": context.pilot_ea_capture_event,
                "ea_insert": context.pilot_ea_insert_event,
                "rl_capture": context.pilot_rl_capture_event,
                "rl_insert": context.pilot_rl_insert_event,
            },
        })
        if not reasons:
            ea_order = CollectionOrdering(
                run_id=context.pilot_run_id, generation=context.generation,
                snapshot_id=groups.snapshot_id,
                replay_capture_event=context.pilot_replay_capture_event,
                fresh_capture_event=context.pilot_ea_capture_event,
                replay_insert_event=context.pilot_ea_insert_event,
                fresh_capture_id=ea[0].fresh_capture.capture_id,
            )
            rl_order = CollectionOrdering(
                run_id=context.pilot_run_id, generation=context.generation,
                snapshot_id=groups.snapshot_id,
                replay_capture_event=context.pilot_replay_capture_event,
                fresh_capture_event=context.pilot_rl_capture_event,
                replay_insert_event=context.pilot_rl_insert_event,
                fresh_capture_id=rl.fresh_capture.capture_id,
            )
            checks = (
                check_disjointness(groups.controls, groups.references, batch_size=batch_size),
                check_disjointness(ea, donors, batch_size=batch_size),
                check_disjointness((rl,), rl_refs, batch_size=batch_size),
                check_disjointness(ea, groups.references, batch_size=batch_size, ordering=ea_order),
                check_disjointness((rl,), groups.references, batch_size=batch_size, ordering=rl_order),
                check_disjointness(groups.controls, donors, batch_size=batch_size, ordering=ea_order),
                check_disjointness(groups.controls, rl_refs, batch_size=batch_size, ordering=rl_order),
                check_disjointness(ea, rl_refs, batch_size=batch_size),
                check_disjointness((rl,), donors, batch_size=batch_size),
            )
            reasons.extend(check.ineligible_reason for check in checks if not check.eligible)
            provenance["gate_results"] = [
                {"eligible": check.eligible, "reason": check.ineligible_reason,
                 "overlap_count": check.overlap_count} for check in checks
            ]
        learner_state = snapshot.learner_agent_state
        def invariant_fingerprints():
            anchor = context.pilot_state_anchor
            counters = {
                name: getattr(anchor.metrics, name)
                for name in (
                    "iterations", "sampled_episodes", "sampled_timesteps",
                    "rl_sampled_timesteps",
                ) if hasattr(anchor.metrics, name)
            }
            ec_state = getattr(anchor, "ec_state", None)
            return (
                tree_fingerprint(learner_state),
                tree_fingerprint(snapshot.critic_opt_state),
                tree_fingerprint(snapshot.baseline_key),
                _replay_metadata_fingerprint(replay_after_rl),
                id(replay_after_rl),
                tree_fingerprint(anchor.agent_state),
                tree_fingerprint(anchor.opt_state),
                tree_fingerprint(anchor.key),
                tree_fingerprint(counters),
                tree_fingerprint(ec_state) if ec_state is not None else None,
            )
        frozen_before = invariant_fingerprints()
        results = {}
        aggregates = {}
        failed_sources = []
        if not reasons:
            def loss_fn(params, batch, key):
                virtual_state = learner_state.replace(
                    params=learner_state.params.replace(critic_params=params)
                )
                return td3_critic_loss(self.agent, virtual_state, batch, key)

            references = {
                "Replay": snapshot.reference_batches,
                "RL-fresh": tuple(group.batch for group in rl_refs),
                "EA-fresh": tuple(group.batch for group in donors),
            }
            reference_keys = {
                "Replay": tuple(
                    derive_reference_eval_key(snapshot.baseline_key, snapshot.generation, i)
                    for i in range(4)
                ),
                "RL-fresh": tuple(
                    jax.random.fold_in(
                        derive_diagnostic_key(snapshot.baseline_key, snapshot.generation, 0x524c4556), i
                    ) for i in range(2)
                ),
                "EA-fresh": tuple(
                    jax.random.fold_in(
                        derive_diagnostic_key(snapshot.baseline_key, snapshot.generation, 0x45414556), i
                    ) for i in range(2)
                ),
            }
            sources = (
                tuple((source.source_id, group.batch)
                      for source, group in zip(context.ea_sources, ea, strict=True))
                + ((rl_source.source_id, rl.batch),)
                + tuple((f"replay_random_{i}", group.batch)
                        for i, group in enumerate(groups.controls))
            )
            mechanism_cache = ReferenceGradientCache(
                checkpoint=context.generation,
                frozen_learner_fingerprint=frozen_before[0],
                frozen_critic_fingerprint=snapshot.critic_fingerprint,
                tolerance=self._source_control_numerical_tolerance(self.config),
            )
            for source_id, batch in sources:
                try:
                    result = evaluate_paired_utility(
                    loss_fn=loss_fn, optimizer=self.optimizer,
                    params=learner_state.params.critic_params,
                    opt_state=snapshot.critic_opt_state,
                    source_batch=batch, references=references,
                    source_update_key=derive_source_update_key(
                        snapshot.baseline_key, snapshot.generation
                    ),
                    reference_keys=reference_keys,
                    step_scales=tuple(self.config.source_control_step_scales),
                    tolerance=self._source_control_numerical_tolerance(self.config),
                    mechanism_cache=mechanism_cache,
                    )
                except FloatingPointError:
                    reasons.append("non_finite_gradient_or_utility")
                    failed_sources.append({
                        "source_id": source_id,
                        "reason": "non_finite_gradient_or_utility",
                        "source_batch_fingerprint": tree_fingerprint(batch),
                    })
                    continue
                except ValueError as error:
                    if "reference_cache_mismatch" not in str(error):
                        raise
                    reasons.append("reference_cache_mismatch")
                    failed_sources.append({
                        "source_id": source_id,
                        "reason": "reference_cache_mismatch",
                        "source_batch_fingerprint": tree_fingerprint(batch),
                    })
                    continue
                results[source_id] = result
                reasons.extend(result.reasons)
                if len(result.mechanism_records) != (
                        len(result.records) + len(result.zero_step_results)):
                    reasons.append("mechanism_utility_key_mismatch")
                for record in result.mechanism_records:
                    if record.step_scale == 0.0 and (
                            record.actual_parameter_displacement_norm != 0.0
                            or record.first_order_predicted_utility != 0.0
                            or abs(record.actual_utility)
                            > self._source_control_numerical_tolerance(self.config)):
                        reasons.append("mechanism_zero_step_failure")
            provenance["mechanism_failed_sources"] = failed_sources
            provenance["reference_gradient_fingerprints"] = {
                f"{family}/{index}": entry.gradient_fingerprint
                for (family, index), entry in mechanism_cache.entries.items()
            }
            provenance["reference_eval_key_fingerprints"] = {
                f"{family}/{index}": entry.reference_eval_key_fingerprint
                for (family, index), entry in mechanism_cache.entries.items()
            }
            provenance["source_gradient_fingerprints"] = {
                name: result.mechanism_records[0].source_gradient_fingerprint
                for name, result in results.items() if result.mechanism_records
            }
            if not reasons:
                for family in ("Replay", "RL-fresh", "EA-fresh"):
                    count = len(references[family])
                    for index in range(count):
                        values = [
                            record.loss_ref_before
                            for result in results.values()
                            for record in result.records
                            if record.reference_family == family
                            and record.reference_index == index
                        ]
                        if (not values or not all(
                                __import__("math").isfinite(value) for value in values)
                                or max(values) - min(values)
                                > self._source_control_numerical_tolerance(self.config)):
                            reasons.append("common_reference_before_mismatch")
            if not reasons:
                for scale in self.config.source_control_step_scales:
                    aggregates[str(scale)] = paired_aggregates(
                        results, scale=scale, replay_source_id="replay_random_0"
                    )
        frozen_after = invariant_fingerprints()
        if frozen_before != frozen_after:
            reasons.append("frozen_state_or_replay_mutation")
        reasons = tuple(dict.fromkeys(reasons))
        diagnostic = CheckpointDiagnostic.from_records(
            checkpoint=context.generation,
            temporal_alignment=context.source_temporal_alignment,
            records=(), reasons=reasons,
            measurement_eligible=not reasons,
            comparison_eligible=not reasons,
            eligibility_reasons=reasons,
            source_statuses=tuple(context.ea_sources) + (rl_source,),
        )
        provenance["source_update_fingerprints"] = {
            name: result.update_fingerprint for name, result in results.items()
        }
        self._pilot_writer.append_checkpoint(
            generation=context.generation,
            status="HOLD" if reasons else "PASS",
            reasons=reasons, source_results=results,
            aggregates=aggregates, provenance=provenance,
            seed=getattr(self.config, "seed", None),
            source_types={
                **{source.source_id: source.source_type for source in context.ea_sources},
                rl_source.source_id: rl_source.source_type,
                **{f"replay_random_{i}": "Replay-random"
                   for i in range(len(groups.controls))},
            },
            frozen_critic_fingerprint=snapshot.critic_fingerprint,
            require_mechanisms=True, failed_sources=failed_sources,
        )
        context.checkpoint_diagnostic = diagnostic
        self._latest_source_control_diagnostic = diagnostic
        return diagnostic
