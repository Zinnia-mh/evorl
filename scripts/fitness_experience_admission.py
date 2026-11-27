"""Fitness-gated EA replay admission with unchanged GA and uniform TD3 sampling."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from evorl.algorithms.erl.erl_td3.d0_placebo import ObservedD0
from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
    make_ea_rollout_provenance,
)
from evorl.utils.jax_utils import right_shift_with_padding, tree_stop_gradient
from evorl.utils.rl_toolkits import (
    flatten_pop_rollout_episode,
    flatten_rollout_trajectory,
)


RANDOM_ADMISSION_TAG = 0x4641444D
ADMISSION_MODES = ("fitness_top", "random")


def select_admitted_individuals(
    fitnesses,
    *,
    mode,
    admission_count,
    seed,
    generation,
):
    """Select population IDs without touching the training PRNG stream."""
    if mode not in ADMISSION_MODES:
        raise ValueError(f"unsupported admission mode: {mode}")
    population_size = fitnesses.shape[0]
    if not 0 < admission_count <= population_size:
        raise ValueError("admission_count must be within the population")
    if mode == "fitness_top":
        selected = jnp.argsort(-fitnesses, stable=True)[:admission_count]
    else:
        root = jax.random.fold_in(
            jax.random.PRNGKey(seed), jnp.uint32(RANDOM_ADMISSION_TAG)
        )
        selected = jax.random.permutation(
            jax.random.fold_in(root, generation), population_size
        )[:admission_count]
    return jnp.sort(selected)


def transition_admission_mask(
    selected_ids,
    *,
    time_size,
    population_size,
    episode_batch,
):
    """Map population admission to flattened [T, population * episodes]."""
    admitted = jnp.zeros((population_size,), dtype=jnp.bool_)
    admitted = admitted.at[selected_ids].set(True)
    return jnp.tile(jnp.repeat(admitted, episode_batch), time_size)


class FitnessExperienceAdmissionWorkflow(ObservedD0):
    """Evaluate every EA individual and admit only selected trajectories."""

    @classmethod
    def _build_from_config(cls, config):
        if str(config.arm) != "uniform" or float(config.alpha) != 0:
            raise ValueError("experience admission requires original Uniform Replay")
        if bool(config.diagnostic_enabled):
            raise ValueError("experience admission does not use diagnostics")
        if str(config.admission_mode) not in ADMISSION_MODES:
            raise ValueError("invalid admission_mode")
        if int(config.admission_count) != 16 or int(config.pop_size) != 32:
            raise ValueError("pilot requires 16 admitted individuals from population 32")
        return super()._build_from_config(config)

    @classmethod
    def enable_jit(cls):
        super().enable_jit()
        cls._admission_ec_rollout_core = jax.jit(
            cls._admission_ec_rollout_core, static_argnums=(0,)
        )

    def _admission_ec_rollout_core(
        self, agent_state, replay_buffer_state, key, generation
    ):
        metrics, raw_trajectory = jax.vmap(
            self.ec_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, self.config.pop_size),
            self.config.episodes_for_fitness,
        )
        fitnesses = metrics.episode_returns.mean(axis=-1)
        selected_ids = select_admitted_individuals(
            fitnesses,
            mode=str(self.config.admission_mode),
            admission_count=int(self.config.admission_count),
            seed=int(self.config.seed),
            generation=generation,
        )

        trajectory = raw_trajectory.replace(next_obs=None)
        trajectory = flatten_pop_rollout_episode(trajectory)
        valid_mask = jnp.logical_not(
            right_shift_with_padding(trajectory.dones, 1)
        )
        trajectory = trajectory.replace(dones=None)
        trajectory, valid_mask = tree_stop_gradient(
            flatten_rollout_trajectory((trajectory, valid_mask))
        )

        flat_size = jtu.tree_leaves(trajectory)[0].shape[0]
        episode_batch = int(self.config.episodes_for_fitness)
        time_size = flat_size // (int(self.config.pop_size) * episode_batch)
        admitted_mask = transition_admission_mask(
            selected_ids,
            time_size=time_size,
            population_size=int(self.config.pop_size),
            episode_batch=episode_batch,
        )
        write_mask = valid_mask & admitted_mask
        provenance = make_ea_rollout_provenance(
            generation=generation,
            population_size=int(self.config.pop_size),
            time_size=time_size,
            episode_batch=episode_batch,
        )
        next_replay = self.replay_buffer.add(
            replay_buffer_state, trajectory, provenance, write_mask
        )
        return (
            metrics,
            trajectory,
            next_replay,
            fitnesses,
            selected_ids,
            valid_mask,
            write_mask,
        )

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        if getattr(self, "_budget_audit", None) is not None:
            self._observed_ec_rollouts += 1
        before_index = replay_buffer_state.current_index
        before_size = replay_buffer_state.buffer_size
        result = self._admission_ec_rollout_core(
            agent_state,
            replay_buffer_state,
            key,
            self._provenance_generation,
        )
        metrics, trajectory, after, fitnesses, selected, valid, written = result
        self.latest_admission = dict(
            generation=self._provenance_generation,
            fitnesses=fitnesses,
            selected_ids=selected,
            all_valid_transitions=valid.sum(),
            written_ea_transitions=written.sum(),
            before_index=before_index,
            after_index=after.current_index,
            before_size=before_size,
            after_size=after.buffer_size,
        )
        return metrics, trajectory, after

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        before_index = replay_buffer_state.current_index
        metrics, trajectory, after = super()._rl_rollout(
            agent_state, replay_buffer_state, key
        )
        self.latest_rl_write = dict(
            before_index=before_index,
            after_index=after.current_index,
            written_rl_transitions=metrics.episode_lengths.sum(),
        )
        return metrics, trajectory, after
