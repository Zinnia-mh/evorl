"""Experiment-only placebo for the frozen D0 replay distribution."""
from collections import defaultdict
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from .erl_utility_replay import ERLUtilityReplayWorkflow, _optimizer_step_count
from .utility_replay_smoke_workflow import reconstruct_replay_draws


def permuted_ranks(table, *, seed, evidence=None):
    """Shuffle rank labels within exact age and residency strata.

    The canonical table stays true to the diagnostic. The cohort/size key is
    stable; ring eviction can change stratum membership, requiring remapping.
    """
    eligible, tokens, measured, counts, rank = map(np.asarray, jax.device_get((
        table.weighted_eligible, table.source_token, table.measured_generation,
        table.resident_count, table.rank_normalized_score,
    )))
    slots = np.flatnonzero(eligible)
    groups = defaultdict(list)
    for i in slots[np.argsort(tokens[slots])]:
        groups[int(measured[i]), int(counts[i])].append(i)
    changed = rank.copy()
    donor = np.arange(len(rank), dtype=np.int32)
    for (generation, count), members in groups.items():
        if len(members) > 1:
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([0x50304430, seed, generation, count])
            ))
            donor[members] = rng.permutation(members)
            changed[members] = rank[donor[members]]
    if evidence is not None:
        evidence.update(
            movable_sources=sum(len(g) for g in groups.values() if len(g) > 1),
            changed_ranks=int(np.count_nonzero(changed[slots] != rank[slots])),
            eligible_mapping=[
                [int(tokens[i]), int(tokens[donor[i]]), int(measured[i]),
                 int(counts[i]), float(rank[i]), float(changed[i])]
                for i in slots
            ],
        )
    return table._replace(rank_normalized_score=jnp.asarray(changed))


class ObservedD0(ERLUtilityReplayWorkflow):
    """Read-only observation of actual learner inputs at the host boundary."""

    def _weighted_rl_update(self, agent, optimizer, replay, key, num_updates, utility):
        self.learner_input = (replay, key, int(num_updates), utility)
        return super()._weighted_rl_update(agent, optimizer, replay, key, num_updates, utility)

    def _rl_update(self, agent, optimizer, replay, key, num_updates):
        self.learner_input = (replay, key, int(num_updates), self.prior_utility)
        return super()._rl_update(agent, optimizer, replay, key, num_updates)

    def step(self, state):
        self.prior_utility = state.utility_replay_state
        self.learner_input = None
        self.permutation = {}
        self.permutation_seconds = 0.0
        started = time.perf_counter()
        metrics, after = super().step(state)
        jax.block_until_ready((metrics, after))
        self.step_seconds = time.perf_counter() - started
        self.baseline_key_ok = bool(np.array_equal(after.key, jax.random.split(state.key, 4)[0]))
        self.prior_utility = None
        return metrics, after

    def evaluate(self, state):
        started = time.perf_counter()
        result = super().evaluate(state)
        jax.block_until_ready(result)
        self.eval_seconds = time.perf_counter() - started
        return result


class Ablation:
    workflow_type = ObservedD0

    def __init__(self, arm, seed, plan_sha256):
        if arm not in ('A', 'D0', 'P'):
            raise ValueError('invalid ablation arm')
        self.identity = dict(arm=arm, seed=seed, plan_sha256=plan_sha256)
        self.arm = arm
        self.seed = seed

    def configure(self, workflow, segment):
        self.segment = segment
        if int(workflow.config.seed) != self.seed or (self.arm == 'A') != (float(workflow.config.alpha) == 0):
            raise ValueError('ablation does not match resolved configuration')
        if self.arm == 'P':
            def transform(table):
                started = time.perf_counter()
                out = permuted_ranks(table, seed=self.seed, evidence=workflow.permutation)
                jax.block_until_ready(out.rank_normalized_score)
                workflow.permutation_seconds = time.perf_counter() - started
                return out
            workflow._ablation_rank_transform = transform

    def before_generation(self, state):
        generation = int(state.metrics.iterations)
        if not 10 <= generation < 120 or int(state.metrics.sampled_timesteps) != (320 + (generation - 10) * 33) * 1000:
            raise ValueError('generation or training budget mismatch')

    def record(self, workflow, state, checkpoint, runtime):
        g = int(state.metrics.iterations)
        expected_steps = (320 + (g - 10) * 33) * 1000
        if int(state.metrics.sampled_timesteps) != expected_steps or not workflow.baseline_key_ok:
            raise ValueError('training budget or baseline PRNG mismatch')
        replay, key, updates, utility = workflow.learner_input
        workflow.learner_input = None
        if updates != 16000:
            raise ValueError('learner update count mismatch')
        evaluation = runtime['evaluations'][-1] if runtime['evaluations'] else None
        evaluated = evaluation is not None and evaluation['generation'] == g
        if evaluated != (g % 10 == 0) or (evaluated and not np.all(np.asarray(evaluation['mean_episode_lengths']) == 1000)):
            raise ValueError('evaluation schedule or step mismatch')
        u = state.utility_replay_state
        lifecycle = getattr(workflow, '_latest_score_lifecycle_audit', {})
        version = int(u.score_table.current_version)
        if self.arm != 'A' and version != g // 10 - 2:
            raise ValueError('D0 diagnostic score cadence mismatch')
        row = dict(
            generation=g, training_environment_steps=expected_steps,
            actor_optimizer_count=_optimizer_step_count(state.opt_state.actor),
            critic_optimizer_count=_optimizer_step_count(state.opt_state.critic),
            checkpoint=str(checkpoint), score_version=version,
            training_seconds=workflow.step_seconds,
            evaluation_seconds=workflow.eval_seconds if evaluated else 0.0,
            diagnostic_seconds=lifecycle.get('diagnostic_seconds', 0.0),
            distribution_seconds=lifecycle.get('distribution_refresh_seconds', 0.0),
            permutation_seconds=workflow.permutation_seconds,
            evaluation=evaluation if evaluated else None, tvd=0.0, ess_fraction=1.0,
            eligible_mass=0.0, eligible_sources=0, eligible_ranks=workflow.permutation,
        )
        if self.arm != 'A':
            dist = utility.distribution
            p, base, t, cdf = map(np.asarray, jax.device_get((
                dist.source_probability, dist.baseline_source_probability,
                dist.transition_probability, dist.transition_cdf,
            )))
            if bool(utility.distribution_prepared):
                from .replay_distribution_audit import audit_distribution
                counts, origin = map(np.asarray, jax.device_get((utility.score_table.resident_count, replay.origin_type)))
                ea = (np.arange(origin.size) < int(replay.buffer_size)) & (origin == 0)
                audit_distribution(p, counts, t, cdf, ea, float(dist.ess_fraction), cdf_in_training_sampling=True)
            p64 = p.astype(np.float64)
            n = int(dist.resident_source_count)
            row.update(tvd=float(np.abs(p64-base).sum()/2),
                       ess_fraction=float(p64.sum()**2 / (np.dot(p64, p64)*n)) if n else 0.,
                       eligible_mass=float(dist.eligible_baseline_mass),
                       eligible_sources=int(dist.eligible_source_count))
        if g in (20, 31, 60, 120):
            trace = jax.device_get(reconstruct_replay_draws(
                replay, key, utility, num_updates=32, num_draws=2,
                batch_size=256, uniform=self.arm == 'A',
            ))
            changed = trace.baseline_indices != trace.final_indices
            if np.any(changed & trace.rl_slot_mask) or not np.all(trace.origin_ok):
                raise ValueError('learner sampling altered RL slots or source type')
            if self.arm == 'A' and np.any(changed):
                raise ValueError('Uniform draw changed')
            row.update(index_evidence=dict(
                scope='first 32 actual learner rounds, deterministically reconstructed',
                changed_ea=int(np.sum(changed & trace.ea_slot_mask)), changed_rl=0,
                baseline=trace.baseline_indices[0, 0, :16].tolist(),
                final=trace.final_indices[0, 0, :16].tolist(),
                ea=trace.ea_slot_mask[0, 0, :16].tolist(),
            ))
        with (self.segment / f'ablation-generation-{g}.json').open('x') as file:
            json.dump(row, file, allow_nan=False)

