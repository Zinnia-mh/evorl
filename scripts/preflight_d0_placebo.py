"""Zero-rollout compatibility and placebo audit against frozen seed-1 CP50."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.d0_placebo import Ablation, ObservedD0
from evorl.algorithms.erl.erl_td3.erl_utility_replay import (
    ERLUtilityReplayWorkflow, alpha_zero_sample, refresh_replay_distribution,
    weighted_replay_sample,
)
from evorl.algorithms.erl.erl_td3.replay_distribution_audit import audit_distribution
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import (
    PilotCheckpointStore, restore_template,
)
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint

BASE = '228a3014902948251998bc8a39cad128c910ec6e'
CHECKPOINT_MODULE = 'evorl.algorithms.erl.erl_td3.utility_replay_checkpoint'


def historical_digest():
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, 'evorl'], text=True).splitlines()
    paths = sorted(p for p in paths if p.endswith('.py')) + ['scripts/run_utility_replay_training.py']
    digest = hashlib.sha256()
    requests = ''.join(f'{BASE}:{p}\n' for p in paths)
    raw = subprocess.check_output(['git', 'cat-file', '--batch'], input=requests.encode())
    offset = 0
    for path in paths:
        header, _, _ = raw[offset:].partition(b'\n')
        size = int(header.split()[-1])
        offset += len(header) + 1
        digest.update(path.encode())
        digest.update(raw[offset:offset + size])
        offset += size + 1
    assert offset == len(raw)
    return digest.hexdigest()


def workflow(cls, config):
    return type('PreflightWorkflow', (cls,), {}).build_from_config(config, enable_jit=True)


def restore(config, step, wf, digest=None):
    # Only the historical source digest is substituted. Native file, config,
    # environment and full-state seals remain active, and the store is read-only.
    context = patch(f'{CHECKPOINT_MODULE}.mechanism_digest', return_value=digest) if digest else nullcontext()
    with context, PilotCheckpointStore(step.parent, config, read_only=True) as store:
        return store.restore(int(step.name), restore_template(wf))


def same(a, b):
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('formal', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    digest = historical_digest()
    runs = args.formal / 'runs'
    output = args.output
    config_a = OmegaConf.load(runs / 'A/seed_1/resolved_config.yaml')
    config_d = OmegaConf.load(runs / 'D0/seed_1/resolved_config.yaml')
    wf_a = workflow(ERLUtilityReplayWorkflow, config_a)
    wf_d = workflow(ObservedD0, config_d)
    wf_p = workflow(ObservedD0, config_d)
    placebo = Ablation('P', 1, 'preflight-only')
    placebo.configure(wf_p, output)
    wf_p.permutation = {}
    key, intervention = jax.random.split(jax.random.PRNGKey(2081))
    cp_a, = (runs / 'A/seed_1').glob('segments/*/checkpoints/50')
    cp_d, = (runs / 'D0/seed_1').glob('segments/*/checkpoints/50')
    assert json.loads((runs / 'A/seed_1/manifest.json').read_text())['commit'] == BASE
    assert json.loads((runs / 'D0/seed_1/manifest.json').read_text())['commit'] == BASE

    a, _ = restore(config_a, cp_a, wf_a, digest)
    expected = wf_a.replay_buffer.sample(a.replay_buffer_state, key)
    alpha_zero = alpha_zero_sample(a.replay_buffer_state, key, intervention, 256)
    same(expected, alpha_zero.batch)
    np.testing.assert_array_equal(alpha_zero.baseline_indices, alpha_zero.final_indices)
    del a, expected, alpha_zero
    state, runtime = restore(config_d, cp_d, wf_d, digest)
    replay = state.replay_buffer_state
    utility = state.utility_replay_state
    frozen_hash = initialization_fingerprint(state)
    original = refresh_replay_distribution(utility, replay, current_generation=50,
                                           freshness_window=30, min_resident=4, alpha=.5,
                                           replay_temperature=.5)
    shuffled = refresh_replay_distribution(utility, replay, current_generation=50,
                                           freshness_window=30, min_resident=4, alpha=.5,
                                           replay_temperature=.5, rank_transform=wf_p._ablation_rank_transform)
    same(utility.distribution, original.distribution)
    same(original.score_table, shuffled.score_table)

    # Both workflows use the frozen scorer on exactly the same resident source
    # and reference, with the same initial critic/Adam state and keys.
    source_slots = np.flatnonzero(np.asarray(replay.source_token) == int(wf_p.permutation['eligible_mapping'][0][0]))[:256]
    assert len(source_slots) == 256
    from evorl.utils.jax_utils import tree_get
    source = tree_get(replay.data, source_slots)
    reference, fingerprint = wf_d._sample_current_reference(replay, jax.random.fold_in(key, 7))
    context = wf_d._make_pre_exposure_context(state)
    wf_d._diagnostic_context = wf_p._diagnostic_context = context
    source_key = jax.random.fold_in(key, 8)
    score_d = wf_d._evaluate_source_utility(source, reference, source_key)
    score_p = wf_p._evaluate_source_utility(source, reference, source_key)
    same(score_d, score_p)

    draws = [weighted_replay_sample(replay, key, intervention, u.distribution,
                                    u.distribution_prepared, 256) for u in (original, shuffled)]
    d, p = draws
    same((d.baseline_indices, d.ea_slot_mask, d.rl_slot_mask, d.next_intervention_key),
         (p.baseline_indices, p.ea_slot_mask, p.rl_slot_mask, p.next_intervention_key))
    d_indices, p_indices, rl = map(np.asarray, (d.final_indices, p.final_indices, p.rl_slot_mask))
    np.testing.assert_array_equal(d_indices[rl], p_indices[rl])
    assert np.any(d_indices != p_indices)
    same(p.batch, tree_get(replay.data, p.final_indices))
    distribution_rows = []
    for u in (original, shuffled):
        dist = u.distribution
        a, b = map(np.asarray, (dist.source_probability, dist.baseline_source_probability))
        ea = (np.arange(replay.origin_type.size) < int(replay.buffer_size)) & (np.asarray(replay.origin_type) == 0)
        audit_distribution(a, np.asarray(u.score_table.resident_count), np.asarray(dist.transition_probability),
                           np.asarray(dist.transition_cdf), ea, float(dist.ess_fraction), cdf_in_training_sampling=True)
        distribution_rows.append(dict(tvd=float(np.abs(a.astype(np.float64)-b).sum()/2),
                                      ess=float(dist.ess_fraction), mass=float(dist.eligible_baseline_mass)))
    np.testing.assert_allclose(list(distribution_rows[0].values()), list(distribution_rows[1].values()), rtol=1e-6, atol=1e-8)

    # Native candidate learner call against immutable inputs, with one actor
    # and two critic updates. This state is never written into the frozen run.
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import _optimizer_step_count
    for wf, u in ((wf_d, original), (wf_p, shuffled)):
        _, _, opt, _ = wf._weighted_rl_update(state.agent_state, state.opt_state, replay, key, 1, u)
        assert _optimizer_step_count(opt.actor) - _optimizer_step_count(state.opt_state.actor) == 1
        assert _optimizer_step_count(opt.critic) - _optimizer_step_count(state.opt_state.critic) == 2

    placebo_state = state.replace(utility_replay_state=shuffled)
    with PilotCheckpointStore(output / 'checkpoint', config_d, interval=1) as store:
        saved = store.save(placebo_state, runtime, force=True)
    restored, restored_runtime = restore(config_d, saved, wf_p)
    assert restored_runtime == runtime
    assert initialization_fingerprint(restored) == initialization_fingerprint(placebo_state)
    assert initialization_fingerprint(state) == frozen_hash
    with (output / 'preflight.json').open('x') as f:
        json.dump(dict(status='PASS', environment_steps=0, historical_digest=digest,
                       cp50_state_sha256=frozen_hash, restored_placebo_checkpoint=str(saved),
                       reference_sha256=fingerprint, rank_mapping=wf_p.permutation,
                       distribution_rows=distribution_rows,
                       first_indices=dict(d0=d_indices[:16].tolist(), placebo=p_indices[:16].tolist(),
                                          rl=rl[:16].tolist()),
                       changed_indices=int(np.count_nonzero(d_indices != p_indices))), f, indent=2)


if __name__ == '__main__':
    main()

