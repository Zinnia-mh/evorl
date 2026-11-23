"""Shared CP19 fork and actual learner-input evidence for the D0/P experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
from omegaconf import OmegaConf

from .d0_placebo import Ablation, ObservedD0
from .source_replay import EA_ORIGIN, RL_ORIGIN
from .utility_replay_training import _counters
from .utility_replay_checkpoint import PilotCheckpointStore, mechanism_digest, restore_template


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_hashes(state):
    groups = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        name = jax.tree_util.keystr(path)
        value = np.asarray(jax.device_get(leaf))
        row = dict(path=name, shape=list(value.shape), dtype=str(value.dtype),
                   sha256=hashlib.sha256(value.tobytes()).hexdigest())
        # State stores its sorted field names in the PyTree auxiliary data.
        group = sorted(state)[path[0].key]
        groups.setdefault(group, []).append(row)
    return dict(tree=str(jax.tree.structure(state)), components={
        group: dict(leaves=rows, sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        for group, rows in groups.items()
    })


def _metadata(checkpoint, config_path):
    files = {str(p.relative_to(checkpoint)): sha(p)
             for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    seal_path = checkpoint.parent.parent / 'checkpoint_checksums/19.json'
    if checkpoint.name != '19' or not files or files != json.loads(seal_path.read_text()):
        raise ValueError('CP19 file seal mismatch')
    return dict(files=files, config_sha256=sha(config_path),
                source_sha256=mechanism_digest())


def _parent_config_path(checkpoint):
    # Canonical layout uses the run's resolved config, above the segment tree.
    return next(p / 'resolved_config.yaml' for p in checkpoint.parents
                if (p / 'resolved_config.yaml').is_file())


def seal_parent(checkpoint, config, state, runtime):
    checkpoint = Path(checkpoint).resolve()
    if int(state.metrics.iterations) != 19 or runtime['evaluations'] or runtime['budget']['eval_episodes']:
        raise ValueError('parent must be CP19 before any evaluation')
    config_path = _parent_config_path(checkpoint)
    meta = _metadata(checkpoint, config_path)
    native = json.loads((checkpoint/'runtime/metadata').read_text())
    from .utility_replay_smoke_workflow import initialization_fingerprint
    if native['runtime'] != runtime or native['state_sha256'] != initialization_fingerprint(state):
        raise ValueError('sealed payload differs from supplied parent')
    counters = _counters(state)
    meta.update(seed=int(config.seed), commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], text=True).strip(), parent_run_id=runtime['run_id'],
        created_utc=datetime.now(timezone.utc).isoformat(), state=component_hashes(state),
        counters=dict(counters, evaluation_episodes=0, evaluations=0))
    with (checkpoint.parent.parent/'shared-cp19.json').open('x') as stream:
        json.dump(meta, stream, sort_keys=True)
    return meta


def restore_parent(checkpoint, config, template, run_id):
    checkpoint = Path(checkpoint).resolve()
    config_path = _parent_config_path(checkpoint)
    a,b = (OmegaConf.to_container(c,resolve=True) for c in (OmegaConf.load(config_path), config))
    for c in (a,b):
        c.pop('output_dir',None)
    if a != b:
        raise ValueError('fork configuration differs from the common parent')
    seal = json.loads((checkpoint.parent.parent/'shared-cp19.json').read_text())
    if any(seal[k] != v for k,v in _metadata(checkpoint,config_path).items()):
        raise ValueError('parent seal mismatch')
    with PilotCheckpointStore(checkpoint.parent, config, read_only=True) as store:
        state,runtime=store.restore(19,template)
    if (seal['state'] != component_hashes(state) or runtime['run_id'] != seal['parent_run_id']
            or int(config.seed) != seal['seed'] or runtime['evaluations']
            or runtime['budget']['eval_episodes']
            or seal['counters'] != dict(_counters(state), evaluation_episodes=0, evaluations=0)):
        raise ValueError('parent semantic state or runtime mismatch')
    return state,dict(runtime,run_id=run_id)


def save_draw_audit(path, trace, replay, size, **metadata):
    baseline, final, learner_keys, update_keys, intervention_keys = map(np.asarray,jax.device_get(trace))
    origin, tokens = map(np.asarray,jax.device_get((replay.origin_type,replay.source_token)))
    if (baseline.shape != final.shape or np.any(baseline < 0) or np.any(final < 0)
            or np.any(baseline >= min(len(origin),size,int(replay.buffer_size)))
            or np.any(final >= min(len(origin),size,int(replay.buffer_size)))):
        raise ValueError('learner replay indices out of range')
    ea = origin[baseline] == EA_ORIGIN
    rl = origin[baseline] == RL_ORIGIN
    if not np.all(ea | rl) or np.any(origin[final] != origin[baseline]) or np.any(final[rl]!=baseline[rl]):
        raise ValueError('learner sampling changed source type or RL index')
    ids,counts=np.unique(tokens[final[ea]],return_counts=True)
    if np.any(ids < 0):
        raise ValueError('missing EA source identity')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, baseline_indices=baseline, final_indices=final,
                           learner_keys=learner_keys, critic_update_keys=update_keys,
                           intervention_keys=intervention_keys, ea_per_batch=ea.sum(-1),
                           rl_per_batch=rl.sum(-1), ea_source_tokens=ids, ea_source_counts=counts,
                           changed_ea_per_batch=((baseline!=final)&ea).sum(-1),
                           replay_origin=origin, replay_source_token=tokens, **metadata)


class AuditedD0(ObservedD0):
    def step(self, state):
        self.audit_initial_key = state.key
        self.audit_initial_ec_key = state.ec_opt_state.key
        return super().step(state)

    def _capture_learner_trace(self, trace):
        self.learner_trace = trace

    def _learner_trace(self, sampled, learner_key, update_key, intervention_key):
        return (sampled.baseline_indices, sampled.final_indices, learner_key,
                update_key, intervention_key)


class SharedStartAblation(Ablation):
    workflow_type = AuditedD0
    reconstruct_sample_evidence = False
    audit_before_checkpoint = True

    def __init__(self, arm, seed, parent):
        if arm not in ('D0','P'):
            raise ValueError('shared start requires D0 or P')
        self.parent = Path(parent).resolve()
        super().__init__(arm,seed,sha(self.parent.parent.parent/'shared-cp19.json'))
        self.identity['parent_checkpoint'] = str(self.parent)

    def fork_state(self, workflow, run_id):
        return restore_parent(self.parent, workflow.config, restore_template(workflow), run_id)

    def validate_resume(self, checkpoint,state):
        generation=int(state.metrics.iterations)
        segment=checkpoint.parent.parent
        latest = max(int(seal.stem) for seal in
                     segment.parent.glob('*/checkpoint_checksums/*.json')
                     if (seal.parent.parent/'checkpoints'/seal.stem).is_dir())
        if latest != generation:
            raise ValueError('resume requires the latest committed generation')
        if not (segment/f'critic-draws-{generation}.npz').is_file():
            raise ValueError('checkpoint lacks actual learner-input audit')
        receipt=json.loads((segment/f'critic-draws-{generation}.json').read_text())
        if receipt['artifact_sha256'] != sha(segment/f'critic-draws-{generation}.npz'):
            raise ValueError('learner-input audit checksum mismatch')
        if receipt['state'] != component_hashes(state):
            raise ValueError('learner-input audit does not belong to the checkpoint state')

    def record(self, workflow, state, checkpoint, runtime):
        replay, key, updates, utility = workflow.learner_input
        generation = int(state.metrics.iterations)
        if len(workflow.learner_trace[0]) != updates:
            raise ValueError('missing learner rounds in trace')
        dist,table=utility.distribution,utility.score_table
        mapping=workflow.permutation.get('eligible_mapping',[])
        if (np.asarray(workflow.learner_trace[0]).shape !=
                (updates, self._draws(workflow), int(workflow.config.batch_size))):
            raise ValueError('unexpected optimizer batch shape')
        path=self.segment/f'critic-draws-{generation}.npz'
        extra={}
        records=getattr(workflow,'_latest_diagnostic_records',())
        if generation % int(workflow.config.diagnostic_interval)==0:
            extra=dict(diagnostic_individuals=np.asarray([r.individual_id for r in records]),
                       diagnostic_fitness=np.asarray([r.fitness for r in records]),
                       diagnostic_utility=np.asarray([np.nan if r.utility is None else r.utility for r in records]))
        save_draw_audit(path,
                        workflow.learner_trace,replay,len(replay.origin_type),
                        source_token=table.source_token, resident_count=table.resident_count,
                        eligible=table.weighted_eligible, measured_generation=table.measured_generation,
                        score_version=table.version, raw_score=table.raw_score,
                        canonical_rank=table.rank_normalized_score,
                        permutation=np.asarray(mapping).reshape(-1,6),
                        baseline_probability=dist.baseline_source_probability,
                        source_probability=dist.source_probability,
                        weighted_probability=dist.weighted_source_probability,
                        transition_probability=dist.transition_probability,
                        transition_cdf=dist.transition_cdf,
                        probability_l1=np.abs(np.asarray(dist.source_probability)-np.asarray(dist.baseline_source_probability)).sum(),
                        ess_fraction=dist.ess_fraction, eligible_mass=dist.eligible_baseline_mass,
                        initial_learner_key=key, final_intervention_key=state.utility_replay_state.intervention_key,
                        initial_training_key=workflow.audit_initial_key,
                        final_training_key=state.key,
                        initial_ec_key=workflow.audit_initial_ec_key,
                        final_ec_key=state.ec_opt_state.key,
                        replay_source_generation=replay.source_generation,
                        replay_score_version=replay.score_version,
                        baseline_key_splits=1+int(generation % int(workflow.config.eval_interval)==0),
                        learner_rounds=updates,
                        probability_tvd=.5*np.abs(np.asarray(dist.source_probability)-np.asarray(dist.baseline_source_probability)).sum(),
                        intervention_advances=np.count_nonzero(
                            np.any(np.asarray(replay.origin_type)[np.asarray(workflow.learner_trace[0])]==EA_ORIGIN,axis=(1,2))
                        ) if bool(utility.distribution_prepared) else 0,
                        **extra)
        workflow.learner_trace=None
        super().record(workflow,state,checkpoint,runtime)
        with (self.segment/f'critic-draws-{generation}.json').open('x') as stream:
            json.dump(dict(artifact_sha256=sha(path),state=component_hashes(state)),stream)

    @staticmethod
    def _draws(workflow):
        return int(workflow.config.actor_update_interval) * int(workflow.config.num_rl_agents)

