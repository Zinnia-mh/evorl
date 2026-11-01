"""Dedicated calibration trajectory and offline analysis; never a formal arm launch."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import yaml
from omegaconf import OmegaConf
try:
    from scripts.calibrate_utility_replay import registration, digest, freeze, FIELDS
except ModuleNotFoundError:
    from calibrate_utility_replay import registration, digest, freeze, FIELDS
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_capture import seed_preflight
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_pipeline import (
    CalibrationObserver, PROTOCOL, simulate, aggregate)
from evorl.algorithms.erl.erl_td3.utility_replay_budget import validate_arm_config, prepare_arm_manifest

ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration')
REPO=Path(__file__).resolve().parents[1]


def git_head():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip()


def require_clean_tree():
    if subprocess.check_output(['git','status','--porcelain'],text=True,cwd=REPO).strip():
        raise ValueError('collection preregistration/run requires committed clean code')


def hashes(directory, filename):
    directory=Path(directory)
    paths=sorted(p for p in directory.rglob('*') if p.is_file() and p.name!=filename)
    (directory/filename).write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(directory).as_posix()}\n' for p in paths))


def verify(directory, filename):
    for line in (directory/filename).read_text().splitlines():
        expected,relative=line.split('  ',1)
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('corpus or preregistration checksum mismatch')


def prepare(config_path, registered_candidates, formal_seeds, directory):
    require_clean_tree()
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT): raise ValueError('calibration artifacts must be outside Git under registered root')
    seeds=seed_preflight(formal_seeds)
    reg=json.loads(Path(registered_candidates).read_text())
    if reg!=registration(): raise ValueError('registered candidates changed')
    config=OmegaConf.load(config_path)
    validate_arm_config(config)
    if config.arm!='uniform' or config.seed!=104729 or config.warmup_iters!=10 or config.diagnostic_interval!=10:
        raise ValueError('dedicated uniform seed 104729, warmup 10 and interval 10 required')
    if config.pop_size!=32 or config.episodes_for_fitness!=1 or config.rollout_episodes!=1 or config.num_rl_agents!=1:
        raise ValueError('registered collection interaction budget mismatch')
    if not config.enable_jit or config.total_episodes!=2300:
        raise ValueError('registered JIT and episode budget required')
    if config.env.env_name!='swimmer' or config.env.env_type!='brax' or config.env.max_episode_steps!=1000 or config.source_batch_size!=256:
        raise ValueError('registered environment, horizon and source batch required')
    directory.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(config,directory/'collection_config.yaml',resolve=True)
    prereg=dict(seeds,protocol=PROTOCOL,protocol_sha256=digest(PROTOCOL),commit=git_head(),
                registration_sha256=reg['sha256'],artifact_root=str(directory))
    (directory/'preregistration.json').write_text(json.dumps(prereg,sort_keys=True,indent=2))
    (directory/'candidates.json').write_text(json.dumps(reg,sort_keys=True,indent=2))
    hashes(directory,'preregistration.sha256')
    return prereg


def run(directory):
    import jax
    import sys
    import jaxlib
    from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
    require_clean_tree()
    directory=Path(directory).resolve(); verify(directory,'preregistration.sha256')
    prereg=json.loads((directory/'preregistration.json').read_text())
    if not directory.is_relative_to(ROOT) or prereg['artifact_root']!=str(directory):
        raise ValueError('collection directory differs from registered external artifact root')
    if prereg['commit']!=git_head() or prereg['protocol']!=PROTOCOL:
        raise ValueError('collection code or protocol differs from preregistration')
    seed_preflight(prereg['formal_seeds'],prereg['calibration_seed'])
    if (directory/'snapshots').exists(): raise FileExistsError('collection already started; no implicit resume')
    config=OmegaConf.load(directory/'collection_config.yaml')
    (directory/'runtime.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,
        jax=jax.__version__,jaxlib=jaxlib.__version__,devices=[str(d) for d in jax.devices()]),sort_keys=True))
    workflow=type('CalibrationOnlyWorkflow',(ERLUtilityReplayWorkflow,),{}).build_from_config(config,enable_jit=True)
    try:
        state=workflow.setup(jax.random.PRNGKey(104729))
        observer=CalibrationObserver(directory/'snapshots'); workflow.attach_calibration_observer(observer)
        updates=0
        for _ in range(PROTOCOL['generations']):
            metrics,state=workflow.step(state)
            updates+=int(metrics.num_updates_per_iter)
        counters=dict(generation=int(state.metrics.iterations),episodes=int(state.metrics.sampled_episodes),
                      timesteps=int(state.metrics.sampled_timesteps),outer_rl_updates=updates)
        if counters['episodes']!=2300 or counters['generation']!=70: raise ValueError('collection budget mismatch')
        (directory/'collection_counters.json').write_text(json.dumps(counters,sort_keys=True))
        hashes(directory/'snapshots','snapshots.sha256')
    finally: workflow.close()


def analyze(directory, output, formal_base_config):
    import numpy as np
    require_clean_tree()
    directory=Path(directory).resolve(); verify(directory,'preregistration.sha256')
    verify(directory/'snapshots','snapshots.sha256')
    prereg=json.loads((directory/'preregistration.json').read_text())
    if not directory.is_relative_to(ROOT) or prereg['artifact_root']!=str(directory):
        raise ValueError('analysis directory differs from registered external artifact root')
    if prereg['commit']!=git_head(): raise ValueError('analysis code differs from registered collection code')
    counters=json.loads((directory/'collection_counters.json').read_text())
    if counters['episodes']!=2300 or counters['generation']!=70: raise ValueError('incomplete calibration run')
    config=OmegaConf.load(formal_base_config)
    if config.seed not in prereg['formal_seeds']:
        raise ValueError('formal base config must use a registered formal seed')
    reg=json.loads((directory/'candidates.json').read_text())
    if reg!=registration() or prereg['protocol']!=PROTOCOL: raise ValueError('registration mismatch')
    paths=sorted((directory/'snapshots').glob('generation_*.json'))
    expected=list(range(11,71))
    if [int(p.stem.split('_')[-1]) for p in paths]!=expected: raise ValueError('missing per-generation snapshot')
    if any(not p.with_suffix('.npz').is_file() for p in paths): raise ValueError('missing replay sidecar archive')
    metrics_dir=directory/'offline_metrics'
    corpus_hash=hashlib.sha256((directory/'snapshots/snapshots.sha256').read_bytes()).hexdigest()
    metrics_identity=dict(corpus_sha256=corpus_hash,protocol_sha256=prereg['protocol_sha256'],commit=prereg['commit'])
    if metrics_dir.exists():
        verify(metrics_dir,'checksums.sha256')
        if json.loads((metrics_dir/'identity.json').read_text())!=metrics_identity:
            raise ValueError('frozen measured metrics belong to another corpus/protocol')
        rows=json.loads((metrics_dir/'aggregated.json').read_text())
        details=json.loads((metrics_dir/'measurements.json').read_text())
    else:
        generations=[]; details=[]
        for path in paths:
            meta=json.loads(path.read_text())
            if meta['generation']!=int(path.stem.split('_')[-1]): raise ValueError('snapshot generation/path mismatch')
            if meta['generation']<PROTOCOL['first_analysis_generation']: continue
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            rows,measurements=simulate(meta,arrays,reg,draws=PROTOCOL['draws'])
            generations.append(rows); details.extend(measurements)
        rows=aggregate(generations,reg)
        metrics_dir.mkdir(exist_ok=False)
        for name,value in [('identity',metrics_identity),('aggregated',rows),('measurements',details)]:
            (metrics_dir/f'{name}.json').write_text(json.dumps(value,sort_keys=True,allow_nan=False))
        hashes(metrics_dir,'checksums.sha256')
    provenance=dict(source='option-b',seed=104729,formal_seeds=prereg['formal_seeds'],commit=prereg['commit'],
                    protocol_sha256=prereg['protocol_sha256'],artifact_root=str(directory),
                    input_sha256=hashlib.sha256((metrics_dir/'aggregated.json').read_bytes()).hexdigest())
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('selection output must be outside Git')
    result=freeze(reg,rows,output,provenance)
    (output/'simulation_measurements.json').write_text(json.dumps(details,sort_keys=True,allow_nan=False))
    write_launch_configs(config,result,output)
    hashes(output,'checksums.sha256')
    return result


def write_launch_configs(config,result,output):
    output=Path(output); configs=[]
    for label,overrides in result['arms'].items():
        arm=OmegaConf.merge(config,dict(overrides,diagnostic_interval=PROTOCOL['diagnostic_interval']))
        arm.arm_manifest_path=str(output/'arm_manifest.json')
        configs.append(arm); OmegaConf.save(arm,output/f'arm_{label}.yaml',resolve=True)
    prepare_arm_manifest(configs,output/'arm_manifest.json')
    result['launch_configs']={label:f'arm_{label}.yaml' for label in 'ABCD'}
    result['launch_status']='matched full arm configs; no experiment launched'
    result['fixed_protocol']=PROTOCOL
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(result,sort_keys=True))


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--config',required=True); p.add_argument('--candidates',required=True)
    p.add_argument('--formal-seeds',nargs=3,type=int,required=True); p.add_argument('--directory',required=True)
    for command in ['run','analyze']:
        p=sub.add_parser(command); p.add_argument('--directory',required=True)
        if command=='analyze':
            p.add_argument('--output',required=True); p.add_argument('--formal-base-config',required=True)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.config,args.candidates,args.formal_seeds,args.directory)
    elif args.command=='run': run(args.directory)
    else: analyze(args.directory,args.output,args.formal_base_config)


if __name__=='__main__': main()
