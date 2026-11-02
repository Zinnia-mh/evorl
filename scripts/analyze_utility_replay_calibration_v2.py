"""Versioned, resumable offline-only analysis of an immutable Option B corpus."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid
import jax
import jaxlib
import numpy as np
import yaml
from omegaconf import OmegaConf

try:
    from scripts.calibrate_utility_replay import registration,freeze
    from scripts.collect_utility_replay_calibration import verify,hashes,require_clean_tree,git_head,write_launch_configs
except ModuleNotFoundError:
    from calibrate_utility_replay import registration,freeze
    from collect_utility_replay_calibration import verify,hashes,require_clean_tree,git_head,write_launch_configs
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_pipeline import PROTOCOL,aggregate
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_offline_v2 import (
    IMPLEMENTATION,CandidateStore,atomic_create,digest,simulate_snapshot)

ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration')
REPO=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    return dict(jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),
                x64=bool(jax.config.jax_enable_x64),devices=[d.device_kind for d in jax.devices()])


def code_hashes():
    files=['scripts/analyze_utility_replay_calibration_v2.py',
           'evorl/algorithms/erl/erl_td3/utility_replay_calibration_offline_v2.py',
           'evorl/algorithms/erl/erl_td3/utility_replay_calibration_pipeline.py',
           'evorl/algorithms/erl/erl_td3/source_replay.py','scripts/calibrate_utility_replay.py']
    return {name:sha(REPO/name) for name in files}


def verify_corpus(directory,expected_sha):
    directory=Path(directory).resolve()
    verify(directory,'preflight.sha256');verify(directory,'preregistration.sha256')
    verify(directory/'snapshots','snapshots.sha256')
    if sha(directory/'snapshots/snapshots.sha256')!=expected_sha: raise ValueError('corpus identity changed')
    prereg=json.loads((directory/'preregistration.json').read_text())
    if prereg['artifact_root']!=str(directory) or prereg['protocol']!=PROTOCOL:
        raise ValueError('collection provenance/protocol mismatch')
    if prereg['calibration_seed']!=104729 or len(prereg['formal_seeds'])!=3 or len(set(prereg['formal_seeds']))!=3 or 104729 in prereg['formal_seeds']:
        raise ValueError('invalid seed separation')
    if json.loads((directory/'candidates.json').read_text())!=registration(): raise ValueError('registered candidates changed')
    if (directory/'logs/collection.exit').read_text().strip()!='0': raise ValueError('incomplete collection')
    counters=json.loads((directory/'collection_counters.json').read_text())
    if counters['episodes']!=2300 or counters['generation']!=70: raise ValueError('collection budget mismatch')
    paths=sorted((directory/'snapshots').glob('generation_*.json'))
    if [int(p.stem.split('_')[-1]) for p in paths]!=list(range(11,71)):
        raise ValueError('incomplete per-generation corpus')
    if any(not p.with_suffix('.npz').is_file() for p in paths): raise ValueError('missing replay arrays')
    return prereg,paths


def prepare(directory,formal_base_config,work_dir,output,corpus_sha256):
    require_clean_tree()
    directory=Path(directory).resolve();work_dir=Path(work_dir).resolve();output=Path(output).resolve()
    if not work_dir.is_relative_to(ROOT) or not output.is_relative_to(ROOT): raise ValueError('outputs must be outside Git')
    if work_dir.is_relative_to(directory) or output.is_relative_to(directory) or output==work_dir:
        raise ValueError('original collection must remain read-only')
    if output==ROOT/'return-blind-v1-selected-head8240863': raise ValueError('cannot impersonate original analysis output')
    if work_dir.exists() or output.exists(): raise FileExistsError('analysis version/output already exists')
    prereg,_=verify_corpus(directory,corpus_sha256)
    config=OmegaConf.load(formal_base_config)
    if config.seed not in prereg['formal_seeds']: raise ValueError('formal base seed not registered')
    current_runtime=runtime()
    if current_runtime['backend']!='gpu': raise ValueError('registered sampler benchmark requires GPU')
    analysis_commit=git_head()
    if analysis_commit==prereg['commit']: raise ValueError('v2 must have a distinct committed analysis implementation')
    identity=dict(schema=2,implementation=IMPLEMENTATION,analysis_commit=analysis_commit,
        collection_commit=prereg['commit'],code_sha256=code_hashes(),runtime=current_runtime,
        corpus=str(directory),corpus_manifest_sha256=corpus_sha256,
        collection_preregistration_sha256=sha(directory/'preregistration.json'),
        collection_files_sha256={name:sha(directory/name) for name in
            ['preflight.sha256','preregistration.sha256','collection_config.yaml','candidates.json','runtime.json']},
        registration_sha256=registration()['sha256'],policy=registration(),protocol=PROTOCOL,
        protocol_sha256=prereg['protocol_sha256'],formal_seeds=prereg['formal_seeds'],calibration_seed=104729,
        formal_base_config=OmegaConf.to_container(config,resolve=True),formal_base_sha256=sha(formal_base_config),
        output=str(output),analysis_generations=list(range(20,71)),draws=PROTOCOL['draws'],
        timing='unchanged per-case warmup then synchronized timed sampler; preparation/checkpoint IO excluded')
    record=dict(identity,sha256=digest(identity))
    work_dir.mkdir(parents=True,exist_ok=False)
    atomic_create(work_dir/'analysis_registration.json',record)
    CandidateStore(work_dir/'candidates',identity)
    return record


def load_registration(work_dir):
    record=json.loads((Path(work_dir)/'analysis_registration.json').read_text())
    expected=record.pop('sha256')
    if digest(record)!=expected or record['implementation']!=IMPLEMENTATION: raise ValueError('analysis registration corrupted')
    if record['policy']!=registration() or record['protocol']!=PROTOCOL: raise ValueError('registered policy changed')
    if record['analysis_commit']!=git_head() or record['code_sha256']!=code_hashes(): raise ValueError('analysis code changed')
    return record


def once_or_equal(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value: raise ValueError('frozen measurement differs; refusing overwrite')
    else: atomic_create(path,value)


def acquire_lock(path,run_id):
    path=Path(path)
    if path.exists():
        owner=json.loads(path.read_text())
        try: os.kill(int(owner['pid']),0)
        except ProcessLookupError:
            path.rename(path.with_name(path.name+'.stale-'+uuid.uuid4().hex))
        else: raise ValueError('another analysis process may still be running')
    atomic_create(path,dict(pid=os.getpid(),run_id=run_id))


def run(work_dir):
    require_clean_tree();work_dir=Path(work_dir).resolve()
    record=load_registration(work_dir)
    if runtime()!=record['runtime']: raise ValueError('analysis backend/version differs from registration')
    directory=Path(record['corpus']);prereg,paths=verify_corpus(directory,record['corpus_manifest_sha256'])
    if sha(directory/'preregistration.json')!=record['collection_preregistration_sha256']:
        raise ValueError('collection preregistration changed')
    for name,expected in record['collection_files_sha256'].items():
        if sha(directory/name)!=expected: raise ValueError('collection provenance file changed')
    output=Path(record['output'])
    if output.exists(): raise FileExistsError('output already exists; inspect it before any resume')
    store=CandidateStore(work_dir/'candidates',record)
    run_id=uuid.uuid4().hex;lock=work_dir/'running.lock.json';acquire_lock(lock,run_id)
    log_dir=work_dir/'runs';log_dir.mkdir(exist_ok=True)
    log=(log_dir/f'{run_id}.jsonl').open('x')
    def progress(event):
        event=dict(event,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=run_id)
        text=json.dumps(event,sort_keys=True);print(text,flush=True);log.write(text+'\n');log.flush()
    try:
        progress(dict(event='analysis_started',implementation=IMPLEMENTATION,analysis_commit=record['analysis_commit']))
        generations=[];details=[]
        for path in paths:
            generation=int(path.stem.split('_')[-1])
            if generation not in record['analysis_generations']: continue
            meta=json.loads(path.read_text())
            if meta['generation']!=generation: raise ValueError('snapshot generation mismatch')
            progress(dict(event='generation_started',generation=generation))
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
                arrays={key:archive[key] for key in archive.files}
            rows,measurements=simulate_snapshot(meta,arrays,record['policy'],draws=record['draws'],checkpoint=store,progress=progress)
            generations.append(rows);details.extend(measurements)
            progress(dict(event='generation_complete',generation=generation,candidates_complete=36))
        rows=aggregate(generations,record['policy'])
        metrics=work_dir/'metrics';metrics.mkdir(exist_ok=True)
        once_or_equal(metrics/'aggregated.json',rows);once_or_equal(metrics/'measurements.json',details)
        provenance=dict(source='option-b',seed=104729,formal_seeds=record['formal_seeds'],
            commit=record['analysis_commit'],protocol_sha256=record['protocol_sha256'],
            artifact_root=str(directory),input_sha256=sha(metrics/'aggregated.json'))
        result=freeze(record['policy'],rows,output,provenance)
        atomic_create(output/'analysis_version.json',dict(record,sha256=digest(record)))
        atomic_create(output/'simulation_measurements.json',details)
        result['analysis_provenance']=dict(implementation=IMPLEMENTATION,analysis_commit=record['analysis_commit'],
            collection_commit=record['collection_commit'],analysis_registration_sha256=digest(record),
            corpus_manifest_sha256=record['corpus_manifest_sha256'])
        write_launch_configs(OmegaConf.create(record['formal_base_config']),result,output)
        hashes(output,'checksums.sha256')
        progress(dict(event='analysis_complete',selected_id=result['selected_id'],output=str(output)))
        return result
    except BaseException as error:
        progress(dict(event='analysis_failed',error_type=type(error).__name__))
        raise
    finally:
        log.close()
        if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():
            lock.rename(work_dir/f'finished-lock-{run_id}.json')


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ['directory','formal-base-config','work-dir','output','corpus-sha256']:
        p.add_argument('--'+name,required=True)
    p=sub.add_parser('run');p.add_argument('--work-dir',required=True)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.directory,args.formal_base_config,args.work_dir,args.output,args.corpus_sha256)
    else: run(args.work_dir)


if __name__=='__main__': main()
