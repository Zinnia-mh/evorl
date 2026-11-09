"""Independent D1 development prepare/run/resume; never launches formal seeds."""
import argparse,json,subprocess,time,uuid,resource,sys,fcntl,os
from pathlib import Path
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/d1')
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--baseline-config',required=True);a.add_argument('--directory',required=True)
    for command in ['run','resume']:
        a=sub.add_parser(command);a.add_argument('--directory',required=True);a.add_argument('--stop-after-generation',type=int)
        if command=='run':a.add_argument('--protocol',required=True)
        else:a.add_argument('--checkpoint',required=True)
    a=sub.add_parser('stop');a.add_argument('--directory',required=True)
    args=p.parse_args();directory=Path(args.directory).resolve()
    if not directory.is_relative_to(ROOT):raise ValueError('D1 artifacts require independent external D1 directory')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('D1 requires committed clean code')
    if args.command=='stop':
        if (directory/'completion.json').exists():raise ValueError('run already complete')
        manifest=json.loads((directory/'manifest.json').read_text())
        with (directory/'.training.lock').open('r') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('no active training writer')
        request=dict(run_id=manifest['run_id'],requested_unix=time.time())
        temporary=directory/('stop_request.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('x') as out:json.dump(request,out);out.flush();os.fsync(out.fileno())
        try:os.link(temporary,directory/'stop_request.json')
        finally:temporary.unlink()
        print(json.dumps(dict(status='STOP_REQUESTED',**request)),flush=True);return
    from evorl.algorithms.erl.erl_td3.utility_replay_d1_training import prepare_d1_protocol,validate_d1_manifest
    from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
    if args.command=='prepare':result=prepare_d1_protocol(args.baseline_config,directory)
    else:
        c=OmegaConf.load(Path(args.protocol)/'resolved_config.yaml' if args.command=='run' else directory/'resolved_config.yaml');validate_d1_manifest(c)
        started=time.perf_counter()
        _,result=run_training(c,directory,resume_checkpoint=args.checkpoint if args.command=='resume' else None,stop_after_generation=args.stop_after_generation,checkpoint_interval=50,keep=3)
        elapsed=time.perf_counter()-started
        log=Path(result['segment_directory'])/'launcher.json'
        with log.open('x') as f:json.dump(dict(wall_seconds=elapsed,host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,command=args.command),f,indent=2)
    print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
