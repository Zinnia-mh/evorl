"""Append-only formal Pilot registry. Host tooling; native learning is unchanged.

Only successful, fully audited runs can enter frozen_baseline. Synthetic test
reports never enter the real registry. No CLI command launches a matrix.
"""
import argparse,contextlib,csv,fcntl,hashlib,importlib.metadata,json,math,os,platform
import shutil,statistics,subprocess,sys,time,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=dict(A='uniform',B='fitness',C='td',D='utility')
SEEDS=[1,2,3]
PACKAGES=['jax','jaxlib','brax','flax','optax','numpy','orbax-checkpoint','omegaconf','scipy','mujoco','mujoco-mjx','jax-cuda12-plugin','jax-cuda12-pjrt']
# Conservative: any shared file change rejects reuse. D improvements must be
# isolated to the Utility scorer or an explicitly dedicated utility_variants module.
D_ONLY=['evorl/algorithms/erl/erl_td3/erl_learner_utility.py']
NONCOMMON=['arm','score_kind','alpha','diagnostic_enabled','replay_temperature',
           'freshness_window','min_resident','diagnostic_interval','utility_variant',
           'utility_options','output_dir','arm_manifest_path','recorders','project',
           'tags','checkpoint']

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def freeze_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def seal(directory):
    directory=Path(directory)
    with (directory/'checksums.sha256').open('x') as f:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p!=directory/'checksums.sha256':f.write(sha(p)+'  '+str(p.relative_to(directory))+'\n')
def verify_seal(directory):
    directory=Path(directory)
    for line in (directory/'checksums.sha256').read_text().splitlines():
        h,p=line.split('  ',1);target=(directory/p).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or sha(target)!=h:raise ValueError('checksum mismatch: '+p)
def common_config(c):return {k:v for k,v in c.items() if k not in NONCOMMON}
def contract(c,code,environment):return dict(config=common_config(c),shared_code=code,environment=environment)
def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a.get(k),b.get(k),prefix+'.'+k) for k in sorted(set(a)|set(b))),[])
    return [] if a==b else [dict(path=prefix,baseline=a,candidate=b)]
def compatible(a,b):
    d=differences(a,b)
    if d:raise ValueError('incompatible baseline: '+json.dumps(d,sort_keys=True))
def environment():
    packages={}
    for p in PACKAGES:
        try:packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:packages[p]=None
    return dict(python=platform.python_version(),system=platform.system(),packages=packages,jax_flags={p:os.environ.get(p,'') for p in ['JAX_ENABLE_X64','JAX_DISABLE_JIT','JAX_DEFAULT_PRNG_IMPL','JAX_PLATFORMS','XLA_FLAGS']})
def provenance():
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean committed source required')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'evorl').rglob('*.py'))}
    files['scripts/run_utility_replay_training.py']=sha(ROOT/'scripts/run_utility_replay_training.py')
    shared={p:h for p,h in files.items() if p not in D_ONLY and '/utility_variants/' not in p}
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=files,shared_code=shared,environment=environment(),tool_sha256=sha(__file__),python=sys.executable)
def protocol(c):
    warm=c['warmup_iters']*c['pop_size']*c['episodes_for_fitness'];per=c['pop_size']*c['episodes_for_fitness']+c['num_rl_agents']*c['rollout_episodes']
    steps=math.ceil((c['total_episodes']-warm)/per);generation=c['warmup_iters']+steps
    checkpoints=[g for g in range(c['warmup_iters']+1,generation+1) if g%c['eval_interval']==0 or g==generation]
    return dict(generation=generation,episodes=warm+steps*per,training_generations=steps,evaluation_generations=checkpoints,checkpoint_interval=50,keep=3)
def validate_report(q,c):
    p=protocol(c);last=q['final'];initial=q['initial'];budget=q['budget'];steps=p['training_generations'];h=c['env']['max_episode_steps']
    if q['status']!='COMPLETE':raise ValueError('only COMPLETE runs can freeze')
    expected_initial=dict(generation=c['warmup_iters'],episodes=c['warmup_iters']*c['pop_size']*c['episodes_for_fitness'],actor_optimizer_count=0,critic_optimizer_count=0)
    expected_initial['timesteps']=expected_initial['episodes']*h
    if initial!=expected_initial:raise ValueError('unexpected warmup counters')
    if c['rl_updates_mode']!='iter':raise ValueError('unsupported update mode')
    actor_per=math.ceil(c['pop_size']*c['episodes_for_fitness']*h*c['rl_updates_frac'])//c['actor_update_interval']
    final=dict(generation=p['generation'],episodes=p['episodes'],timesteps=p['episodes']*h,actor_optimizer_count=steps*actor_per,critic_optimizer_count=steps*actor_per*c['actor_update_interval'])
    if last!=final:raise ValueError('actual final counter mismatch')
    expected=dict(arm=c['arm'],score_kind=c['score_kind'],environment_episodes=last['episodes']-initial['episodes'],environment_timesteps=last['timesteps']-initial['timesteps'],rl_updates=last['actor_optimizer_count'],ea_ask=steps,ea_tell=steps,batch_size=c['batch_size'],evaluation_checkpoints=p['evaluation_generations'],evaluation_episodes=len(p['evaluation_generations'])*c['eval_episodes']*c['num_rl_agents'],configured_total_episodes=c['total_episodes'])
    if any(budget.get(k)!=v for k,v in expected.items()):raise ValueError('actual training/evaluation audit mismatch')
    rows=q['evaluations']
    if [r['generation'] for r in rows]!=p['evaluation_generations']:raise ValueError('missing/duplicate evaluations')
    per=c['pop_size']*c['episodes_for_fitness']+c['num_rl_agents']*c['rollout_episodes']
    for row in rows:
        x=(initial['episodes']+(row['generation']-c['warmup_iters'])*per)*h
        if row['environment_timesteps']!=x or not math.isfinite(row['return_mean']):raise ValueError('invalid evaluation record')
    auc=sum(.5*(a['return_mean']+b['return_mean'])*(b['environment_timesteps']-a['environment_timesteps']) for a,b in zip(rows,rows[1:]))
    late=[r['return_mean'] for r in rows[-5:]]
    return dict(auc=auc,auc_domain=[rows[0]['environment_timesteps'],rows[-1]['environment_timesteps']],auc_normalized=auc/(rows[-1]['environment_timesteps']-rows[0]['environment_timesteps']),final_return=rows[-1]['return_mean'],late_mean=statistics.mean(late),late_checkpoint_sd=statistics.stdev(late),late_evaluation_count=len(late),late_generations=[r['generation'] for r in rows[-5:]],variability_note='late_checkpoint_sd is not independent-seed uncertainty')
def aggregate(rows):
    seeds=[r['seed'] for r in rows]
    if sorted(seeds)!=SEEDS:raise ValueError('exactly three independent seeds 1,2,3 required')
    return dict(n_seeds=3,seeds=seeds,**{k:dict(mean=statistics.mean([r[k] for r in rows]),sample_sd=statistics.stdev([r[k] for r in rows])) for k in ['auc','final_return','late_mean']})
def config_data(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path),resolve=True)
def checked_id(value):
    if not value or not all(c.isalnum() or c in '-_.' for c in value) or value in ['.','..']:raise ValueError('unsafe experiment id')
    return value

def prepare(frozen,directory):
    from evorl.algorithms.erl.erl_td3.utility_replay_training import prepare_protocol
    directory=Path(directory).resolve();prov=provenance();directory.mkdir(parents=True,exist_ok=False)
    freeze_json(directory/'provenance.json',prov)
    entries=[]
    for seed in SEEDS:
        prepare_protocol(frozen,directory/'protocols'/f'seed_{seed}',seed=seed,eval_interval=10,eval_episodes=128,total_episodes=100000)
        for arm in ARMS:
            cfg=directory/'protocols'/f'seed_{seed}'/f'arm_{arm}.yaml';c=config_data(cfg);p=protocol(c)
            if (p['generation'],p['episodes'],len(p['evaluation_generations']))!=(3031,100013,303):raise ValueError('formal budget mismatch')
            rel=f'manifests/{arm}_seed_{seed}.json';output=directory/'runs'/('D0' if arm=='D' else arm)/f'seed_{seed}'
            entry=dict(schema=1,status='READY_NOT_RUN',arm=arm,version='D0' if arm=='D' else 'ABC-v1',seed=seed,commit=prov['commit'],config_path=str(cfg),config_sha256=sha(cfg),output_directory=str(output),environment=prov['environment'],contract=contract(c,prov['shared_code'],prov['environment']),protocol=p,checkpoint_policy=dict(interval=50,keep=3),evaluation_prng='native evaluate advances baseline key; identical frozen schedule',development_policy='D variants use independent development seeds before registration; never select on formal seeds')
            freeze_json(directory/rel,entry);entries.append(rel)
    freeze_json(directory/'experiment.json',dict(schema=1,seeds=SEEDS,calibration_seed=104729,development_seed=2027,manifests=entries,baseline_status='NOT_RUN',selection_policy='retain every D version including failures; no formal-seed parameter selection',late_window='last five scheduled evaluations; across-seed statistics use seeds only'))
    seal(directory)
    return dict(status='READY',manifests=12,directory=str(directory),checksums_sha256=sha(directory/'checksums.sha256'))

def entry_path(directory,arm,seed,version='D0'):
    if arm=='D' and version!='D0':return directory/'versions'/checked_id(version)/f'manifest_seed_{seed}.json'
    return directory/'manifests'/f'{arm}_seed_{seed}.json'
def get_entry(directory,arm,seed,version='D0'):
    verify_seal(directory);p=entry_path(directory,arm,seed,version)
    if arm=='D' and version!='D0':verify_seal(p.parent)
    e=load(p)
    if sha(e['config_path'])!=e['config_sha256']:raise ValueError('registered config changed')
    return e

def archive(directory,entry):
    """Called automatically only after native training completes successfully."""
    run=Path(entry['output_directory']);completion=load(run/'completion.json');segment=Path(completion['segment']);q=load(segment/'report.json')
    if sha(segment/'report.json')!=completion['report_sha256']:raise ValueError('completion report checksum mismatch')
    c=config_data(run/'resolved_config.yaml');original=config_data(entry['config_path'])
    if {k:v for k,v in c.items() if k!='output_dir'}!={k:v for k,v in original.items() if k!='output_dir'}:raise ValueError('actual resolved config differs from registered config')
    manifest=load(run/'manifest.json')
    if sha(run/'resolved_config.yaml')!=manifest['resolved_config_sha256'] or manifest['commit']!=entry['commit'] or manifest['checkpoint_policy']['interval']!=50 or manifest['checkpoint_policy']['keep']!=3:raise ValueError('run provenance/policy mismatch')
    metrics=validate_report(q,c)
    destination=directory/'registry'/('baselines' if entry['arm']!='D' else 'D_versions')/(entry['arm'] if entry['arm']!='D' else entry['version'])/f"seed_{entry['seed']}"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with (destination.parent/'.freeze.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists():raise FileExistsError('immutable frozen entry exists')
        temp=destination.parent/('.pending-'+uuid.uuid4().hex);temp.mkdir()
        for name in ['manifest.json','resolved_config.yaml','completion.json']:shutil.copy2(run/name,temp/name)
        freeze_json(temp/'registered_manifest.json',entry)
        # All logs and reports survive resume, including interrupted segments.
        for p in sorted((run/'segments').rglob('*')):
            if p.is_file() and 'checkpoints' not in p.parts:
                target=temp/'segments'/p.relative_to(run/'segments');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        if (run/'launcher_logs').exists():shutil.copytree(run/'launcher_logs',temp/'launcher_logs')
        cps=sorted((p for p in (run/'segments').glob('*/checkpoints/*') if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name))[-3:]
        if not cps or str(Path(q['last_checkpoint']).resolve())!=str(cps[-1].resolve()) or int(cps[-1].name)!=protocol(c)['generation']:raise ValueError('missing final checkpoint')
        for cp in cps:
            seal_path=cp.parent.parent/'checkpoint_checksums'/f'{cp.name}.json';expected=load(seal_path)
            actual={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()}
            if actual!=expected:raise ValueError('checkpoint seal mismatch')
            shutil.copytree(cp,temp/'checkpoints'/cp.name);shutil.copy2(seal_path,temp/f'checkpoint_{cp.name}_sha256.json')
        with (temp/'evaluations.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(q['evaluations'][0]));writer.writeheader();writer.writerows(q['evaluations'])
        attempts=directory/'attempts'/entry['version']/entry['arm']/f"seed_{entry['seed']}"
        timings=[load(p) for p in attempts.glob('*/result.json')]
        if not timings or any(x['status']=='RUNNING' for x in timings):raise ValueError('missing completed launcher timing')
        wall=sum(x['wall_seconds'] for x in timings)
        unknown=[str(p) for p in attempts.glob('*') if p.is_dir() and not (p/'result.json').exists()]
        shutil.copytree(attempts,temp/'attempt_history')
        freeze_json(temp/'entry.json',dict(status='frozen_baseline' if entry['arm']!='D' else 'frozen_D_result',arm=entry['arm'],version=entry['version'],seed=entry['seed'],contract=entry['contract'],commit=entry['commit'],metrics=metrics,actual=q['final'],actual_budget=q['budget'],wall_seconds=wall if not unknown else None,known_wall_seconds=wall,unknown_wall_attempts=unknown,wall_scope='sum of all launcher attempts including interrupted native execution, excludes archive copy; hard-killed attempts without an end record are UNKNOWN',source_run=str(run),attempts=[str(p) for p in attempts.glob('*')]))
        seal(temp);verify_seal(temp);temp.rename(destination)
    return dict(status='FROZEN',directory=str(destination),checksums_sha256=sha(destination/'checksums.sha256'))

def run(directory,arm,seed,version,resume=None,stop=None):
    e=get_entry(directory,arm,seed,version);prov=provenance()
    if prov['commit']!=e['commit'] or prov['environment']!=e['environment']:raise ValueError('registered source/environment mismatch')
    from omegaconf import OmegaConf
    from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training
    import jax
    if jax.default_backend()!='gpu':raise ValueError('formal runs require the GPU backend')
    c=OmegaConf.load(e['config_path'])
    attempt=directory/'attempts'/e['version']/arm/f'seed_{seed}'/uuid.uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
    freeze_json(attempt/'start.json',dict(status='RUNNING',manifest=e,start_utc=time.time(),resume=str(resume) if resume else None))
    output=Path(e['output_directory']);logs=output/'launcher_logs';started=time.monotonic()
    # Redirect the same native stdout/stderr to append-only per-attempt logs.
    log=attempt/'training.log'
    try:
        with log.open('x',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            # fd redirection also captures native JAX/LLVM messages.
            oldout,olderr=os.dup(1),os.dup(2)
            try:
                os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
                _,q=run_training(c,output,resume_checkpoint=resume,stop_after_generation=stop,checkpoint_interval=50,keep=3)
            finally:os.dup2(oldout,1);os.dup2(olderr,2);os.close(oldout);os.close(olderr)
        result=dict(status=q['status'],wall_seconds=time.monotonic()-started)
    except BaseException as error:
        freeze_json(attempt/'result.json',dict(status='FAILED',error=repr(error),wall_seconds=time.monotonic()-started));raise
    freeze_json(attempt/'result.json',result)
    logs.mkdir(exist_ok=True);shutil.copy2(log,logs/(attempt.name+'.log'))
    if result['status']=='COMPLETE':return archive(directory,e)
    return dict(status='PAUSED',last_checkpoint=q['last_checkpoint'])

def register_d(directory,version,config_directory,development_evidence):
    from omegaconf import OmegaConf
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import validate_arm_manifest
    checked_id(version)
    if version=='D0':raise ValueError('D0 is already preregistered')
    evidence=load(development_evidence)
    if evidence.get('seed') in SEEDS+[104729] or evidence.get('seed') is None:raise ValueError('independent development seed evidence required')
    prov=provenance();configs=[]
    if evidence.get('head')!=prov['commit'] or evidence.get('status') not in ['PASS','COMPLETE']:raise ValueError('development evidence must identify this committed D version and successful validation')
    if not evidence.get('report_path') or sha(evidence['report_path'])!=evidence.get('report_sha256'):raise ValueError('development report checksum missing or invalid')
    for seed in SEEDS:
        source=Path(config_directory)/f'seed_{seed}'/'arm_D.yaml';c=OmegaConf.load(source);validate_arm_manifest(c);data=OmegaConf.to_container(c,resolve=True)
        if data['seed']!=seed or data['arm']!='utility':raise ValueError('wrong D seed/arm')
        a=get_entry(directory,'A',seed);compatible(a['contract'],contract(data,prov['shared_code'],prov['environment']));configs.append((seed,source,data))
    target=directory/'versions'/version;target.mkdir(parents=True,exist_ok=False)
    shutil.copy2(development_evidence,target/'development_evidence.json')
    freeze_json(target/'provenance.json',prov)
    for seed,source,data in configs:
        # Pin both parsed config and its native four-arm manifest. Validation
        # excludes manifest path, so moving it never changes sampling settings.
        folder=target/f'seed_{seed}';folder.mkdir();shutil.copy2(data['arm_manifest_path'],folder/'arm_manifest.json')
        data['arm_manifest_path']=str(folder/'arm_manifest.json');OmegaConf.save(OmegaConf.create(data),folder/'arm_D.yaml')
        old=get_entry(directory,'D',seed);old.update(version=version,commit=prov['commit'],config_path=str(folder/'arm_D.yaml'),config_sha256=sha(folder/'arm_D.yaml'),output_directory=str(directory/'runs'/version/f'seed_{seed}'),environment=prov['environment'],contract=contract(data,prov['shared_code'],prov['environment']))
        freeze_json(target/f'manifest_seed_{seed}.json',old)
    seal(target);return dict(status='READY_NOT_RUN',version=version)

def compare(directory,version):
    groups={};seed_reference={};curves={}
    for arm in ARMS:
        rows=[];curves[arm]={}
        for seed in SEEDS:
            p=directory/'registry'/('D_versions' if arm=='D' else 'baselines')/(version if arm=='D' else arm)/f'seed_{seed}'
            verify_seal(p);entry=load(p/'entry.json')
            if entry['status']!=('frozen_D_result' if arm=='D' else 'frozen_baseline'):raise ValueError('not a frozen completed result')
            if arm=='A':seed_reference[seed]=entry['contract']
            else:compatible(seed_reference[seed],entry['contract'])
            with (p/'evaluations.csv').open() as f:curves[arm][str(seed)]=list(csv.DictReader(f))
            if arm!='A' and [x['environment_timesteps'] for x in curves[arm][str(seed)]]!=[x['environment_timesteps'] for x in curves['A'][str(seed)]]:raise ValueError('different evaluation step grid')
            rows.append(dict(seed=seed,wall_seconds=entry['wall_seconds'],actual=entry['actual'],**entry['metrics']))
        groups[arm]=dict(per_seed=rows,aggregate=aggregate(rows))
    output=directory/'comparisons'/checked_id(version)/uuid.uuid4().hex;output.mkdir(parents=True)
    freeze_json(output/'metrics.json',dict(status='COMPARABLE',version=version,arms=groups,statistics='three independent matched seeds; sample SD across seeds, not checkpoints'))
    freeze_json(output/'curves.json',curves);seal(output)
    return dict(status='COMPARABLE',directory=str(output))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--frozen',required=True);s.add_argument('--directory',required=True)
    for name in ['run','resume']:
        s=sub.add_parser(name);s.add_argument('--directory',required=True);s.add_argument('--arm',choices=list(ARMS),required=True);s.add_argument('--seed',type=int,choices=SEEDS,required=True);s.add_argument('--version',default='D0');s.add_argument('--stop-after-generation',type=int)
        if name=='resume':s.add_argument('--checkpoint',required=True)
    s=sub.add_parser('register-d');s.add_argument('--directory',required=True);s.add_argument('--version',required=True);s.add_argument('--configs',required=True);s.add_argument('--development-evidence',required=True)
    s=sub.add_parser('compare');s.add_argument('--directory',required=True);s.add_argument('--version',default='D0')
    s=sub.add_parser('archive');s.add_argument('--directory',required=True);s.add_argument('--arm',choices=list(ARMS),required=True);s.add_argument('--seed',type=int,choices=SEEDS,required=True);s.add_argument('--version',default='D0')
    args=p.parse_args();directory=Path(args.directory).resolve()
    artifact_root=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot').resolve()
    if not directory.is_relative_to(artifact_root):raise ValueError('registry artifacts must stay outside Git under Pilot artifact root')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    if args.command=='prepare':result=prepare(args.frozen,directory)
    elif args.command in ['run','resume']:result=run(directory,args.arm,args.seed,args.version,getattr(args,'checkpoint',None),args.stop_after_generation)
    elif args.command=='register-d':result=register_d(directory,args.version,args.configs,args.development_evidence)
    elif args.command=='archive':result=archive(directory,get_entry(directory,args.arm,args.seed,args.version))
    else:result=compare(directory,args.version)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
