"""Durable Pilot host loop; training step/evaluate and their PRNG are unchanged."""
import contextlib,csv,fcntl,hashlib,json,math,subprocess,traceback,uuid,time
from pathlib import Path
import jax,numpy as np
from omegaconf import OmegaConf
from .erl_utility_replay import ERLUtilityReplayWorkflow,_optimizer_step_count
from .utility_replay_budget import BudgetAudit,prepare_arm_manifest,validate_arm_manifest,validate_arm_set
from .utility_replay_checkpoint import PilotCheckpointStore,config_contract,mechanism_digest,restore_template
from .utility_replay_smoke_workflow import initialization_fingerprint

WORKFLOW='evorl.algorithms.erl.erl_td3.erl_utility_replay.ERLUtilityReplayWorkflow'

def _runner_components(config):
    if config.workflow_cls == WORKFLOW:
        return ERLUtilityReplayWorkflow, PilotCheckpointStore, validate_arm_manifest
    from .utility_replay_d1_training import D1_WORKFLOW, D1TrainingWorkflow, validate_d1_manifest
    from .utility_replay_d1 import D1CheckpointStore
    if config.workflow_cls == D1_WORKFLOW:
        return D1TrainingWorkflow, D1CheckpointStore, validate_d1_manifest
    raise ValueError('unsupported durable workflow')

def _write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False,default=_json)
def _json(value):
    if isinstance(value,(np.ndarray,jax.Array)):return np.asarray(value).tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def _counters(state):
    return dict(generation=int(state.metrics.iterations),episodes=int(state.metrics.sampled_episodes),timesteps=int(state.metrics.sampled_timesteps),actor_optimizer_count=_optimizer_step_count(state.opt_state.actor),critic_optimizer_count=_optimizer_step_count(state.opt_state.critic))
def _budget(audit):return {k:v for k,v in vars(audit).items() if k!='config'}
def _attach(workflow,runtime):
    b=runtime['budget'];audit=BudgetAudit(workflow.config,initial_episodes=b['initial_episodes'],initial_iteration=b['initial_iteration'],initial_timesteps=b['initial_timesteps'])
    for key,value in b.items():
        if key not in vars(audit) or key=='config':raise ValueError('unknown budget checkpoint field')
        setattr(audit,key,value)
    workflow._budget_audit=audit;workflow._observed_ec_rollouts=0
    return audit
def _validate_runtime(state,runtime,audit):
    actual=_counters(state);initial=runtime['initial']
    if actual['generation']!=initial['generation']+audit.steps or actual['episodes']!=initial['episodes']+audit.episodes or actual['timesteps']!=initial['timesteps']+audit.timesteps:raise ValueError('resume state/budget mismatch')
    if actual['actor_optimizer_count']-initial['actor_optimizer_count']!=audit.updates or actual['critic_optimizer_count']-initial['critic_optimizer_count']!=audit.updates*int(audit.config.actor_update_interval):raise ValueError('resume optimizer count mismatch')
    if audit.ask!=audit.steps or audit.tell!=audit.steps or audit.checkpoints!=[r['generation'] for r in runtime['evaluations']]:raise ValueError('resume population/evaluation journal mismatch')
    if len(audit.checkpoints)!=len(set(audit.checkpoints)):raise ValueError('duplicate evaluation checkpoint')

@contextlib.contextmanager
def _lock(directory):
    with (directory/'.training.lock').open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another training/resume writer is active')
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def run_training(config,directory,*,resume_checkpoint=None,stop_after_generation=None,checkpoint_interval=50,keep=3):
    c=OmegaConf.create(OmegaConf.to_container(config,resolve=True));directory=Path(directory).resolve()
    workflow_type,checkpoint_type,validator=_runner_components(c)
    sealed=validator(c)
    if c.workflow_cls!=WORKFLOW and dict(interval=checkpoint_interval,keep=keep)!=sealed['checkpoint_policy']:raise ValueError('D1 checkpoint policy differs from frozen manifest')
    if not c.save_replay_buffer or c.checkpoint.enable:raise ValueError('durable Pilot requires full replay and one runner-owned native checkpoint manager')
    if checkpoint_interval<1 or keep<2:raise ValueError('checkpoint interval>=1 and keep>=2 required')
    c.output_dir=str(directory)
    if resume_checkpoint is None:
        directory.mkdir(parents=True,exist_ok=False)
        manifest=dict(schema=1,run_id=uuid.uuid4().hex,config_contract=config_contract(c),mechanism_digest=mechanism_digest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint_policy=dict(interval=checkpoint_interval,keep=keep,manager='native Orbax Composite(state,runtime)'),logging='append-only segments; full canonical evaluation prefix per segment')
        OmegaConf.save(c,directory/'resolved_config.yaml',resolve=True)
        manifest['resolved_config_sha256']=hashlib.sha256((directory/'resolved_config.yaml').read_bytes()).hexdigest()
        _write(directory/'manifest.json',manifest)
    else:
        if (directory/'completion.json').exists():raise ValueError('run is already complete')
        manifest=json.loads((directory/'manifest.json').read_text())
        if hashlib.sha256((directory/'resolved_config.yaml').read_bytes()).hexdigest()!=manifest['resolved_config_sha256'] or config_contract(c)!=manifest['config_contract'] or mechanism_digest()!=manifest['mechanism_digest']:raise ValueError('incompatible resume configuration or code')
        resume_checkpoint=Path(resume_checkpoint).resolve()
        if not resume_checkpoint.is_relative_to(directory/'segments'):raise ValueError('checkpoint belongs to another run directory')
    with _lock(directory):
        if (directory/'completion.json').exists():raise ValueError('run is already complete')
        segment=directory/'segments'/uuid.uuid4().hex;segment.mkdir(parents=True,exist_ok=False)
        _write(segment/'segment.json',dict(parent_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        workflow=type('CheckpointPilot',(workflow_type,),{}).build_from_config(c,enable_jit=bool(c.enable_jit))
        restored_hash=None;last_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        try:
            if resume_checkpoint:
                template=restore_template(workflow)
                with checkpoint_type(resume_checkpoint.parent,c,read_only=True) as reader:state,runtime=reader.restore(int(resume_checkpoint.name),template)
                if runtime['run_id']!=manifest['run_id']:raise ValueError('checkpoint run identity mismatch')
                workflow.recorder.init();restored_hash=initialization_fingerprint(state)
                audit=_attach(workflow,runtime)
            else:
                state=workflow.init(jax.random.PRNGKey(int(c.seed)));initial=_counters(state)
                audit=BudgetAudit(c,initial_episodes=initial['episodes'],initial_iteration=initial['generation'],initial_timesteps=initial['timesteps'])
                workflow._budget_audit=audit;workflow._observed_ec_rollouts=0
                per_step=int(c.pop_size)*int(c.episodes_for_fitness)+int(c.num_rl_agents)*int(c.rollout_episodes)
                target=initial['generation']+math.ceil(max(0,int(c.total_episodes)-initial['episodes'])/per_step)
                runtime=dict(run_id=manifest['run_id'],initial=initial,budget=_budget(audit),evaluations=[],target_generation=target)
            _validate_runtime(state,runtime,audit)
            target=int(runtime['target_generation']);stop=target if stop_after_generation is None else min(target,int(stop_after_generation))
            if stop<int(state.metrics.iterations):raise ValueError('stop generation precedes restored state')
            checkpoint_directory=resume_checkpoint.parent if c.workflow_cls!=WORKFLOW and resume_checkpoint else segment/'checkpoints'
            requested_stop=False
            with checkpoint_type(checkpoint_directory,c,interval=checkpoint_interval,keep=keep) as store,(segment/'events.jsonl').open('x') as events:
                if resume_checkpoint is None:last_checkpoint=str(store.save(state,runtime,force=True))
                for generation in range(int(state.metrics.iterations)+1,stop+1):
                    before=state;step_started=time.perf_counter()
                    metrics,state=workflow.step(state);jax.block_until_ready((metrics,state))
                    observer=getattr(workflow,'observe_training_step',None)
                    if observer is not None:observer(before,metrics,state,segment,time.perf_counter()-step_started)
                    events.write(json.dumps(dict(type='train',**_counters(state),metrics=metrics.to_local_dict()),default=_json,allow_nan=False)+'\n');events.flush()
                    if generation%int(c.eval_interval)==0 or generation==target:
                        evaluation,state=workflow.evaluate(state);jax.block_until_ready((evaluation,state))
                        returns=np.asarray(evaluation.rl_episode_returns)
                        if not np.all(np.isfinite(returns)):raise ValueError('nonfinite evaluation return')
                        row=dict(generation=generation,environment_timesteps=int(state.metrics.sampled_timesteps),environment_episodes=int(state.metrics.sampled_episodes),return_mean=float(returns.mean()),return_mean_by_rl_agent=returns.tolist(),mean_episode_lengths=np.asarray(evaluation.rl_episode_lengths).tolist())
                        runtime['evaluations'].append(row)
                        events.write(json.dumps(dict(type='evaluation',**row),allow_nan=False)+'\n');events.flush()
                    runtime['budget']=_budget(audit);_validate_runtime(state,runtime,audit)
                    stop_hook=getattr(workflow,'stop_requested',None)
                    requested_stop=bool(stop_hook(directory,manifest['run_id'])) if stop_hook else False
                    saved=store.save(state,runtime,force=generation in [stop,target] or requested_stop)
                    if saved:last_checkpoint=str(saved)
                    if requested_stop:break
                done=int(state.metrics.iterations)==target
                budget=audit.finalize(final_episodes=int(state.metrics.sampled_episodes),final_iteration=int(state.metrics.iterations),final_timesteps=int(state.metrics.sampled_timesteps)) if done else None
                rows=runtime['evaluations'];auc=sum(.5*(a['return_mean']+b['return_mean'])*(b['environment_timesteps']-a['environment_timesteps']) for a,b in zip(rows,rows[1:])) if len(rows)>1 else None
                report=dict(status='COMPLETE' if done else 'PAUSED',initial=runtime['initial'],final=_counters(state),budget=budget,partial_budget=_budget(audit),evaluations=rows,auc=auc,auc_domain=[rows[0]['environment_timesteps'],rows[-1]['environment_timesteps']] if rows else None,final_return=rows[-1]['return_mean'] if done and rows else None,last_checkpoint=last_checkpoint,segment_directory=str(segment),restored_state_sha256=restored_hash,state_sha256=initialization_fingerprint(state))
                with (segment/'evaluations.csv').open('x',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['generation','environment_timesteps','return_mean']);writer.writeheader();writer.writerows(rows)
                _write(segment/'report.json',report)
                if requested_stop:workflow.acknowledge_stop(directory,segment)
                if done:_write(directory/'completion.json',dict(segment=str(segment),checkpoint=last_checkpoint,report_sha256=hashlib.sha256((segment/'report.json').read_bytes()).hexdigest()))
                return state,report
        except Exception:
            _write(segment/'error.json',dict(error=traceback.format_exc(),last_checkpoint=last_checkpoint))
            raise
        finally:workflow.close()

def prepare_protocol(frozen,directory,*,seed=1,eval_interval=10,eval_episodes=128,total_episodes=100000):
    from .utility_replay_smoke import verify
    frozen=Path(frozen).resolve();directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError(directory)
    if eval_interval<1 or eval_episodes<1:raise ValueError('invalid evaluation schedule')
    verify(frozen,'checksums.sha256')
    configs=[OmegaConf.load(frozen/f'arm_{label}.yaml') for label in 'ABCD'];validate_arm_set(configs)
    for c in configs:validate_arm_manifest(c)
    directory.mkdir(parents=True,exist_ok=False)
    for label,c in zip('ABCD',configs):
        c.seed=int(seed);c.eval_interval=int(eval_interval);c.eval_episodes=int(eval_episodes);c.num_eval_envs=int(eval_episodes);c.total_episodes=int(total_episodes);c.arm_manifest_path=str(directory/'arm_manifest.json')
        c.workflow_cls=WORKFLOW;c.checkpoint.enable=False;c.save_replay_buffer=True
        OmegaConf.save(c,directory/f'arm_{label}.yaml',resolve=True)
    prepare_arm_manifest(configs,directory/'arm_manifest.json')
    c=configs[0];initial=int(c.warmup_iters)*int(c.pop_size)*int(c.episodes_for_fitness);per_step=int(c.pop_size)*int(c.episodes_for_fitness)+int(c.num_rl_agents)*int(c.rollout_episodes);iterations=math.ceil(max(0,int(c.total_episodes)-initial)/per_step)
    manifest=dict(schema=1,kind='configuration-only; no training launched',frozen_source=str(frozen),frozen_checksums_sha256=hashlib.sha256((frozen/'checksums.sha256').read_bytes()).hexdigest(),seed=int(seed),configs={a:f'arm_{a}.yaml' for a in 'ABCD'},training_generations=iterations,final_generation=int(c.warmup_iters)+iterations,effective_total_episodes=initial+iterations*per_step,evaluation_prng='unchanged native evaluate advances baseline state.key; same frozen schedule for all arms',config_checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()})
    _write(directory/'protocol_manifest.json',manifest);return manifest
