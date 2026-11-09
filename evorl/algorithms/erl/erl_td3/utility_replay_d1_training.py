"""D1-only sealed development protocol and read-only Host instrumentation."""
import hashlib,json,math,subprocess,time
from pathlib import Path
import jax,numpy as np
from omegaconf import OmegaConf
from .utility_replay_d1 import ERLUtilityReplayD1Workflow,validate_d1_config,priority_value
from .utility_replay_budget import validate_arm_manifest,validate_arm_config
from .utility_replay_checkpoint import mechanism_digest,environment_contract
from .utility_replay_smoke_workflow import reconstruct_replay_draws,DrawRecorder
from .replay_distribution_audit import audit_distribution

D1_WORKFLOW='evorl.algorithms.erl.erl_td3.utility_replay_d1.ERLUtilityReplayD1Workflow'
PARAMETERS=dict(utility_variant='d1-v1',rho=.3,half_life=15.,priority_cap=.1,utility_scale=.01,freshness_window=30,min_resident=4,diagnostic_interval=1,alpha=.3)
OVERRIDES=set(PARAMETERS)|{'workflow_cls','arm_manifest_path','output_dir'}

def source_hashes():
    root=Path(__file__).resolve().parents[4]
    paths=['evorl/algorithms/erl/erl_td3/'+n+'.py' for n in ['utility_replay_d1','utility_replay_d1_training','utility_replay_training','utility_replay_checkpoint','utility_replay_budget','erl_utility_replay','erl_learner_utility','source_replay','source_residency_batch','d1_residency']]+['scripts/run_utility_replay_d1_training.py','configs/agent/erl/erl-utility-replay-d1.yaml']
    return {p:sha(root/p) for p in paths}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(c):
    d=OmegaConf.to_container(c,resolve=True)
    for k in ['output_dir','arm_manifest_path']:d.pop(k,None)
    return d
def digest(c):return hashlib.sha256(json.dumps(canonical(c),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def shared(c):return {k:v for k,v in canonical(c).items() if k not in OVERRIDES}
def _write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
def _git():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def prepare_d1_protocol(baseline_config,directory):
    baseline_config=Path(baseline_config).resolve();directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError(directory)
    base=OmegaConf.load(baseline_config);validate_arm_manifest(base)
    if int(base.seed)!=2027 or str(base.arm)!='utility':raise ValueError('D1 development requires independent seed2027 Utility baseline protocol')
    c=OmegaConf.create(OmegaConf.to_container(base,resolve=True))
    for k,v in PARAMETERS.items():OmegaConf.update(c,k,v,force_add=True)
    c.workflow_cls=D1_WORKFLOW;c.arm_manifest_path=str(directory/'protocol.json')
    validate_d1_config(c);validate_arm_config(c)
    assert shared(c)==shared(base)
    directory.mkdir(parents=True,exist_ok=False);OmegaConf.save(c,directory/'resolved_config.yaml',resolve=True)
    initial=int(c.warmup_iters)*int(c.pop_size)*int(c.episodes_for_fitness);per=int(c.pop_size)*int(c.episodes_for_fitness)+int(c.num_rl_agents)*int(c.rollout_episodes)
    target=int(c.warmup_iters)+math.ceil((int(c.total_episodes)-initial)/per)
    manifest=dict(schema=1,kind='D1-development-only',status='READY_NOT_RUN',commit=_git(),mechanism_digest=mechanism_digest(),environment=environment_contract(),source_sha256=source_hashes(),baseline_config=str(baseline_config),baseline_config_sha256=sha(baseline_config),config_hash=digest(c),parameters=PARAMETERS,config_sha256=sha(directory/'resolved_config.yaml'),final_generation=target,effective_episodes=initial+(target-int(c.warmup_iters))*per,evaluation_generations=[g for g in range(int(c.warmup_iters)+1,target+1) if g%int(c.eval_interval)==0 or g==target],index_audit_generations=[50,300],checkpoint_policy=dict(interval=50,keep=3),shared_protocol=shared(c),seed=int(c.seed))
    _write(directory/'protocol.json',manifest)
    with (directory/'checksums.sha256').open('x') as f:
        for name in ['protocol.json','resolved_config.yaml']:f.write(sha(directory/name)+'  '+name+'\n')
    return manifest

def validate_d1_manifest(c):
    validate_d1_config(c);validate_arm_config(c)
    if str(c.workflow_cls)!=D1_WORKFLOW or int(c.seed)!=2027:raise ValueError('D1 independent development workflow/seed required')
    for k,v in PARAMETERS.items():
        if getattr(c,k)!=v:raise ValueError('unfrozen D1 parameter: '+k)
    p=Path(c.arm_manifest_path).resolve();m=json.loads(p.read_text())
    for line in (p.parent/'checksums.sha256').read_text().splitlines():
        h,name=line.split(None,1)
        if name not in ['protocol.json','resolved_config.yaml'] or sha(p.parent/name)!=h:raise ValueError('D1 protocol checksum mismatch')
    if {line.split(None,1)[1] for line in (p.parent/'checksums.sha256').read_text().splitlines()}!={'protocol.json','resolved_config.yaml'}:raise ValueError('incomplete D1 seal')
    if m.get('schema')!=1 or m.get('kind')!='D1-development-only' or m['config_hash']!=digest(c) or m['parameters']!=PARAMETERS or m['config_sha256']!=sha(p.parent/'resolved_config.yaml'):raise ValueError('D1 configuration/manifest mismatch')
    if m['source_sha256']!=source_hashes() or m['commit']!=_git() or m['mechanism_digest']!=mechanism_digest() or m['environment']!=environment_contract():raise ValueError('D1 code/version/environment mismatch')
    if sha(m['baseline_config'])!=m['baseline_config_sha256'] or shared(c)!=m['shared_protocol']:raise ValueError('D1 shared baseline protocol changed')
    return m

class D1TrainingWorkflow(ERLUtilityReplayD1Workflow):
    def stop_requested(self,directory,run_id):
        p=Path(directory)/'stop_request.json'
        if not p.exists():return False
        request=json.loads(p.read_text())
        if request.get('run_id')!=run_id:raise ValueError('stop request belongs to another run')
        return True

    def acknowledge_stop(self,directory,segment):
        (Path(directory)/'stop_request.json').rename(Path(segment)/'accepted_stop_request.json')

    def observe_training_step(self,before,metrics,state,segment,step_seconds):
        """Called after unchanged JIT learning; no callbacks, keys, state writes."""
        observation_start=time.perf_counter()
        records=self._latest_diagnostic_records;g=int(state.metrics.iterations)
        if len(records)!=int(self.config.pop_size) or any(not r.eligible or r.utility is None or r.loss_before is None or not np.isfinite(r.utility) or not np.isfinite(r.loss_before) for r in records):raise ValueError('D1 new-source diagnostics incomplete/nonfinite')
        np.testing.assert_array_equal(state.key,jax.random.split(before.key,4)[0])
        table=state.utility_replay_state.score_table
        ids,measured,versions,valid,counts=jax.device_get((table.source_token,table.measured_generation,table.version,table.validity,table.resident_count))
        lookup={int(ids[i]):int(i) for i in np.flatnonzero(np.asarray(counts)>0)}
        sources=[]
        for r in records:
            token=g*int(self.config.pop_size)+int(r.individual_id);i=lookup.get(token)
            if i is None or not valid[i] or int(measured[i])!=g or int(versions[i])!=int(table.current_version):raise ValueError('D1 new score identity/version/residency mismatch')
            sources.append(dict(source_token=token,U=float(r.utility),L_before=float(r.loss_before),r=float(r.utility)/max(abs(float(r.loss_before)),1e-8),v=priority_value(r.utility,r.loss_before),measured_generation=int(measured[i]),version=int(versions[i]),resident_count=int(counts[i])))
        values=np.asarray([r.utility for r in records]);row=dict(self._latest_d1_metrics,step_seconds=step_seconds,new_source_count=len(records),positive_utility_fraction=float(np.mean(values>0)),negative_utility_fraction=float(np.mean(values<0)),zero_utility_fraction=float(np.mean(values==0)),utility_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist(),sources=sources,TV=self._latest_d1_metrics['probability_l1']/2,distribution_build_count=self._latest_score_lifecycle_audit['distribution_build_count'],diagnostic_seconds=self._latest_score_lifecycle_audit['diagnostic_seconds'],source_statistics_seconds=self._latest_score_lifecycle_audit['resident_count_seconds'],distribution_refresh_seconds=self._latest_score_lifecycle_audit['distribution_refresh_seconds'],learner_seconds=self._latest_score_lifecycle_audit['weighted_learner_phase_seconds'])
        if not np.all(np.isfinite([row[k] for k in ['probability_l1','ESS','priority_mass','score_coverage']])):raise ValueError('nonfinite D1 sampler metrics')
        selected=json.loads(Path(self.config.arm_manifest_path).read_text())['index_audit_generations']
        if g in selected:
            replay,key,updates,u=self._latest_d1_inputs
            live=(np.arange(replay.source_token.size)<int(replay.buffer_size))&(np.asarray(replay.origin_type)==0)
            tokens,n=np.unique(np.asarray(replay.source_token)[live],return_counts=True)
            actual={int(t):int(n) for t,n in zip(tokens,n)}
            if {int(ids[i]):int(counts[i]) for i in np.flatnonzero(np.asarray(counts)>0)}!=actual:raise ValueError('D1 Ring residency mismatch')
            mask=live
            row['distribution_audit']=audit_distribution(*jax.device_get((u.distribution.source_probability,table.resident_count,u.distribution.transition_probability,u.distribution.transition_cdf,mask)),float(u.distribution.ess_fraction),cdf_in_training_sampling=bool(u.distribution_prepared))
            trace=reconstruct_replay_draws(replay,key,u,num_updates=int(updates),num_draws=int(self.config.actor_update_interval)*int(self.config.num_rl_agents),batch_size=int(self.config.batch_size),uniform=False);jax.block_until_ready(trace)
            np.testing.assert_array_equal(trace.terminal_intervention_key,state.utility_replay_state.intervention_key)
            recorder=DrawRecorder();recorder.record(*jax.device_get(trace[:5]));draw=recorder.summary()
            if draw['rl_changed'] or draw['origin_changed']:raise ValueError('D1 real learner index isolation failed')
            if row['eligible_count'] and bool(u.distribution_prepared) and not draw['changed_ea']:raise ValueError('D1 real EA intervention absent')
            row['index_reconstruction']=draw;row['index_scope']='deterministic reconstruction of actual TD3 sampler inputs; terminal intervention key verified; not JIT callback'
            np.savez_compressed(segment/f'index_examples_g{g}.npz',baseline=np.asarray(trace.baseline_indices[:2]),final=np.asarray(trace.final_indices[:2]),ea=np.asarray(trace.ea_slot_mask[:2]),rl=np.asarray(trace.rl_slot_mask[:2]))
        row['host_observer_seconds']=time.perf_counter()-observation_start
        from .utility_replay_training import _json
        with (segment/'d1_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False,default=_json)+'\n');f.flush()
