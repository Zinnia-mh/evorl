"""Immutable Smoke preparation and strict validation; no implicit GPU launch."""
import hashlib,json,math
from pathlib import Path
from omegaconf import OmegaConf
from .utility_replay_budget import validate_arm_set,validate_arm_manifest,prepare_arm_manifest

STEPS=12
LABELS='ABCD'
FAIRNESS_POLICY='matched-settings-and-budget-v1'
def initialization_settings_fingerprint(config):
    values=OmegaConf.to_container(config,resolve=True)
    # The only preregistered arm differences; all other resolved settings,
    # including seed, networks, environment, EA/RL mechanisms and budgets bind.
    for name in ['arm','score_kind','alpha','diagnostic_enabled']:values.pop(name,None)
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def validate_ess_evidence(row):
    # Legacy records retain the original strict bound check below. An audit
    # version cannot be used to silently waive it without numerical evidence.
    if 'ess_audit_version' not in row:
        if any(k in row for k in ['ess_raw_float32','ess_fraction_audit64','source_probability_squared_mass64']):raise ValueError('missing ESS audit version')
        return
    names=['ess_raw_float32','ess_fraction_audit64','effective_source_count_audit64','source_count','source_probability_mass64','source_probability_squared_mass64','transition_probability_mass64','cdf_sampling_mass64','cdf_probability_max_abs_error','cdf_non_ea_interval_mass64']
    if row['ess_audit_version']!='normalized-variance-float64-v1' or any(k not in row for k in names):raise ValueError('missing/unknown ESS audit')
    if any(not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) for k in names):raise ValueError('nonfinite ESS evidence')
    n=row['source_count'];mass=row['source_probability_mass64'];squares=row['source_probability_squared_mass64'];fraction=row['ess_fraction_audit64']
    if n<0 or int(n)!=n or not 0<=fraction<=1 or row['ess_fraction']!=fraction:raise ValueError('invalid ESS evidence')
    if not math.isclose(row['effective_source_count_audit64'],n*fraction,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('ESS count mismatch')
    if n:
        if abs(mass-1)>1e-5 or abs(row['transition_probability_mass64']-1)>1e-5 or squares<=0:raise ValueError('ESS mass mismatch')
        if not math.isclose(fraction,mass*mass/(n*squares),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('ESS reference mismatch')
        if row.get('cdf_in_training_sampling',True) and row['cdf_sampling_mass64']!=1:raise ValueError('CDF mass mismatch')
    elif any(row[k]!=0 for k in ['ess_fraction','source_probability_mass64','source_probability_squared_mass64','transition_probability_mass64','cdf_sampling_mass64']):raise ValueError('invalid empty ESS')
    if row['cdf_probability_max_abs_error']<0:raise ValueError('invalid CDF evidence')
    if row.get('cdf_in_training_sampling',True):
        if 'cdf_valid_for_sampling' in row and row['cdf_valid_for_sampling'] is not True:raise ValueError('invalid active sampling CDF')
        bound=row.get('cdf_accumulation_error_bound',0.)
        if not 0<=bound<=1e-5:raise ValueError('invalid CDF error bound')
        if row.get('cdf_max_abs_reference_error',0.)>bound:raise ValueError('CDF reference mismatch')
        # Signed raw-CDF interval differences are not empirical probabilities.
        allowance=2*bound*row.get('cdf_negative_delta_count',0)
        if not -allowance<=row['cdf_non_ea_interval_mass64']<=1+allowance:raise ValueError('invalid CDF interval evidence')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False,indent=2)
def checksums(root,name):
    root=Path(root)
    with (root/name).open('x') as f:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!=name:f.write(f'{sha(p)}  {p.relative_to(root).as_posix()}\n')
def verify(root,name):
    root=Path(root).resolve()
    for line in (root/name).read_text().splitlines():
        digest,relative=line.split('  ',1);p=(root/relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=digest:raise ValueError('checksum/config tampering')

def prepare_smoke(frozen,directory,*,seed,commit):
    frozen=Path(frozen).resolve();directory=Path(directory).resolve()
    if seed not in [1,2,3]:raise ValueError('unregistered Smoke seed')
    if directory.exists():raise FileExistsError(directory)
    if directory.is_relative_to(frozen):raise ValueError('frozen calibration is read-only')
    verify(frozen,'checksums.sha256')
    import yaml
    resolved=yaml.safe_load((frozen/'resolved_config.yaml').read_text())
    if resolved['status']!='selected' or resolved['provenance']['formal_seeds']!=[1,2,3]:raise ValueError('missing frozen selection')
    originals=[OmegaConf.load(frozen/f'arm_{a}.yaml') for a in LABELS]
    validate_arm_set(originals)
    for c in originals:validate_arm_manifest(c)
    expected=dict(alpha=.5,replay_temperature=.5,freshness_window=30,min_resident=4,diagnostic_interval=10)
    for c in originals[1:]:
        if any(getattr(c,k)!=v for k,v in expected.items()):raise ValueError('frozen parameters differ')
    configs=[]
    for c in originals:
        c=OmegaConf.create(OmegaConf.to_container(c,resolve=True))
        if c.env.env_name!='swimmer' or c.env.max_episode_steps!=1000 or c.warmup_iters!=10:raise ValueError('unexpected formal environment/warmup')
        c.seed=seed;c.total_episodes=716;c.rl_updates_frac=.001
        c.eval_interval=10;c.eval_episodes=2;c.num_eval_envs=2;c.save_replay_buffer=False
        c.workflow_cls='evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow.SmokeWorkflow'
        c.arm_manifest_path=str(directory/'arm_manifest.json')
        configs.append(c)
    validate_arm_set(configs)
    directory.mkdir(parents=True,exist_ok=False)
    for a,c in zip(LABELS,configs):OmegaConf.save(c,directory/f'arm_{a}.yaml',resolve=True)
    prepare_arm_manifest(configs,directory/'arm_manifest.json')
    settings={initialization_settings_fingerprint(c) for c in configs}
    if len(settings)!=1:raise ValueError('shared initialization settings mismatch')
    manifest=dict(schema=2,kind='smoke-preflight',commit=commit,seed=seed,environment='swimmer',steps=STEPS,
        fairness_policy=FAIRNESS_POLICY,initialization_settings_sha256=settings.pop(),
        configs={a:f'arm_{a}.yaml' for a in LABELS},frozen_directory=str(frozen),
        frozen_checksums_sha256=sha(frozen/'checksums.sha256'),frozen_resolved_sha256=sha(frozen/'resolved_config.yaml'),
        source_collection_commit=resolved['analysis_provenance']['collection_commit'],source_analysis_commit=resolved['analysis_provenance']['analysis_commit'],
        budget=dict(warmup_generations=10,training_generations=12,initial_episodes=320,final_episodes=716,training_episodes=396,training_timesteps=396000,actor_updates=192,critic_updates=384,batch_size=256,evaluation_generations=[20,22],evaluation_episodes=4),
        freshness_test='actual resident ages plus read-only counterfactual W+1 neutral probe; not a claim of observed expiry',
        sampling_audit='deterministic index reconstruction outside learner JIT; immutable actual learner inputs; runtime terminal-key and optimizer-count checks; direct-trace equivalence tests')
    write(directory/'smoke_manifest.json',manifest);checksums(directory,'preflight.sha256')
    return manifest

def verify_smoke_manifest(directory):
    directory=Path(directory);verify(directory,'preflight.sha256')
    m=json.loads((directory/'smoke_manifest.json').read_text())
    if m.get('schema')!=2 or m['steps']!=12 or m['seed'] not in [1,2,3] or m['environment']!='swimmer' or set(m['configs'])!=set(LABELS):raise ValueError('Smoke matrix mismatch')
    frozen=Path(m['frozen_directory']);verify(frozen,'checksums.sha256')
    if sha(frozen/'checksums.sha256')!=m['frozen_checksums_sha256'] or sha(frozen/'resolved_config.yaml')!=m['frozen_resolved_sha256']:raise ValueError('frozen source changed')
    configs=[OmegaConf.load(directory/m['configs'][a]) for a in LABELS];validate_arm_set(configs)
    for c in configs:
        validate_arm_manifest(c)
        if c.seed!=m['seed'] or c.total_episodes!=716 or c.replay_buffer_capacity!=1000000 or c.batch_size!=256 or c.rl_updates_frac!=.001:raise ValueError('Smoke budget mismatch')
    if 'fairness_policy' in m:
        if m['fairness_policy']!=FAIRNESS_POLICY or any(initialization_settings_fingerprint(c)!=m.get('initialization_settings_sha256') for c in configs):raise ValueError('shared initialization settings mismatch')
    return m

def validate_reports(reports,*,require_all=True):
    if (require_all and set(reports)!=set(LABELS)) or not reports or not set(reports)<=set(LABELS):raise ValueError('missing arm')
    expected=dict(configured_total_episodes=716,environment_episodes=396,environment_timesteps=396000,rl_updates=192,batch_size=256,ea_ask=12,ea_tell=12,evaluation_checkpoints=[20,22],evaluation_episodes=4)
    baseline=None;common_initialization=None;common_seed=None
    for a in sorted(reports):
        report=reports[a]
        initial_hash=report.get('initialization_sha256','');seed=report.get('seed')
        settings=report.get('initialization_settings_sha256','')
        hexadecimal=lambda value:isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
        if not hexadecimal(initial_hash) or not hexadecimal(settings) or seed not in [1,2,3] or report.get('fairness_policy')!=FAIRNESS_POLICY:raise ValueError('missing initialization settings/seed audit')
        if common_initialization is None:common_initialization=settings;common_seed=seed
        elif settings!=common_initialization or seed!=common_seed:raise ValueError('common initialization settings/seed mismatch')
        if not report.get('complete') or any(k not in report.get('budget',{}) for k in expected):raise ValueError('missing budget audit')
        budget={k:report['budget'][k] for k in expected}
        if budget!=expected or (baseline is not None and budget!=baseline):raise ValueError('actual budget mismatch')
        kind=dict(A='uniform',B='fitness',C='td',D='utility')[a]
        if report['budget'].get('arm')!=kind or report['budget'].get('score_kind')!=kind:raise ValueError('actual arm identity mismatch')
        baseline=budget
        if report.get('initial')!=dict(episodes=320,timesteps=320000,generation=10) or report.get('final')!=dict(episodes=716,timesteps=716000,generation=22):raise ValueError('actual total budget mismatch')
        rows=report.get('rows',[])
        if [r['generation'] for r in rows]!=list(range(11,23)):raise ValueError('missing generation audit')
        for r in rows:
            validate_ess_evidence(r)
            if 'cdf_in_training_sampling' in r and r['cdf_in_training_sampling'] is not (a!='A' and bool(r.get('distribution_prepared',False))):raise ValueError('CDF actual-use scope mismatch')
            required=['observed_batches','ea_slots','rl_slots','changed_ea','rl_changed','origin_changed','finite','probability_l1','eligible_baseline_mass','ess_fraction','reference_fingerprints','freshness_probe_neutral','sampling_probe_seconds','actor_optimizer_steps','critic_optimizer_steps','baseline_draw_sha256','fresh_fraction','expired_fraction','unscored_fraction','baseline_prng_transition_verified','intervention_terminal_key_verified','sampling_observation']
            if any(k not in r for k in required):raise ValueError('missing sampling audit')
            if not r['finite'] or r['rl_changed'] or r['origin_changed'] or not r['freshness_probe_neutral']:raise ValueError('sampling invariant failure')
            if r['sampling_observation']!='deterministic_reconstruction' or r['baseline_prng_transition_verified'] is not True or r['intervention_terminal_key_verified'] is not True:raise ValueError('within-run PRNG/reconstruction verification failed')
            if r['observed_batches']!=32 or r['ea_slots']+r['rl_slots']!=8192:raise ValueError('actual sampling budget mismatch')
            if r['actor_optimizer_steps']!=16 or r['critic_optimizer_steps']!=32:raise ValueError('actual optimizer counter mismatch')
            if len(r['baseline_draw_sha256'])!=64 or not 0<=r['changed_ea']<=r['ea_slots']:raise ValueError('invalid actual draw audit')
            if any(not 0<=r[k]<=1 for k in ['ess_fraction','eligible_baseline_mass','fresh_fraction','expired_fraction','unscored_fraction']) or abs(sum(r[k] for k in ['fresh_fraction','expired_fraction','unscored_fraction'])-1)>1e-6:raise ValueError('invalid source coverage')
            if any(isinstance(v,float) and not math.isfinite(v) for v in r.values()):raise ValueError('nonfinite audit')
            if a=='A' and (r['changed_ea'] or r['probability_l1']):raise ValueError('baseline intervention')
        # Feedback may change Replay contents, realized origins and draw hashes.
        # Fairness is common initialization/budget plus strict within-run checks.
        if a!='A':
            if not any(r['probability_l1']>1e-8 and r['eligible_baseline_mass']>0 and r['changed_ea']>0 for r in rows if r['generation']>=20):raise ValueError('no observable weighted intervention')
            if not any(r['reference_fingerprints'] for r in rows if r['generation']==20):raise ValueError('missing pre-exposure reference audit')
            checkpoint=next(r for r in rows if r['generation']==20)
            if not (checkpoint.get('valid_score_source_count',0)>0 and checkpoint.get('distribution_prepared') is True and checkpoint.get('eligible_source_count',0)>0 and checkpoint['probability_l1']>1e-8 and checkpoint['changed_ea']>0):raise ValueError('checkpoint has no valid actual weighted intervention')
    return dict(status='PASS' if require_all else 'PASS_SINGLE_ARM',arms=sorted(reports),budget=baseline,actual_runtime_budget_equal=require_all,fairness_policy=FAIRNESS_POLICY,initialization_settings_sha256=common_initialization,postwarmup_state_sha256={a:r['initialization_sha256'] for a,r in reports.items()},postwarmup_bitwise_equality_required=False)

def validate_smoke_outputs(directory,arm=None):
    directory=Path(directory);manifest=verify_smoke_manifest(directory);reports={}
    for a in LABELS if arm is None else [arm]:
        p=directory/a/'report.json'
        if not p.is_file():raise ValueError(f'missing arm output {a}')
        verify(p.parent,'checksums.sha256');reports[a]=json.loads(p.read_text())
        actual=OmegaConf.load(p.parent/'resolved_config.yaml')
        expected=OmegaConf.load(directory/manifest['configs'][a])
        if OmegaConf.to_container(actual,resolve=True)!=OmegaConf.to_container(expected,resolve=True):raise ValueError('actual run config differs from registered config')
        if reports[a].get('initialization_settings_sha256')!=manifest.get('initialization_settings_sha256') or initialization_settings_fingerprint(actual)!=manifest.get('initialization_settings_sha256'):raise ValueError('actual initialization settings differ from manifest')
    return validate_reports(reports,require_all=arm is None)
