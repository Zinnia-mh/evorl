"""Explicit prepare/run-arm/validate commands; never launches a matrix implicitly."""
import argparse,csv,json,subprocess,time
from pathlib import Path
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.utility_replay_smoke import (prepare_smoke,verify_smoke_manifest,validate_smoke_outputs,initialization_settings_fingerprint,FAIRNESS_POLICY,write,checksums)

ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/smoke')
def head():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def clean():
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Smoke requires clean committed code')
def run_arm(directory,arm):
    import jax
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import SmokeWorkflow,initialization_fingerprint
    from evorl.algorithms.erl.erl_td3.utility_replay_budget import BudgetAudit,validate_arm_manifest
    clean();directory=Path(directory);m=verify_smoke_manifest(directory)
    if m['commit']!=head():raise ValueError('Smoke code version changed')
    if jax.default_backend()!='gpu':raise ValueError('actual Smoke requires GPU')
    c=OmegaConf.load(directory/m['configs'][arm]);validate_arm_manifest(c)
    if m.get('fairness_policy')!=FAIRNESS_POLICY or initialization_settings_fingerprint(c)!=m.get('initialization_settings_sha256'):raise ValueError('missing/mismatched preregistered initialization settings')
    output=directory/arm;output.mkdir(exist_ok=False)
    (output/'audit').mkdir();(output/'metrics').mkdir()
    write(output/'manifest.json',dict(smoke_manifest=str(directory/'smoke_manifest.json'),arm=arm,commit=head()))
    OmegaConf.save(c,output/'resolved_config.yaml',resolve=True)
    workflow=type(f'SmokeArm{arm}',(SmokeWorkflow,),{}).build_from_config(c,enable_jit=True)
    workflow._smoke_audit_directory=output/'audit'
    actual_settings_hash=initialization_settings_fingerprint(workflow.config)
    if actual_settings_hash!=m['initialization_settings_sha256']:raise ValueError('workflow resolved settings mismatch')
    def counters(state):return dict(episodes=int(state.metrics.sampled_episodes),timesteps=int(state.metrics.sampled_timesteps),generation=int(state.metrics.iterations))
    rows=[];evaluations=[];start=time.perf_counter()
    try:
        state=workflow.setup(jax.random.PRNGKey(int(c.seed)));initial=counters(state)
        initial_hash=initialization_fingerprint(state)
        write(output/'audit/initialization.json',dict(counters=initial,state_sha256=initial_hash,initialization_settings_sha256=actual_settings_hash,seed=int(c.seed)))
        audit=BudgetAudit(c,initial_episodes=initial['episodes'],initial_iteration=initial['generation'],initial_timesteps=initial['timesteps'])
        workflow._budget_audit=audit;workflow._observed_ec_rollouts=0
        with (output/'progress.jsonl').open('x') as log:
            for _ in range(m['steps']):
                metrics,state=workflow.step(state);row=workflow._latest_smoke_audit;rows.append(row)
                if not row['finite'] or row['rl_changed'] or row['origin_changed'] or not row['freshness_probe_neutral']:raise ValueError('HOLD actual sampling invariant failure')
                g=int(state.metrics.iterations)
                if g%int(c.eval_interval)==0 or g==22:
                    evaluation,state=workflow.evaluate(state);jax.block_until_ready((evaluation,state))
                    evaluations.append(dict(generation=g,episode_returns=jax.device_get(evaluation.rl_episode_returns).tolist(),episode_lengths=jax.device_get(evaluation.rl_episode_lengths).tolist()))
                log.write(json.dumps(dict(generation=g,completed=len(rows),steps=m['steps']),sort_keys=True)+'\n');log.flush()
        final=counters(state)
        budget=audit.finalize(final_episodes=final['episodes'],final_iteration=final['generation'],final_timesteps=final['timesteps'])
        report=dict(complete=True,seed=int(c.seed),initialization_sha256=initial_hash,initialization_settings_sha256=actual_settings_hash,fairness_policy=FAIRNESS_POLICY,initial=initial,final=final,budget=budget,rows=rows,elapsed_seconds=time.perf_counter()-start)
        write(output/'report.json',report);write(output/'audit/budget_counters.json',budget)
        write(output/'audit/reference_fingerprints.json',[dict(generation=r['generation'],fingerprints=r['reference_fingerprints'],scope=r['eval_reference_check']) for r in rows])
        with (output/'audit/source_sampling.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(output/'metrics/evaluation.json',evaluations);checksums(output,'checksums.sha256')
        print(json.dumps(dict(status='COMPLETE_PENDING_FOUR_ARM_VALIDATION',arm=arm,output=str(output))))
    finally:workflow.close()

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--frozen',required=True);p.add_argument('--directory',required=True);p.add_argument('--seed',type=int,choices=[1,2,3],default=1)
    p=sub.add_parser('run-arm');p.add_argument('--directory',required=True);p.add_argument('--arm',choices=list('ABCD'),required=True)
    p=sub.add_parser('validate');p.add_argument('--directory',required=True);p.add_argument('--arm',choices=list('ABCD'))
    args=parser.parse_args();directory=Path(args.directory).resolve()
    if not directory.is_relative_to(ROOT):raise ValueError('Smoke artifacts must remain under external Smoke root')
    if args.command=='prepare':
        clean();print(json.dumps(prepare_smoke(args.frozen,directory,seed=args.seed,commit=head()),sort_keys=True))
    elif args.command=='run-arm':run_arm(directory,args.arm)
    else:
        result=validate_smoke_outputs(directory,arm=args.arm);(directory/'aggregate').mkdir(exist_ok=True)
        write(directory/'aggregate'/('smoke_validation.json' if args.arm is None else f'arm_{args.arm}_validation.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
