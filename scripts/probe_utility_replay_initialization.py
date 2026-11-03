"""Independent initialization evidence; never calls step/learn/evaluate."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from omegaconf import OmegaConf

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('probe');p.add_argument('--smoke-directory',required=True);p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--label',required=True);p.add_argument('--stage',choices=['zero','warmup'],required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('compare');p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    from evorl.algorithms.erl.erl_td3.initialization_evidence import capture_evidence,compare_evidence
    if args.command=='compare':
        result=compare_evidence(args.left,args.right)
        with Path(args.output).open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
        print(json.dumps(result,sort_keys=True));return
    import jax,jaxlib
    from evorl.algorithms.erl.erl_workflow import ERLWorkflowBase
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke import verify_smoke_manifest
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import SmokeWorkflow
    directory=Path(args.smoke_directory);m=verify_smoke_manifest(directory)
    if jax.default_backend()!='gpu':raise ValueError('original GPU backend required for initialization evidence')
    config=OmegaConf.load(directory/m['configs'][args.arm]);assert config.random_timesteps==0
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert head==m['commit']
    output=Path(args.output).resolve()
    allowed=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/diagnostics')
    if not output.is_relative_to(allowed):raise ValueError('independent diagnostic output required')
    if output.exists():raise FileExistsError(output)
    original=json.loads((directory/args.arm/'report.json').read_text())
    meta=dict(label=args.label,arm=args.arm,stage=args.stage,head=head,config_sha256=hashlib.sha256((directory/m['configs'][args.arm]).read_bytes()).hexdigest(),python=sys.executable,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),jit_disabled=bool(jax.config.jax_disable_jit),original_run_legacy_sha256=original['initialization_sha256'])
    workflow=type('SmokeArm'+args.arm,(SmokeWorkflow,),{}).build_from_config(config,enable_jit=True)
    start=time.perf_counter();seed=jax.random.PRNGKey(int(config.seed))
    try:
        if args.stage=='zero':
            state=ERLWorkflowBase.setup(workflow,seed)
            assert int(state.metrics.iterations)==0 and int(state.metrics.sampled_timesteps)==0
            meta.update(environment_steps=0,pilot_state='not yet created by real setup path')
        else:
            # Capture the original core setup return at a Host boundary. The
            # original warmup scan/function is neither replaced nor instrumented.
            original_setup=ERLWorkflowBase.setup
            def capture_core(self,key):
                state=original_setup(self,key)
                capture_evidence(state,output.with_name(output.name+'-pre'),metadata=dict(meta,stage='pre_warmup'))
                return state
            ERLWorkflowBase.setup=capture_core
            try:state=workflow.setup(seed)
            finally:ERLWorkflowBase.setup=original_setup
            assert int(state.metrics.iterations)==10 and int(state.metrics.sampled_episodes)==320 and int(state.metrics.sampled_timesteps)==320000
            meta.update(environment_steps=320000,rl_optimizer_updates=0)
        result=capture_evidence(state,output,metadata=meta)
        print(json.dumps(dict(label=args.label,stage=args.stage,legacy_sha256=result['legacy_sha256'],canonical_sha256=result['canonical_sha256'],matches_original_run=(result['legacy_sha256']==original['initialization_sha256']) if args.stage=='warmup' else None,leaf_count=len(result['leaves']),environment_steps=meta['environment_steps'],elapsed_seconds=time.perf_counter()-start,output=str(output)),sort_keys=True),flush=True)
    finally:workflow.close()
if __name__=='__main__':main()
