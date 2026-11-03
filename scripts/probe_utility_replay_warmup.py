"""Isolated Host-staged warmup evidence. Not the original fused setup scan.

Calls the original ask/rollout/tell implementations, exposing Host boundaries;
does not change their code, insert callbacks, or call TD3 learning/evaluation.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import jax,jax.numpy as jnp
from omegaconf import OmegaConf
from evorl.types import State,PyTreeDict
from evorl.algorithms.erl.erl_workflow import ERLWorkflowBase
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import erl_replace_td3_actor_params
from evorl.algorithms.erl.erl_td3.initialization_evidence import capture_evidence
from evorl.algorithms.erl.erl_td3.utility_replay_smoke import verify_smoke_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke-directory',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['pilot','original'],required=True);p.add_argument('--generations',type=int,default=1)
    a=p.parse_args();root=Path(a.output).resolve()
    allowed=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/diagnostics')
    if not root.is_relative_to(allowed) or root.exists():raise ValueError('exclusive diagnostic directory required')
    if not 1<=a.generations<=10:raise ValueError('bounded warmup only')
    source=Path(a.smoke_directory);manifest=verify_smoke_manifest(source)
    c=OmegaConf.load(source/manifest['configs']['A']);head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head==manifest['commit'] and c.random_timesteps==0 and jax.default_backend()=='gpu'
    if a.kind=='pilot':
        from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import SmokeWorkflow as Workflow
    else:
        from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow as Workflow
    w=type('Diagnostic'+a.kind,(Workflow,),{}).build_from_config(c,enable_jit=True)
    root.mkdir(parents=True,exist_ok=False)
    meta=dict(kind=a.kind,path='Host-staged diagnostic; not fused setup scan',head=head,config_sha256=hashlib.sha256((source/manifest['configs']['A']).read_bytes()).hexdigest(),backend=jax.default_backend(),jax=jax.__version__)
    def save(name,state):
        evidence=capture_evidence(state,root/name,metadata=meta)
        print(json.dumps(dict(stage=name,sha256=evidence['legacy_sha256'])),flush=True)
    state=ERLWorkflowBase.setup(w,jax.random.PRNGKey(int(c.seed)));save('g00-input',state)
    ask=jax.jit(w.ec_optimizer.ask);tell=jax.jit(w.ec_optimizer.tell)
    fitness_fn=jax.jit(lambda metrics:metrics.episode_returns.mean(axis=-1))
    try:
        for g in range(1,a.generations+1):
            key,rollout_key=jax.random.split(state.key,2)
            pop,ec=ask(state.ec_opt_state)
            save(f'g{g:02d}-ask',State(replay_buffer_state=state.replay_buffer_state,pop=pop,ec_opt_state=ec,key=key,rollout_key=rollout_key))
            agents=erl_replace_td3_actor_params(state.agent_state,pop)
            if a.kind=='pilot':w._provenance_generation=jnp.uint32(g)
            try:metrics,trajectory,replay=w._ec_rollout(agents,state.replay_buffer_state,rollout_key)
            finally:
                if a.kind=='pilot':del w._provenance_generation
            fitness=fitness_fn(metrics)
            if g==1:
                fn=type(w)._baseline_ec_rollout if a.kind=='pilot' else type(w)._ec_rollout
                lowered=fn.lower(w,agents,state.replay_buffer_state,rollout_key)
                (root/'rollout_lowered_hlo.txt').write_text(lowered.compiler_ir(dialect='hlo').as_hlo_text())
                (root/'rollout_compiled_hlo.txt').write_text(lowered.compile().as_text())
            save(f'g{g:02d}-rollout',State(replay_buffer_state=replay,trajectory=trajectory,fitness=fitness,metrics=metrics,key=key,rollout_key=rollout_key))
            ec_metrics,ec_after=tell(ec,fitness)
            save(f'g{g:02d}-tell',State(replay_buffer_state=replay,ec_opt_state=ec_after,fitness=fitness,tell_input_key=ec.key,key=key))
            state=state.replace(key=key,ec_opt_state=ec_after,replay_buffer_state=replay,metrics=state.metrics.replace(iterations=state.metrics.iterations+1,sampled_timesteps=state.metrics.sampled_timesteps+metrics.episode_lengths.sum().astype(jnp.uint32),sampled_episodes=state.metrics.sampled_episodes+jnp.uint32(c.episodes_for_fitness*c.pop_size)))
            save(f'g{g:02d}-output',state)
    finally:w.close()

if __name__=='__main__':main()
