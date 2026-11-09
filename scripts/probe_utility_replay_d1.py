"""Bounded real JIT mechanism validation; no formal/performance experiment."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import argparse,json,time,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax,jax.numpy as jnp,numpy as np
from hydra import initialize_config_dir,compose
from evorl.algorithms.erl.erl_td3.utility_replay_d1 import ERLUtilityReplayD1Workflow,D1CheckpointStore
from evorl.algorithms.erl.erl_td3.erl_utility_replay import _optimizer_step_count,weighted_replay_sample,alpha_zero_sample
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import reconstruct_replay_draws,DrawRecorder,initialization_fingerprint
from evorl.replay_buffers.replay_buffer import ReplayBuffer

def run(directory):
 directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
 overrides=['agent=erl/erl-utility-replay-d1','env=brax/swimmer','env.max_episode_steps=8','num_envs=1','pop_size=32','num_elites=4','num_rl_agents=1','episodes_for_fitness=1','rollout_episodes=1','batch_size=16','replay_buffer_capacity=512','warmup_iters=1','actor_update_interval=2','rl_updates_frac=1.0','agent_network.critic_hidden_layer_sizes=[8,8]','agent_network.actor_hidden_layer_sizes=[8,8]','num_eval_envs=1','eval_episodes=1','source_batch_size=4','seed=2027']
 with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):c=compose(config_name='config',overrides=overrides)
 from omegaconf import OmegaConf
 OmegaConf.save(c,directory/'resolved_config.yaml',resolve=True)
 workflow=ERLUtilityReplayD1Workflow.build_from_config(c,enable_jit=True);rows=[];state=workflow.init(jax.random.PRNGKey(2027));cp_verified=False
 try:
  for _ in range(3):
   before=state;start=time.perf_counter();metrics,state=workflow.step(state);jax.block_until_ready((metrics,state));step_seconds=time.perf_counter()-start
   replay,key,updates,u=workflow._latest_d1_inputs;g=int(state.metrics.iterations);records=workflow._latest_diagnostic_records
   assert len(records)==32 and all(r.eligible and r.utility is not None and r.generation==g and r.measurement_phase=='pre_exposure' for r in records)
   assert len({r.reference_fingerprint for r in records})==1
   assert workflow._latest_score_lifecycle_audit['ordering'].index('diagnostic_complete')<workflow._latest_score_lifecycle_audit['ordering'].index('ea_insertion_complete')
   np.testing.assert_array_equal(state.key,jax.random.split(before.key,4)[0])
   actor=_optimizer_step_count(state.opt_state.actor)-_optimizer_step_count(before.opt_state.actor);critic=_optimizer_step_count(state.opt_state.critic)-_optimizer_step_count(before.opt_state.critic)
   assert actor==int(updates)==128 and critic==256
   trace=reconstruct_replay_draws(replay,key,u,num_updates=int(updates),num_draws=2,batch_size=16,uniform=False);jax.block_until_ready(trace)
   np.testing.assert_array_equal(trace.terminal_intervention_key,state.utility_replay_state.intervention_key)
   recorder=DrawRecorder();recorder.record(*jax.device_get(trace[:5]));draw=recorder.summary();assert draw['rl_changed']==draw['origin_changed']==0 and draw['changed_ea']>0
   table=state.utility_replay_state.score_table;tokens,origin,n=jax.device_get((state.replay_buffer_state.source_token,state.replay_buffer_state.origin_type,state.replay_buffer_state.buffer_size));resident=(np.arange(len(tokens))<int(n))&(origin==0);ids,counts=np.unique(tokens[resident],return_counts=True)
   actual=dict(zip(map(int,ids),map(int,counts)));tids,tc,raw,measured,version,valid=jax.device_get((table.source_token,table.resident_count,table.raw_score,table.measured_generation,table.version,table.validity))
   assert actual=={int(t):int(n) for t,n in zip(tids,tc) if n>0}
   for r in records:
    i=int(np.flatnonzero(tids==g*32+r.individual_id)[0]);assert int(measured[i])==g and int(version[i])==g-2;assert float(raw[i])==np.float32(r.utility)
   for t,n in actual.items():
    if t//32>=2 and n>=4:assert bool(valid[int(np.flatnonzero(tids==t)[0])])
   sampler=jax.jit(lambda r,k,v:weighted_replay_sample(r,k,jax.random.PRNGKey(77),v.distribution,v.distribution_prepared,16))
   start=time.perf_counter();jax.block_until_ready(sampler(replay,jax.random.PRNGKey(999),u));sampler_cold=time.perf_counter()-start
   times=[]
   for k in range(5):
    start=time.perf_counter();jax.block_until_ready(sampler(replay,jax.random.PRNGKey(990+k),u));times.append(time.perf_counter()-start)
   uniform=alpha_zero_sample(replay,jax.random.PRNGKey(9),u.intervention_key,16);base=ReplayBuffer(capacity=512,min_sample_timesteps=1,sample_batch_size=16).sample(replay,jax.random.PRNGKey(9))
   for a,b in zip(jax.tree.leaves(uniform.batch),jax.tree.leaves(base)):np.testing.assert_array_equal(a,b)
   np.testing.assert_array_equal(uniform.intervention_key,u.intervention_key)
   row=dict(workflow._latest_d1_metrics,**draw,actor_updates=actor,critic_updates=critic,step_seconds=step_seconds,diagnostic_seconds=workflow._latest_score_lifecycle_audit['diagnostic_seconds'],source_statistics_seconds=workflow._latest_score_lifecycle_audit['resident_count_seconds'],discarded_d0_refresh_seconds=workflow._latest_score_lifecycle_audit['distribution_refresh_seconds'],sampler_cold_seconds=sampler_cold,sampler_steady_seconds=float(np.mean(times)),sampling_scope='deterministic reconstruction of actual learner inputs and verified terminal key; independent-key sampler timing, not in-training profiler')
   rows.append(row)
   with (directory/f'g{g}.json').open('x') as f:json.dump(row,f,indent=2,allow_nan=False)
   np.savez_compressed(directory/f'g{g}_draws.npz',baseline=trace.baseline_indices,final=trace.final_indices,ea=trace.ea_slot_mask,rl=trace.rl_slot_mask)
   if g==3:
    with D1CheckpointStore(directory/'checkpoint',c) as store:store.save(state,dict(generation=g),force=True)
    with D1CheckpointStore(directory/'checkpoint',c,read_only=True) as store:restored,runtime=store.restore(g,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),state))
    assert initialization_fingerprint(restored)==initialization_fingerprint(state) and runtime==dict(generation=g);state=restored;cp_verified=True
  assert cp_verified and int(state.metrics.iterations)==4
  result=dict(status='PASS',backend=jax.default_backend(),enable_jit=True,seed=2027,warmup_generations=1,training_generations=3,episodes=int(state.metrics.sampled_episodes),environment_steps=int(state.metrics.sampled_timesteps),checkpoint_roundtrip=True,resumed_step_generation=4,rows=rows,base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'evorl/algorithms/erl/erl_td3/utility_replay_d1.py',ROOT/'scripts/probe_utility_replay_d1.py',ROOT/'configs/agent/erl/erl-utility-replay-d1.yaml']})
  with (directory/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
  print(json.dumps(result,allow_nan=False),flush=True)
 finally:workflow.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);run(p.parse_args().directory)
