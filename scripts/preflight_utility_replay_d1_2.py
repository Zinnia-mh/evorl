"""Bounded D1-2 admission: GPU old/new equivalence, ages, two real steps."""
import argparse,json,time,sys,types,subprocess,resource,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax,jax.numpy as jnp,numpy as np
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.utility_replay_d1 import ERLUtilityReplayD1Workflow,D1CheckpointStore,build_d1_distribution,priority_value
from evorl.algorithms.erl.erl_td3.d1_residency import recount_and_map
from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table,compute_source_eligibility
from evorl.algorithms.erl.erl_td3.erl_utility_replay import init_utility_replay_state,weighted_replay_sample,_optimizer_step_count
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import restore_template
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import reconstruct_replay_draws,DrawRecorder,initialization_fingerprint
from evorl.algorithms.erl.erl_td3.replay_distribution_audit import audit_distribution

def reference():
 p='evorl/algorithms/erl/erl_td3/utility_replay_d1.py';source=subprocess.check_output(['git','-C',str(ROOT),'show','9441b986ed4da411a5fa299a1cdd8600f585a962:'+p],text=True)
 m=types.ModuleType('evorl.algorithms.erl.erl_td3.d1_9441_reference');m.__package__='evorl.algorithms.erl.erl_td3';exec(compile(source,p,'exec'),m.__dict__);return m
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def timed(fn,n=3):
 start=time.perf_counter();value=fn();jax.block_until_ready(value);cold=time.perf_counter()-start;warm=[]
 for _ in range(n):
  start=time.perf_counter();value=fn();jax.block_until_ready(value);warm.append(time.perf_counter()-start)
 return value,dict(first_seconds=cold,warm_seconds=warm)
def same(a,b,exact=True):
 for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
  if exact:np.testing.assert_array_equal(x,y)
  else:np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)

def run(O,scale_only=False):
 assert jax.default_backend()=='gpu';O.mkdir(parents=True,exist_ok=False);old=reference()
 c=OmegaConf.load('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/formal/swimmer/abc-baseline-v1-head228a301/protocols/seed_1/arm_D.yaml');c.seed=2027;c.alpha=.3;c.rho=.3;c.utility_variant='d1-v1';c.half_life=15.;c.priority_cap=.1;c.utility_scale=.01;c.diagnostic_interval=1;c.workflow_cls='evorl.algorithms.erl.erl_td3.utility_replay_d1.ERLUtilityReplayD1Workflow'
 w=ERLUtilityReplayD1Workflow.build_from_config(c,enable_jit=True);abstract=restore_template(w,abstract=False);rb=abstract.replay_buffer_state;C=1000000;source_count=1024;ea_size=source_count*960;mask=jnp.arange(C)<ea_size;tokens=jnp.where(mask,jnp.arange(C)//960,-1).astype(jnp.int32)
 rb=rb.replace(buffer_size=jnp.int32(C),current_index=jnp.int32(0),origin_type=jnp.where(mask,0,1).astype(jnp.int8),source_token=tokens)
 t=init_source_score_table(C);active=jnp.arange(C)<source_count;t=t._replace(source_token=jnp.where(active,jnp.arange(C),-1),resident_count=jnp.where(active,960,0),validity=active,score_kind=jnp.full(C,3,jnp.int8),measured_generation=jnp.where(active,100-jnp.arange(C)//32,-1),version=jnp.where(active,jnp.arange(C)//32,-1),current_version=jnp.int32(31),rank_normalized_score=jnp.where(active,.1+.8*(jnp.arange(C)%32)/31,0),raw_score=jnp.where(active,jnp.sin(jnp.arange(C))*.01,0))
 (rec,slots),recount_time=timed(lambda:recount_and_map(t,tokens,mask));t=compute_source_eligibility(rec.table,100,30,4).table
 (a,ma),oldtime=timed(lambda:old.build_d1_distribution(t,tokens,mask,100))
 (b,mb),newtime=timed(lambda:build_d1_distribution(t,tokens,mask,100,source_slots=slots));same(a,b,False)
 sample=jax.jit(lambda r,k,d:weighted_replay_sample(r,k,jax.random.PRNGKey(2027),d,True,256))
 for key in range(20):
  x=sample(rb,jax.random.PRNGKey(key),a);y=sample(rb,jax.random.PRNGKey(key),b);same(x,y)
  np.testing.assert_array_equal(y.final_indices[y.rl_slot_mask],y.baseline_indices[y.rl_slot_mask]);assert np.all(np.asarray(rb.origin_type[y.final_indices[y.ea_slot_mask]])==0)
 audit=audit_distribution(*jax.device_get((b.source_probability,t.resident_count,b.transition_probability,b.transition_cdf,mask)),float(b.ess_fraction),cdf_in_training_sampling=True)
 age_rows=[];v=t._replace(measured_generation=jnp.where(active,100,-1),weighted_eligible=active)
 first,m0=build_d1_distribution(v,tokens,mask,100,source_slots=slots)
 for age in range(32):
  aged=compute_source_eligibility(v,100+age,30,4).table
  d,m=build_d1_distribution(aged,tokens,mask,100+age,source_slots=slots);jax.block_until_ready((d,m))
  if age<31:
   np.testing.assert_allclose(d.source_probability-d.baseline_source_probability,(first.source_probability-first.baseline_source_probability)*2**(-age/15),atol=1e-7);np.testing.assert_allclose(m,m0*2**(-age/15),rtol=1e-6)
  else:same(d.source_probability,d.baseline_source_probability);assert float(m)==0
  if age in [0,15,30,31]:
   ref,mref=old.build_d1_distribution(aged,tokens,mask,100+age);same(ref,d,False)
   for key in range(8):same(sample(rb,jax.random.PRNGKey(key+50),ref),sample(rb,jax.random.PRNGKey(key+50),d))
  age_rows.append(dict(age=age,priority_mass=float(m),l1=float(jnp.abs(d.source_probability-d.baseline_source_probability).sum()),ess=float(d.ess_fraction)))
 _,sampletime=timed(lambda:sample(rb,jax.random.PRNGKey(2027),b),n=5)
 write(O/'scale_age_equivalence.json',dict(status='PASS',capacity=C,eligible_mixed=int(b.eligible_source_count),eligible_mass=float(b.eligible_baseline_mass),sampler=sampletime,mixed_age_audit=audit,old_d1_build=oldtime,new_d1_build=newtime,recount_and_mapping=recount_time,age_rows=age_rows,indices_identical=True,rng_identical=True))
 print('FORMAL_SHAPE_EQ_AND_AGES_PASS',flush=True);w.close()
 if scale_only:return
 # Real trajectories: shortest episodes that supply formal source_batch_size256.
 c.env.max_episode_steps=256;c.warmup_iters=1;c.output_dir=str(O/'real');OmegaConf.save(c,O/'real_config.yaml',resolve=True)
 refworkflow=old.ERLUtilityReplayD1Workflow.build_from_config(c,enable_jit=True)
 rows=[];reference_checked=False
 class Observed(ERLUtilityReplayD1Workflow):
  def _weighted_rl_update(self,agent,opt,replay,key,updates,u):
   nonlocal reference_checked
   baseline=None
   if not reference_checked:
    refworkflow._provenance_generation=int(self._provenance_generation);refworkflow._latest_score_lifecycle_audit={}
    baseline=refworkflow._weighted_rl_update(agent,opt,replay,key,updates,u);jax.block_until_ready(baseline)
   result=super()._weighted_rl_update(agent,opt,replay,key,updates,u);jax.block_until_ready(result)
   if baseline is not None:
    same(baseline,result);reference_checked=True
   return result
 workflow=Observed.build_from_config(c,enable_jit=True);state=workflow.init(jax.random.PRNGKey(2027))
 try:
  for step in range(2):
   before=state;start=time.perf_counter();metrics,state=workflow.step(state);jax.block_until_ready((metrics,state));elapsed=time.perf_counter()-start;g=int(state.metrics.iterations)
   records=workflow._latest_diagnostic_records;assert len(records)==32 and all(r.eligible and np.isfinite(r.utility) and np.isfinite(r.loss_before) for r in records)
   replay,key,updates,u=workflow._latest_d1_inputs;trace=reconstruct_replay_draws(replay,key,u,num_updates=int(updates),num_draws=2,batch_size=256,uniform=False);jax.block_until_ready(trace);same(trace.terminal_intervention_key,state.utility_replay_state.intervention_key);same(state.key,jax.random.split(before.key,4)[0]);recorder=DrawRecorder();recorder.record(*jax.device_get(trace[:5]));draw=recorder.summary();assert draw['rl_changed']==draw['origin_changed']==0
   assert int(updates)==4096;assert _optimizer_step_count(state.opt_state.actor)-_optimizer_step_count(before.opt_state.actor)==4096;assert _optimizer_step_count(state.opt_state.critic)-_optimizer_step_count(before.opt_state.critic)==8192
   score=[dict(U=r.utility,L_before=r.loss_before,r=r.utility/max(abs(r.loss_before),1e-8),v=priority_value(r.utility,r.loss_before),generation=r.generation,individual=r.individual_id) for r in records];assert np.ptp([r['v'] for r in score])>0
   row=dict(workflow._latest_d1_metrics,**draw,step_seconds=elapsed,score=score,diagnostic_seconds=workflow._latest_score_lifecycle_audit['diagnostic_seconds'],source_seconds=workflow._latest_score_lifecycle_audit['resident_count_seconds'],learner_seconds=workflow._latest_score_lifecycle_audit['weighted_learner_phase_seconds'],distribution_build_count=workflow._latest_score_lifecycle_audit['distribution_build_count']);rows.append(row);write(O/f'real_g{g}.json',row)
   if step==0:
    with D1CheckpointStore(O/'checkpoint',c) as store:store.save(state,dict(generation=g),force=True)
    with D1CheckpointStore(O/'checkpoint',c,read_only=True) as store:restored,runtime=store.restore(g,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),state))
    assert initialization_fingerprint(restored)==initialization_fingerprint(state);state=restored
  assert reference_checked and sum(x['changed_ea'] for x in rows)>0
  write(O/'report.json',dict(status='PASS',seed=2027,backend=jax.default_backend(),old_new_real_td3_results_bitwise_equal=True,checkpoint_resume=True,episodes=int(state.metrics.sampled_episodes),steps=int(state.metrics.sampled_timesteps),gpu_memory=jax.devices()[0].memory_stats(),host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows,scope='two real post-warmup generations; first step includes an extra discarded old-reference learner validation; no Return-based decision'))
  print('REAL_FORMAL_NETWORK_AND_RESUME_PASS',flush=True)
 finally:workflow.close();refworkflow.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--scale-only',action='store_true');args=p.parse_args();run(Path(args.directory),scale_only=args.scale_only)
