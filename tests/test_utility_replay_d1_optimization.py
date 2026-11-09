import importlib.util,subprocess,types
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from evorl.algorithms.erl.erl_td3.utility_replay_d1 import build_d1_distribution
from evorl.algorithms.erl.erl_td3.d1_residency import recount_and_map
from evorl.algorithms.erl.erl_td3.source_residency_batch import recount_resident_sources
from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table
from evorl.algorithms.erl.erl_td3.erl_utility_replay import weighted_replay_sample

def reference():
 root=Path(__file__).resolve().parents[1];p='evorl/algorithms/erl/erl_td3/utility_replay_d1.py'
 code=subprocess.check_output(['git','-C',str(root),'show','9441b986ed4da411a5fa299a1cdd8600f585a962:'+p],text=True)
 m=types.ModuleType('evorl.algorithms.erl.erl_td3.reference_d1_9441');m.__package__='evorl.algorithms.erl.erl_td3';exec(compile(code,p,'exec'),m.__dict__);return m

def test_recount_and_mapping_exact_under_repeated_physical_wraps():
 rng=np.random.default_rng(2027);t=init_source_score_table(64)
 for generation in range(1,15):
  tokens=rng.integers(-1,22,256,dtype=np.int32);mask=tokens>=0
  old=recount_resident_sources(t,jnp.asarray(tokens),jnp.asarray(mask));(new,slots)=recount_and_map(t,jnp.asarray(tokens),jnp.asarray(mask))
  for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(new)):np.testing.assert_array_equal(a,b)
  np.testing.assert_array_equal(np.asarray(new.table.source_token)[np.asarray(slots)[mask]],tokens[mask]);t=new.table

def test_probabilities_cdf_draws_and_rng_are_equivalent():
 ref=reference();t=init_source_score_table(64);tokens=jnp.arange(256)%17;mask=jnp.arange(256)%11!=0;tokens=jnp.where(mask,tokens,-1)
 result,slots=recount_and_map(t,tokens,mask);t=result.table;valid=t.resident_count>0
 t=t._replace(validity=valid,version=jnp.where(valid,1,-1),measured_generation=jnp.where(valid,100,-1),raw_score=jnp.arange(64)*.001,rank_normalized_score=jnp.linspace(.05,.95,64),weighted_eligible=valid)
 from evorl.replay_buffers.source_aware_replay_buffer import SourceAwareReplayBuffer
 rb=SourceAwareReplayBuffer(capacity=256,min_sample_timesteps=1,sample_batch_size=256);r=rb.init(jnp.float32(0)).replace(data=jnp.arange(256,dtype=jnp.float32),buffer_size=jnp.int32(256),source_token=tokens,origin_type=jnp.where(mask,0,1).astype(jnp.int8))
 for age in [0,1,15,30,31]:
  a,m=ref.build_d1_distribution(t,tokens,mask,100+age);b,n=build_d1_distribution(t,tokens,mask,100+age,source_slots=slots)
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
  for seed in range(8):
   x=weighted_replay_sample(r,jax.random.PRNGKey(seed),jax.random.PRNGKey(2027),a,m>0,256);y=weighted_replay_sample(r,jax.random.PRNGKey(seed),jax.random.PRNGKey(2027),b,n>0,256)
   np.testing.assert_array_equal(x.final_indices,y.final_indices);np.testing.assert_array_equal(x.next_intervention_key,y.next_intervention_key);np.testing.assert_array_equal(x.rl_slot_mask,y.rl_slot_mask)

def test_complete_age_window_preserves_decay_without_renormalizing():
 t=init_source_score_table(32)._replace(source_token=jnp.arange(32),resident_count=jnp.full(32,8),validity=jnp.ones(32,bool),version=jnp.ones(32,jnp.int32),measured_generation=jnp.full(32,100),raw_score=jnp.ones(32),rank_normalized_score=jnp.linspace(.05,.95,32),weighted_eligible=jnp.ones(32,bool))
 tokens=jnp.arange(256)//8;mask=jnp.ones(256,bool);first,m0=build_d1_distribution(t,tokens,mask,100)
 for age in range(32):
  d,m=build_d1_distribution(t,tokens,mask,100+age)
  if age<=30:
   np.testing.assert_allclose(d.source_probability-d.baseline_source_probability,(first.source_probability-first.baseline_source_probability)*2**(-age/15),atol=1e-7)
   np.testing.assert_allclose(m,m0*2**(-age/15),rtol=1e-6)
  else:np.testing.assert_array_equal(d.source_probability,d.baseline_source_probability);assert float(m)==0

def test_d1_step_is_structurally_the_frozen_step_with_one_refresh_hook():
 import ast,inspect,textwrap
 from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
 from evorl.algorithms.erl.erl_td3.utility_replay_d1 import ERLUtilityReplayD1Workflow
 a=textwrap.dedent(inspect.getsource(ERLUtilityReplayWorkflow._weighted_step)).replace('refresh_replay_distribution(', 'self._refresh_d1_distribution(')
 b=textwrap.dedent(inspect.getsource(ERLUtilityReplayD1Workflow._weighted_step))
 assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))

def test_scored_sources_survive_first_physical_wrap_with_identical_metadata():
 p=Path(__file__).with_name('test_utility_replay_ring_scores.py');spec=importlib.util.spec_from_file_location('ring_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 _,rb,u=m.wrapped();mask=rb.origin_type==0;expected=recount_resident_sources(u.score_table,rb.source_token,mask);actual,slots=recount_and_map(u.score_table,rb.source_token,mask)
 for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(x,y)
 assert bool(actual.table.validity[0]) and bool(actual.table.validity[1])
