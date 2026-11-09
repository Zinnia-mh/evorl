import jax,jax.numpy as jnp,numpy as np,pytest
from evorl.algorithms.erl.erl_td3.utility_replay_d1 import priority_value,build_d1_distribution
from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table

def fixture(g=0):
 t=init_source_score_table(3)._replace(source_token=jnp.array([10,20,30]),resident_count=jnp.array([4,4,4]),rank_normalized_score=jnp.array([.9,.3,.5]),measured_generation=jnp.array([0,0,-1]),validity=jnp.array([True,True,False]),version=jnp.array([0,0,-1]),weighted_eligible=jnp.array([True,True,False]))
 tokens=jnp.array([10]*4+[20]*4+[30]*4+[-1]*4);mask=tokens>=0
 return t,tokens,mask
def dist(g=0,rho=.3):
 t,tokens,mask=fixture();return build_d1_distribution(t,tokens,mask,g,rho=rho)
def test_negative_utilities_are_ordered_bounded_not_positive_help():
 v=[priority_value(u,1.) for u in [-100.,-1.,-.01,0.,.01,1.,100.]]
 assert all(.049<v0<.951 for v0 in v) and all(a<b for a,b in zip(v,v[1:])) and v[2]<.5==v[3]
def test_relative_scale_is_cross_generation_invariant():
 assert priority_value(.01,1.)==pytest.approx(priority_value(.1,10.))
def test_normalization_floor_full_pool_mass():
 d,m=dist();p=np.asarray(d.source_probability);assert p.sum()==pytest.approx(1.);assert np.all(np.asarray(d.transition_probability)>=0)
 assert p[0]>1/3 and p[:2].sum()>2/3 and p[2]>0 and float(d.uniform_component_total_mass)>=.7
def test_equal_age_decay_not_cancelled_by_normalization():
 p0=np.asarray(dist(0)[0].source_probability);p15=np.asarray(dist(15)[0].source_probability);b=np.full(3,1/3)
 np.testing.assert_allclose(p15-b,(p0-b)*.5,atol=1e-7)
 assert dist(15)[1]==pytest.approx(float(dist(0)[1])*.5)
def test_individual_old_score_extra_effect_declines():
 t,tokens,mask=fixture();t=t._replace(measured_generation=jnp.array([15,15,-1]));old=t._replace(measured_generation=jnp.array([0,15,-1]));a=build_d1_distribution(t,tokens,mask,15)[0];b=build_d1_distribution(old,tokens,mask,15)[0]
 assert b.source_probability[0]<a.source_probability[0]
@pytest.mark.parametrize('age',[31,40,100])
def test_hard_window_restores_uniform(age):
 d,m=dist(age);np.testing.assert_array_equal(d.source_probability,d.baseline_source_probability);assert float(m)==0
def test_rho_zero_is_exact_baseline():
 d,m=dist(rho=0);np.testing.assert_array_equal(d.source_probability,d.baseline_source_probability);assert float(m)==0
def test_no_eligible_is_neutral():
 t,tokens,mask=fixture();t=t._replace(validity=jnp.zeros(3,bool));d,m=build_d1_distribution(t,tokens,mask,0);np.testing.assert_array_equal(d.source_probability,d.baseline_source_probability);assert float(m)==0
@pytest.mark.parametrize('kw',[dict(rho=-1),dict(rho=1.1),dict(half_life=0),dict(priority_cap=0)])
def test_invalid_parameters_rejected(kw):
 t,tokens,mask=fixture()
 with pytest.raises(ValueError):build_d1_distribution(t,tokens,mask,0,**kw)
def test_priority_cap_before_decay():
 t,tokens,mask=fixture();t=t._replace(validity=jnp.array([True,False,False]));d,m=build_d1_distribution(t,tokens,mask,0);_,later=build_d1_distribution(t,tokens,mask,15)
 assert float(m)<=.030001 and float(later)==pytest.approx(float(m)*.5)
def test_transition_probabilities_match_source_counts():
 d,m=dist();p=np.asarray(d.transition_probability);assert p.sum()==pytest.approx(1)
 np.testing.assert_allclose([p[:4].sum(),p[4:8].sum(),p[8:12].sum()],d.source_probability,atol=1e-7);assert np.all(p[12:]==0)

def test_d1_small_real_ring_and_actual_draws_preserve_rl():
 import importlib.util
 from pathlib import Path
 spec=importlib.util.spec_from_file_location('d0_ring_helpers',Path(__file__).with_name('test_utility_replay_ring_scores.py'));helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
 wrapped,refresh,assert_actual_counts=helpers.wrapped,helpers.refresh,helpers.assert_actual_counts
 from evorl.algorithms.erl.erl_td3.erl_utility_replay import weighted_replay_sample,alpha_zero_sample
 rb,replay,u=wrapped();u=refresh(u,replay);assert_actual_counts(u.score_table,replay)
 t=u.score_table._replace(rank_normalized_score=jnp.array([.9,.1,.5]));d,m=build_d1_distribution(t,replay.source_token,replay.origin_type==0,31)
 key=jax.random.PRNGKey(2027);draw=weighted_replay_sample(replay,key,u.intervention_key,d,m>0,256)
 ea=np.asarray(draw.ea_slot_mask);rl=np.asarray(draw.rl_slot_mask)
 assert np.any(np.asarray(draw.final_indices)[ea]!=np.asarray(draw.baseline_indices)[ea]);np.testing.assert_array_equal(draw.final_indices[rl],draw.baseline_indices[rl]);np.testing.assert_array_equal(draw.batch,replay.data[draw.final_indices])
 neutral=weighted_replay_sample(replay,key,u.intervention_key,d,False,256);base=alpha_zero_sample(replay,key,u.intervention_key,256)
 np.testing.assert_array_equal(neutral.final_indices,base.final_indices);np.testing.assert_array_equal(neutral.next_intervention_key,u.intervention_key)

def test_d1_checkpoint_rejects_d0_variant(tmp_path):
 from omegaconf import OmegaConf
 from evorl.algorithms.erl.erl_td3.utility_replay_d1 import D1CheckpointStore
 with pytest.raises(ValueError,match='variant'):D1CheckpointStore(tmp_path,OmegaConf.create(dict(alpha=.5)))

def test_all_negative_priorities_have_nonzero_limited_mass():
 t,tokens,mask=fixture();t=t._replace(raw_score=jnp.array([-1.,-2.,0.]),rank_normalized_score=jnp.array([priority_value(-1.,1.),priority_value(-2.,1.),.5]))
 d,m=build_d1_distribution(t,tokens,mask,0);assert 0<float(m)<.15 and d.source_probability[0]>d.source_probability[1]

def test_single_priority_empty_ea_and_future_scores():
 t,tokens,mask=fixture();d,m=build_d1_distribution(t._replace(resident_count=jnp.zeros(3,jnp.int32)),tokens,jnp.zeros(16,bool),0);assert not bool(d.has_resident_ea) and float(m)==0
 t=t._replace(measured_generation=jnp.array([10,10,-1]));d,m=build_d1_distribution(t,tokens,mask,0);assert float(m)==0

def test_fixed_priority_channel_is_normalized_and_reconstructs_final_probability():
 d,m=dist(15);assert float(d.weighted_source_probability.sum())==pytest.approx(1.)
 np.testing.assert_allclose(d.source_probability,.7*d.baseline_source_probability+.3*d.weighted_source_probability,atol=1e-7)

def test_d0_eligible_mass_conserved_but_d1_receives_global_mass():
 from evorl.algorithms.erl.erl_td3.source_replay import build_weighted_source_distribution
 t,tokens,mask=fixture();old=build_weighted_source_distribution(t,tokens,mask,alpha=.3,replay_temperature=.5);new,m=build_d1_distribution(t,tokens,mask,0)
 assert float(old.source_probability[:2].sum())==pytest.approx(2/3)
 assert float(new.source_probability[:2].sum())>2/3

def test_zero_rho_workflow_build_uses_actual_uniform_path_without_mutating_input():
 from hydra import initialize_config_dir,compose
 from pathlib import Path
 from evorl.algorithms.erl.erl_td3.utility_replay_d1 import ERLUtilityReplayD1Workflow
 with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):c=compose(config_name='config',overrides=['agent=erl/erl-utility-replay-d1','env=brax/swimmer','rho=0','replay_buffer_capacity=32','warmup_iters=0'])
 workflow=ERLUtilityReplayD1Workflow.build_from_config(c,enable_jit=True)
 try:
  assert workflow.config.alpha==0 and workflow.config.arm=='uniform' and not workflow._diagnostic_enabled and c.arm=='utility' and c.alpha==.3
 finally:workflow.close()
