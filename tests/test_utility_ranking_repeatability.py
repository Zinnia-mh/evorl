"""Frozen diagnostics: factor isolation, ranking metrics and genuine TD3 safety."""
import importlib
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax

def module():
    try:
        return importlib.import_module("scripts.measure_utility_ranking_repeatability")
    except ModuleNotFoundError as exc:
        if exc.name == "scripts.measure_utility_ranking_repeatability":
            pytest.fail("frozen ranking diagnostic is not implemented")
        raise

def test_factor_schedule_changes_only_the_designated_randomness():
    m=module()
    anchor=m.factor_keys(20261125,50,"A",0,640)
    for mode in "ABC":
        np.testing.assert_array_equal(m.factor_keys(20261125,50,mode,0,640),anchor)
        keys=[np.asarray(m.factor_keys(20261125,50,mode,r,640)) for r in range(8)]
        varying={"A":1,"B":2,"C":0}[mode]
        for factor in range(3):
            count=len({x[factor].tobytes() for x in keys})
            assert count==(8 if factor==varying else 1)
    # Reference randomness shared across sources; loss/source randomness independent.
    other=m.factor_keys(20261125,50,"A",0,641)
    np.testing.assert_array_equal(anchor[1],other[1])
    assert not np.array_equal(anchor[0],other[0])
    assert not np.array_equal(anchor[2],other[2])

def test_reversed_measurements_have_negative_correlations_and_all_pair_flips():
    s=module().rank_reliability(np.array([[0.,1.,2.,3.],[3.,2.,1.,0.]]))
    assert s["spearman_mean"]==pytest.approx(-1)
    assert s["kendall_mean"]==pytest.approx(-1)
    assert s["top25_overlap_mean"]==0
    assert s["pair_reversal_mean"]==1
    assert s["repeat_pairs"]==1

def test_all_ties_are_reported_undefined_not_perfectly_reliable():
    s=module().rank_reliability(np.ones((3,4)))
    assert s["spearman_mean"] is None
    assert s["kendall_mean"] is None
    assert s["strict_pair_fraction"]==0
    assert s["pair_reversal_mean"] is None
    assert s["top25_boundary_tie_repeats"]==3

def test_cohort_normalization_does_not_leak_between_different_cohort_offsets():
    m=module()
    raw=np.array([0.,1.,100.,101.],np.float32)
    np.testing.assert_array_equal(m.cohort_ranks(raw,np.array([20,20,30,30])),[0.,1.,0.,1.])

def test_probability_projection_preserves_noneligible_mass_and_uses_real_formula():
    m=module()
    from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table
    table=init_source_score_table(3)._replace(
        source_token=jnp.array([10,20,30]),resident_count=jnp.array([10,10,20]),
        weighted_eligible=jnp.array([True,True,False]))
    p=m.probability_projection(table,np.array([0,1]),np.array([0.,1.]))
    e=np.exp(2)
    np.testing.assert_allclose(p,[.125+.25/(1+e),.375-.25/(1+e),.5],rtol=1e-6)
    assert p.sum()==pytest.approx(1)

def test_real_td3_virtual_measurement_is_repeatable_and_does_not_write_state():
    m=module()
    from evorl.algorithms.td3 import make_mlp_td3_agent
    from evorl.algorithms.erl.erl_td3.erl_learner_utility import ERLLearnerUtilityWorkflow
    from evorl.envs import Box
    from evorl.sample_batch import SampleBatch
    from evorl.types import PyTreeDict
    from types import SimpleNamespace
    agent=make_mlp_td3_agent(Box(low=-jnp.ones(2),high=jnp.ones(2)),
        critic_hidden_layer_sizes=(8,8),actor_hidden_layer_sizes=(8,8))
    state=agent.init(Box(low=-jnp.ones(3),high=jnp.ones(3)),Box(low=-jnp.ones(2),high=jnp.ones(2)),jax.random.PRNGKey(1))
    optimizer=optax.adam(.0003)
    opt=optimizer.init(state.params.critic_params)
    batch=SampleBatch(obs=jnp.arange(24,dtype=jnp.float32).reshape(8,3)/24,
        actions=jnp.zeros((8,2)),rewards=jnp.arange(8,dtype=jnp.float32)/8,
        extras=PyTreeDict(env_extras=PyTreeDict(ori_obs=jnp.ones((8,3)),termination=jnp.zeros(8))))
    frozen=[np.asarray(x).copy() for x in jax.tree.leaves((state,opt))]
    kernel=m.make_measure_kernel(agent,state,optimizer,opt)
    key=jax.random.PRNGKey(3)
    td,result=kernel(batch,batch,key)
    _,again=kernel(batch,batch,key)
    context=SimpleNamespace(agent=agent,optimizer=optimizer,_diagnostic_context=
        SimpleNamespace(learner_agent_state=state,critic_opt_state=opt))
    _,native=ERLLearnerUtilityWorkflow._evaluate_source_utility(context,batch,batch,key)
    assert np.isfinite(float(result.utility)) and np.isfinite(float(td.mean))
    np.testing.assert_allclose(result,native,rtol=1e-4,atol=1e-5)
    np.testing.assert_array_equal(result,again)
    for before,after in zip(frozen,jax.tree.leaves((state,opt))):
        np.testing.assert_array_equal(before,np.asarray(after))
