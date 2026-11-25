"""Only the new budget schedule/selection plus its existing sampler boundary."""
import importlib
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

def module():
    try:return importlib.import_module("evorl.algorithms.erl.erl_td3.budgeted_utility_scoring")
    except ModuleNotFoundError as exc:
        if exc.name.endswith("budgeted_utility_scoring"):pytest.fail("budgeted scoring is not implemented")
        raise

def test_quota_matches_burst_budget_in_each_ten_generation_block():
    q=module().score_quota
    assert [q(g,32,10,10) for g in range(11,21)]==[3,3,3,3,4,3,3,3,3,4]
    assert sum(q(g,32,10,10) for g in range(11,51))==128
    assert all(q(g,32,10,10)==0 for g in range(11))
    for start in (11,21,31,101):
        assert sum(q(g,32,10,10) for g in range(start,start+10))==32

@pytest.mark.parametrize("pop,interval",[(4,10),(32,10),(1,1),(33,10)])
def test_integer_quota_handles_zero_score_generations(pop,interval):
    q=module().score_quota
    assert sum(q(g,pop,interval,10) for g in range(11,11+interval))==pop

def test_selection_preserves_source_identity_and_baseline_prng():
    m=module()
    from evorl.algorithms.erl.erl_td3.learner_utility import SourceBatch
    key=jax.random.PRNGKey(7)
    obj=SimpleNamespace(config=SimpleNamespace(pop_size=32,diagnostic_interval=10,warmup_iters=10),
        _diagnostic_context=SimpleNamespace(generation=11,key=key))
    sources=tuple(SourceBatch(f"gen_11_ind_{i}",i,None,True,None,1000) for i in range(32))
    before=np.asarray(key).copy()
    a=m.DistributedScoringMixin._select_diagnostic_sources(obj,sources)
    b=m.DistributedScoringMixin._select_diagnostic_sources(obj,sources)
    assert len(a)==3 and len({x.individual_id for x in a})==3
    assert a==b and all(x is sources[x.individual_id] for x in a)
    np.testing.assert_array_equal(key,before)

def test_original_diagnostic_selection_hook_is_identity():
    module()
    from evorl.algorithms.erl.erl_td3.erl_learner_utility import ERLLearnerUtilityWorkflow
    sources=(object(),object())
    assert ERLLearnerUtilityWorkflow._select_diagnostic_sources(None,sources) is sources

def test_selected_score_update_reaches_sampler_preserving_rl_slots():
    module()
    from evorl.algorithms.erl.erl_td3.source_replay import (
        init_source_score_table,build_weighted_source_distribution,
        apply_weighted_ea_overlay)
    replay=SimpleNamespace(origin_type=jnp.array([0,1]*8,dtype=jnp.int8),
        source_token=jnp.where(jnp.arange(16)%2==0,
        jnp.where(jnp.arange(16)<8,10,20),-1).astype(jnp.int32))
    table=init_source_score_table(2)._replace(source_token=jnp.array([10,20]),
        resident_count=jnp.array([4,4]),weighted_eligible=jnp.ones(2,bool),
        rank_normalized_score=jnp.array([0.,1.]))
    mask=replay.origin_type==0
    distribution=build_weighted_source_distribution(table,replay.source_token,mask,.5,.5)
    baseline=jax.random.randint(jax.random.PRNGKey(22),(1024,),0,16)
    result=apply_weighted_ea_overlay(baseline,replay.origin_type,distribution,jax.random.PRNGKey(23))
    b,f=map(np.asarray,(result.baseline_indices,result.final_indices))
    rl=np.asarray(replay.origin_type)[b]==1
    assert np.array_equal(f[rl],b[rl]) and np.any(f[~rl]!=b[~rl])
    assert np.all(np.asarray(replay.origin_type)[f]==np.asarray(replay.origin_type)[b])
    assert distribution.transition_cdf.dtype==jnp.float64
    assert not jax.config.jax_enable_x64


def test_shared_warmup_contract_allows_only_declared_arm_controls():
    from omegaconf import OmegaConf
    from scripts import run_budgeted_utility_pilot as runner
    assert hasattr(runner,"validate_shared_parent"), "shared warmup validation missing"
    a=OmegaConf.create(dict(arm="uniform",score_kind="uniform",alpha=0,
        diagnostic_enabled=False,output_dir="/old",arm_manifest_path="/old/manifest",
        seed=3079,replay_buffer_capacity=1000,optimizer=dict(lr=.0003)))
    b=OmegaConf.create(OmegaConf.to_container(a))
    b.arm="utility";b.score_kind="utility";b.alpha=.5;b.diagnostic_enabled=True
    b.output_dir="/new";b.arm_manifest_path="/new/manifest"
    runner.validate_shared_parent(a,b)
    b.optimizer.lr=.001
    with pytest.raises(ValueError,match="shared"):
        runner.validate_shared_parent(a,b)
