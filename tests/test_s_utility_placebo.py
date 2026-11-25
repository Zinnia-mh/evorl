"""Only new permutation and explicit continuation contracts; no broad audit."""
import importlib
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

def module():
    try:return importlib.import_module("scripts.run_s_utility_extension")
    except ModuleNotFoundError as e:
        if e.name=="scripts.run_s_utility_extension":pytest.fail("S placebo continuation not implemented")
        raise

def table():
    from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table
    return init_source_score_table(8)._replace(
        source_token=jnp.arange(10,18),resident_count=jnp.full(8,4),
        weighted_eligible=jnp.array([True]*7+[False]),
        measured_generation=jnp.array([20,20,20,21,21,21,21,21]),
        rank_normalized_score=jnp.array([0.,.5,1.,0.,1/3,2/3,1.,.123]),
        raw_score=jnp.arange(8,dtype=jnp.float32))

def test_permutation_preserves_cohort_rank_multisets_and_canonical_state():
    m=module();t=table();before=np.asarray(t.rank_normalized_score).copy()
    p=m.permute_cohort_ranks(t,3079)
    q=m.permute_cohort_ranks(t,3079)
    np.testing.assert_array_equal(p.rank_normalized_score,q.rank_normalized_score)
    for group in ([0,1,2],[3,4,5,6]):
        np.testing.assert_array_equal(np.sort(np.asarray(p.rank_normalized_score)[group]),
            np.sort(before[group]))
    assert float(p.rank_normalized_score[-1])==float(t.rank_normalized_score[-1])
    assert np.any(np.asarray(p.rank_normalized_score)!=before)
    np.testing.assert_array_equal(t.rank_normalized_score,before)
    for name in t._fields:
        if name!="rank_normalized_score":
            np.testing.assert_array_equal(getattr(p,name),getattr(t,name))

def test_permuted_probability_distribution_reaches_real_sampler_with_same_rl_slots():
    m=module();t=table();p=m.permute_cohort_ranks(t,3079)
    from evorl.algorithms.erl.erl_td3.source_replay import build_weighted_source_distribution,apply_weighted_ea_overlay
    origins=jnp.array([0,1]*32,dtype=jnp.int8)
    tokens=jnp.where(origins==0,10+jnp.arange(64)//8,-1)
    d=build_weighted_source_distribution(t,tokens,origins==0,.5,.5)
    dp=build_weighted_source_distribution(p,tokens,origins==0,.5,.5)
    np.testing.assert_allclose(np.sort(d.source_probability[:3]),np.sort(dp.source_probability[:3]))
    np.testing.assert_allclose(np.sort(d.source_probability[3:7]),np.sort(dp.source_probability[3:7]))
    assert float(d.source_probability[-1])==float(dp.source_probability[-1])
    base=jax.random.randint(jax.random.PRNGKey(7),(4096,),0,64)
    result=apply_weighted_ea_overlay(base,origins,dp,jax.random.PRNGKey(8))
    b,f=np.asarray(base),np.asarray(result.final_indices)
    rl=np.asarray(origins)[b]==1
    np.testing.assert_array_equal(b[rl],f[rl])
    assert np.any(b[~rl]!=f[~rl]) and np.all(np.asarray(origins)[f]==np.asarray(origins)[b])
    assert not jax.config.jax_enable_x64

def test_continuation_contract_only_extends_budget_and_declared_arm_controls():
    m=module()
    a=OmegaConf.create(dict(arm="uniform",score_kind="uniform",alpha=0.,diagnostic_enabled=False,
        total_episodes=1640,seed=3079,batch_size=256,output_dir="old",arm_manifest_path="old"))
    b=OmegaConf.create(OmegaConf.to_container(a));b.total_episodes=4940
    b.arm="utility";b.score_kind="utility";b.alpha=.5;b.diagnostic_enabled=True
    b.output_dir="new";b.arm_manifest_path="new"
    m.validate_extension_config(a,b)
    b.batch_size=128
    with pytest.raises(ValueError,match="contract"):m.validate_extension_config(a,b)

def test_p_and_s_keep_same_source_selection_and_utility_path():
    m=module()
    s=m.ExtensionObserver("S",None,{})
    p=m.ExtensionObserver("P",None,{})
    assert s.workflow_type is p.workflow_type
    # Same declared workflow means same real scoring method; P changes distribution only.
    from evorl.algorithms.erl.erl_td3.learner_utility import SourceBatch
    objects=[]
    for observer in (s,p):
        obj=SimpleNamespace(config=SimpleNamespace(pop_size=32,diagnostic_interval=10,warmup_iters=10),
            _diagnostic_context=SimpleNamespace(generation=21,key=jax.random.PRNGKey(3)))
        sources=tuple(SourceBatch(str(i),i,None,True,None,1000) for i in range(32))
        objects.append(observer.workflow_type._select_diagnostic_sources(obj,sources))
    assert [x.individual_id for x in objects[0]]==[x.individual_id for x in objects[1]]


def test_resumed_score_journal_does_not_double_count_rolled_back_generations():
    m=module()
    assert hasattr(m,"canonical_score_rows"), "resume journal canonicalization missing"
    rows=[dict(generation=51,new_scores=3,step_seconds=1),
          dict(generation=52,new_scores=3,step_seconds=2),
          dict(generation=51,new_scores=3,step_seconds=9)]
    accepted=m.canonical_score_rows(rows)
    assert [r["generation"] for r in accepted]==[51,52]
    assert sum(r["new_scores"] for r in accepted)==6
    assert accepted[0]["step_seconds"]==9
    with pytest.raises(ValueError,match="score"):
        m.canonical_score_rows(rows+[dict(generation=51,new_scores=4)])


def test_fork_has_a_sealed_restart_checkpoint_before_first_training_step(tmp_path):
    m=module()
    assert hasattr(m,"create_bootstrap_checkpoint"), "fork start checkpoint missing"
    from evorl.types import State,PyTreeDict
    from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore
    state=State(key=jax.random.PRNGKey(7),metrics=PyTreeDict(iterations=jnp.uint32(50)),
        payload=jnp.arange(4,dtype=jnp.float32))
    config=OmegaConf.create(dict(seed=3079))
    segment=tmp_path/"run/segments/first";segment.mkdir(parents=True)
    runtime=dict(run_id="child",target_generation=150,evaluations=[dict(generation=50)])
    cp=m.create_bootstrap_checkpoint(segment,config,state,runtime)
    incomplete=segment/"checkpoints/100/runtime"
    incomplete.mkdir(parents=True);(incomplete/"metadata").write_text("{}")
    assert m.find_resume_checkpoint(tmp_path/"run")==cp
    template=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),state)
    with PilotCheckpointStore(cp.parent,config,read_only=True) as store:
        restored,journal=store.restore(50,template)
    np.testing.assert_array_equal(restored.payload,state.payload)
    np.testing.assert_array_equal(restored.key,state.key)
    assert journal==runtime
