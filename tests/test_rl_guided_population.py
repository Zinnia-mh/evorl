import importlib
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from evorl.ec.optimizers.erl_ga import ERLGAMod

def module():
    try:return importlib.import_module("scripts.rl_guided_population")
    except ModuleNotFoundError as e:
        if e.name=="scripts.rl_guided_population":pytest.fail("online population intervention not implemented")
        raise

def population():
    return {"Dense_0":{"kernel":jnp.arange(32*12,dtype=jnp.float32).reshape(32,4,3)/1000,
                       "bias":jnp.zeros((32,3))}}

def same(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

def test_paired_actor_perturbations_enter_exactly_tail8_and_preserve24():
    m=module();pop=population();actor=jax.tree.map(lambda x:x[5],pop)
    saved=jax.tree.map(np.asarray,actor)
    changed,candidates=m.inject_population(pop,actor,jax.random.PRNGKey(42),strength=.005)
    for p,q,c,a in zip(jax.tree.leaves(pop),jax.tree.leaves(changed),jax.tree.leaves(candidates),jax.tree.leaves(actor)):
        np.testing.assert_array_equal(q[:24],p[:24])
        np.testing.assert_array_equal(q[24:],c)
        np.testing.assert_allclose((c[:4]+c[4:])/2,jnp.broadcast_to(a,c[:4].shape),atol=3e-8)
        assert np.any(np.asarray(c[:4])!=np.asarray(a))
    same(actor,saved)

@pytest.mark.parametrize("external",[False,True])
@pytest.mark.parametrize("crossover",[False,True])
def test_pedigree_metadata_preserves_actual_original_ga_outputs_and_rng(external,crossover):
    m=module();kwargs=dict(pop_size=32,num_elites=4,external_size=1,enable_crossover=crossover)
    original=ERLGAMod(**kwargs);tracked=m.PedigreeGA(**kwargs)
    pop=population();state=original.init(pop,jax.random.PRNGKey(31))
    if external:state=state.replace(external_pop=jax.tree.map(lambda x:x[:1],pop))
    fitness=jnp.arange(32,dtype=jnp.float32)
    method="tell_external" if external else "tell"
    _,expected=getattr(original,method)(state,fitness)
    metadata,actual=getattr(tracked,method)(state,fitness)
    same(expected,actual)
    assert metadata.parent_a.shape==(32,) and metadata.parent_b.shape==(32,)
    if external:
        assert metadata.parent_a[-1]==-1
        assert 0 not in np.asarray(metadata.parent_a[:-1])

def test_injected_lineages_survive_real_elite_selection_and_next_generation():
    m=module();ga=m.PedigreeGA(pop_size=32,num_elites=4,external_size=1,enable_crossover=False)
    pop=population();actor=jax.tree.map(lambda x:x[2],pop)
    pop,_=m.inject_population(pop,actor,jax.random.PRNGKey(7),strength=.005)
    lineage=m.initial_lineage(32)
    for i in range(24,32):lineage[i]=dict(roots=[4000+i-24],depth=0,standard_rl=False)
    state=ga.init(pop,jax.random.PRNGKey(3))
    fitness=jnp.concatenate([jnp.zeros(24),jnp.arange(8,dtype=jnp.float32)+100])
    metadata,state=ga.tell(state,fitness)
    lineage=m.propagate_lineage(lineage,metadata.parent_a,metadata.parent_b)
    assert all(x["roots"] for x in lineage[:4])
    assert any(x["roots"] for x in lineage[4:])
    metadata,state=ga.tell(state,jnp.arange(32,dtype=jnp.float32))
    lineage=m.propagate_lineage(lineage,metadata.parent_a,metadata.parent_b)
    assert any(x["roots"] and x["depth"]>=2 for x in lineage)

def test_bootstrap_resume_restores_initial_pedigree_without_nonexistent_journal():
    from types import SimpleNamespace
    from pathlib import Path
    from scripts.run_rl_guided_population_pilot import Observer
    from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
    state={"x":jnp.array([1.])}
    plan={"shared_parent":{"checkpoint":"/not-read","generation":50,"full_state_sha256":initialization_fingerprint(state)}}
    observer=Observer(plan);observer.w=SimpleNamespace(lineage=None)
    observer.validate_resume(Path("50"),state)
    assert len(observer.w.lineage)==32
    assert not any(x["roots"] for x in observer.w.lineage)

def test_recovery_rows_use_segment_time_not_random_uuid_order(tmp_path):
    import json,os
    from scripts.run_rl_guided_population_pilot import accepted_rows
    for name,time,fitness in [("z-old",100,1),("a-new",200,2)]:
        path=tmp_path/"segments"/name/"evolution.jsonl";path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"generation":11,"fitness":fitness})+"\n")
        os.utime(path,(time,time))
    assert accepted_rows(tmp_path)[0]["fitness"]==2

def test_diagnostics_find_earlier_sealed_checkpoints_after_resume(tmp_path):
    from scripts.diagnose_rl_guided_population import checkpoint_segment
    for name,generation in [("first",50),("later",100),("later",150)]:
        segment=tmp_path/"segments"/name
        (segment/f"checkpoints/{generation}").mkdir(parents=True)
        (segment/"checkpoint_checksums").mkdir(exist_ok=True)
        (segment/f"checkpoint_checksums/{generation}.json").write_text("{}")
    assert checkpoint_segment(tmp_path,50).name=="first"
    assert checkpoint_segment(tmp_path,150).name=="later"
