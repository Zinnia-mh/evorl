"""Only new seed binding and measured threshold accounting."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf

def module():
    try:return importlib.import_module("scripts.run_s_utility_replication")
    except ModuleNotFoundError as e:
        if e.name=="scripts.run_s_utility_replication":pytest.fail("independent replication runner missing")
        raise

def test_first_threshold_uses_first_inclusive_evaluation_and_actual_recorded_time():
    m=module()
    rows=[dict(generation=20,return_mean=349.99,training_steps=650000,training_elapsed_seconds=90),
          dict(generation=30,return_mean=350.,training_steps=980000,training_elapsed_seconds=135),
          dict(generation=40,return_mean=500.,training_steps=1310000,training_elapsed_seconds=180)]
    result=m.first_threshold(rows)
    assert result["generation"]==30
    assert result["training_steps"]==980000
    assert result["training_elapsed_seconds"]==135
    assert m.first_threshold(rows[:1]) is None

def test_replication_config_changes_only_seed_and_output_contract_paths():
    m=module()
    old=OmegaConf.create(dict(seed=3079,alpha=.5,replay_temperature=.5,batch_size=256,
        agent_network=dict(actor_hidden_layer_sizes=[256,256]),arm_manifest_path="old",output_dir="old"))
    c=m.replication_config(old,"new-manifest","new-output")
    assert c.seed==3083 and old.seed==3079
    a,b=OmegaConf.to_container(old),OmegaConf.to_container(c)
    assert {k for k in a if a[k]!=b[k]}=={"seed","arm_manifest_path","output_dir"}

def test_p_uses_new_seed_in_the_same_permutation_function():
    m=module()
    from evorl.algorithms.erl.erl_td3.source_replay import init_source_score_table
    import jax.numpy as jnp
    from scripts.run_s_utility_extension import permute_cohort_ranks
    table=init_source_score_table(4)._replace(source_token=jnp.arange(10,14),
        resident_count=jnp.ones(4,dtype=jnp.int32),weighted_eligible=jnp.ones(4,bool),
        measured_generation=jnp.full(4,20),rank_normalized_score=jnp.array([0.,1/3,2/3,1.]))
    observer=m.ReplicationObserver("P",None,{})
    workflow=SimpleNamespace(config=SimpleNamespace(seed=3083),permutation={},permutation_seconds=0.)
    observer.configure(workflow,None)
    result=workflow._ablation_rank_transform(table)
    expected=permute_cohort_ranks(table,3083)
    np.testing.assert_array_equal(result.rank_normalized_score,expected.rank_normalized_score)
    assert workflow.permutation["preserved_rank_multisets"]
