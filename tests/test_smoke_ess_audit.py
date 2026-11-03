from types import SimpleNamespace
from typing import NamedTuple
import copy
import jax,jax.numpy as jnp,numpy as np,pytest
from evorl.types import State,PyTreeDict
from evorl.algorithms.erl.erl_td3 import utility_replay_smoke_workflow as module
from evorl.algorithms.erl.erl_td3.erl_utility_replay import init_utility_replay_state
from evorl.algorithms.erl.erl_td3.utility_replay_smoke import validate_reports
import runpy
from pathlib import Path
reports=runpy.run_path(str(Path(__file__).with_name('test_smoke_feedback_validation.py')))['reports']

class Count(NamedTuple):count:object

def test_real_observe_generation_uses_audit64_and_keeps_raw_without_mutation(monkeypatch):
    utility=init_utility_replay_state(8,jax.random.PRNGKey(4))
    replay=PyTreeDict(source_token=jnp.array([0,1,2,-1,-1,-1,-1,-1]),origin_type=jnp.array([0,0,0,-1,-1,-1,-1,-1]),buffer_size=jnp.int32(3))
    state=State(key=jax.random.PRNGKey(1),metrics=PyTreeDict(iterations=jnp.uint32(1)),replay_buffer_state=replay,utility_replay_state=utility,agent_state=jnp.zeros(1),opt_state=PyTreeDict(actor=Count(jnp.int32(0)),critic=Count(jnp.int32(0))))
    original=module.build_weighted_source_distribution;captured=[]
    def rounded(*args):
        d=original(*args)._replace(ess_fraction=jnp.float32(1.000000238418579));captured.append(d);return d
    monkeypatch.setattr(module,'build_weighted_source_distribution',rounded)
    monkeypatch.setattr(module,'alpha_zero_sample',lambda *args:jnp.int32(0))
    observer=SimpleNamespace(config=SimpleNamespace(alpha=0.,freshness_window=30,min_resident=4,replay_temperature=.5,batch_size=2,enable_eval_loss_endpoint=False),draw_recorder=module.DrawRecorder())
    row=module.SmokeWorkflow.observe_generation(observer,state,state,0.)
    assert row['ess_fraction']==row['ess_fraction_audit64']==1.
    assert row['ess_raw_float32']==1.000000238418579
    assert row['ess_audit_version']=='normalized-variance-float64-v1'
    assert float(captured[0].ess_fraction)==row['ess_raw_float32']
    np.testing.assert_array_equal(state.key,jax.random.PRNGKey(1))
    assert row['source_count']==3

def audit_fields(row):
    row.update(ess_audit_version='normalized-variance-float64-v1',ess_raw_float32=1.000000238418579,ess_fraction=1.,ess_fraction_audit64=1.,effective_source_count_audit64=3.,source_count=3,source_probability_mass64=1.,source_probability_squared_mass64=1/3,transition_probability_mass64=1.,cdf_sampling_mass64=1.,cdf_probability_max_abs_error=1e-7,cdf_non_ea_interval_mass64=0.)

def test_validator_accepts_high_precision_uniform_while_preserving_out_of_bound_raw():
    data=reports()
    for arm in data.values():
        for row in arm['rows']:audit_fields(row)
    assert validate_reports(data)['status']=='PASS'

@pytest.mark.parametrize('fault',['missing','missing_version','fraction','raw_nan','mass','effective','version'])
def test_validator_rejects_incomplete_or_tampered_numerical_evidence(fault):
    data=reports();row=data['A']['rows'][0];audit_fields(row)
    if fault=='missing':del row['ess_raw_float32']
    elif fault=='missing_version':del row['ess_audit_version']
    elif fault=='fraction':row['ess_fraction_audit64']=.9
    elif fault=='raw_nan':row['ess_raw_float32']=float('nan')
    elif fault=='mass':row['source_probability_mass64']=.9
    elif fault=='effective':row['effective_source_count_audit64']=4.
    else:row['ess_audit_version']='unknown'
    with pytest.raises(ValueError):validate_reports(data)

def test_legacy_out_of_bounds_ess_is_still_rejected():
    data=reports();data['A']['rows'][0]['ess_fraction']=1.000000238418579
    with pytest.raises(ValueError):validate_reports(data)
