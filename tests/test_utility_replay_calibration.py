import copy
import pytest
from scripts.calibrate_utility_replay import registration, select, freeze


def corpus():
    reg = registration()
    rows = [dict(candidate_id=i, ess_fraction=.7, max_source_probability=.2,
                 uniform_floor=1-reg['candidates'][i]['alpha'], fresh_coverage=.6, expired_coverage=.4,
                 repeat_rate=.1, numerical_ok=True, memory_mib=20., overhead_ms=1.)
            for i in range(len(reg['candidates']))]
    return reg, rows


@pytest.mark.parametrize('field', ['evaluation_return', 'AUC', 'final_return', 'pilot_performance'])
def test_calibration_rejects_return_auc_and_final_return_columns(field):
    reg, rows = corpus(); rows[0][field] = 1
    with pytest.raises(ValueError): select(reg, rows)


def test_correctness_filter_rejects_invalid_candidates():
    reg, rows = corpus(); rows[0]['numerical_ok'] = False
    assert select(reg, rows)['selected_id'] != 0
    for row in rows: row['numerical_ok']=False
    with pytest.raises(ValueError,match='HOLD'): select(reg,rows)


def test_operational_band_filter_is_deterministic():
    reg, rows = corpus(); rows[0]['memory_mib'] = 100000
    assert select(reg, rows) == select(reg, copy.deepcopy(rows))
    assert select(reg, rows)['selected_id'] != 0


def test_ess_target_selects_minimum_registered_distance():
    reg, rows = corpus(); rows[1]['ess_fraction'] = reg['ess_target']
    assert select(reg, rows)['selected_id'] == 1


def test_tie_break_uses_max_mass_repeat_rate_overhead_then_lexicographic_order():
    reg, rows = corpus()
    rows[0]['max_source_probability'] = .3
    rows[1]['repeat_rate'] = .2
    rows[2]['overhead_ms'] = 2.
    result = select(reg, rows)
    assert result['selected_id'] == 3


def test_resolved_config_contains_candidate_hashes_thresholds_trace_and_checksums(tmp_path):
    reg, rows = corpus(); result = freeze(reg, rows, tmp_path/'out', {'source':'synthetic-test', 'input_sha256':'a'*64})
    assert all((tmp_path/'out'/name).exists() for name in ['candidates.json','calibration_metrics.csv','selection_trace.json','resolved_config.yaml','checksums.sha256'])
    assert result['registration_sha256'] and result['thresholds'] and result['trace']


def test_resolved_config_freezes_common_parameters_for_bcd(tmp_path):
    reg, rows = corpus(); result = freeze(reg, rows, tmp_path/'out', {'source':'synthetic-test', 'input_sha256':'a'*64})
    assert result['arms']['A']['alpha'] == 0
    assert len({tuple(result['arms'][arm][k] for k in ['alpha','replay_temperature','freshness_window','min_resident']) for arm in ['B','C','D']}) == 1


def test_resolved_config_is_reproducible_from_same_inputs(tmp_path):
    reg, rows = corpus(); p={'source':'synthetic-test','input_sha256':'a'*64}
    assert freeze(reg, rows, tmp_path/'a', p) == freeze(reg, rows, tmp_path/'b', p)
    assert (tmp_path/'a/checksums.sha256').read_bytes() == (tmp_path/'b/checksums.sha256').read_bytes()
