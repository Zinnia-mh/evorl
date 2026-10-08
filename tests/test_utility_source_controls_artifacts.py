import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.utility_source_control_artifacts import (
    CHECKPOINT_SUMMARY_HEADER,
    CROSS_SOURCE_HEADER,
    PER_SOURCE_HEADER,
    REFERENCE_STABILITY_HEADER,
    SOURCE_STATUS_HEADER,
    STEP_SCALE_HEADER,
    UtilitySourceControlArtifactWriter,
    classify_cases,
    compute_ea_rank_metrics,
)
from evorl.algorithms.erl.erl_td3.utility_source_controls import (
    CheckpointDiagnostic,
    SourceControlBatch,
    SourceUtilityRecord,
)


EXPECTED_PER_SOURCE_HEADER = (
    "checkpoint",
    "generation",
    "source_type",
    "source_id",
    "reference_id",
    "reference_fingerprint",
    "step_scale",
    "record_kind",
    "source_batch_size",
    "valid_transition_count",
    "loss_ref_before",
    "loss_ref_after",
    "utility",
    "numerical_tolerance",
    "td_error_mean",
    "td_error_median",
    "td_error_std",
    "evolutionary_fitness",
    "eligible",
    "ineligible_reason",
    "temporal_alignment",
)
EXPECTED_SOURCE_STATUS_HEADER = (
    "checkpoint",
    "source_type",
    "source_id",
    "eligible",
    "reason",
    "valid_transition_count",
    "evolutionary_fitness",
    "measurement_eligible",
    "comparison_eligible",
    "checkpoint_valid",
)
EXPECTED_CHECKPOINT_HEADER = (
    "checkpoint",
    "valid",
    "measurement_eligible",
    "comparison_eligible",
    "checkpoint_status",
    "temporal_alignment",
    "reasons",
    "eligibility_reasons",
    "warnings",
    "num_ea_sources",
    "num_eligible_ea",
    "rl_own_eligible",
    "num_replay_random",
    "num_eligible_replay_random",
    "num_reference_self",
    "num_eligible_reference_self",
    "formal_record_count",
    "zero_step_record_count",
    "ea_spearman_rho",
    "ea_kendall_tau_b",
    "ea_comparable_pair_count",
    "ea_inversion_count",
    "ea_inversion_rate",
    "top_f_utility_rank",
    "top_f_equals_top_u",
    "num_top_f_ties",
    "num_top_u_ties",
    "rank_metric_reason",
)
EXPECTED_CROSS_HEADER = (
    "source_type",
    "step_scale",
    "n_checkpoints",
    "n_sources",
    "n_records",
    "utility_mean",
    "utility_median",
    "negative_rate",
    "near_zero_rate",
)
EXPECTED_STEP_HEADER = EXPECTED_CROSS_HEADER
EXPECTED_REFERENCE_HEADER = (
    "checkpoint",
    "source_type",
    "source_id",
    "step_scale",
    "utility_mean",
    "utility_median",
    "utility_std",
    "negative_fraction",
    "near_zero_fraction",
    "sign_consistency",
    "all_near_zero",
)


def _record(
    checkpoint, source_type, source_id, reference, scale, utility, fitness=None
):
    return SourceUtilityRecord(
        checkpoint=checkpoint,
        generation=checkpoint,
        source_type=source_type,
        source_id=source_id,
        reference_id=f"reference_{reference}",
        reference_fingerprint=f"r{reference}",
        step_scale=scale,
        source_batch_size=2,
        loss_ref_before=10.0,
        loss_ref_after=10.0 - utility,
        utility=utility,
        numerical_tolerance=1e-6,
        td_error_mean=2.0,
        td_error_median=2.0,
        td_error_std=0.5,
        evolutionary_fitness=fitness,
        temporal_alignment="exact",
    )


def _diagnostic(checkpoint=1, *, valid=True, measurement=True, comparison=True):
    source_defs = (
        ("EA", "ea_0", 1.0, 30.0),
        ("EA", "ea_1", 2.0, 20.0),
        ("EA", "ea_2", 3.0, 10.0),
        ("RL-own", "rl", 1.0, None),
        ("Replay-random", "replay_0", -1.0, None),
        ("Reference-self", "reference_self_0", 0.5, None),
    )
    statuses = tuple(
        SourceControlBatch(
            source_type=source_type,
            source_id=source_id,
            batch=None,
            valid_transition_count=2,
            evolutionary_fitness=fitness,
        )
        for source_type, source_id, _, fitness in source_defs
    )
    records = tuple(
        _record(
            checkpoint,
            source_type,
            source_id,
            reference,
            scale,
            base * scale + reference * 0.01,
            fitness,
        )
        for source_type, source_id, base, fitness in source_defs
        for scale in (1.0, 0.5, 0.1)
        for reference in range(4)
    )
    zero = tuple(
        replace(record, step_scale=0.0, utility=0.0, loss_ref_after=10.0)
        for record in records
        if record.step_scale == 1.0
    )
    return CheckpointDiagnostic(
        checkpoint=checkpoint,
        temporal_alignment="exact",
        records=records,
        zero_step_results=zero,
        valid=valid,
        reasons=() if valid else ("zero_step_failure",),
        measurement_eligible=measurement,
        comparison_eligible=comparison,
        source_statuses=statuses,
    )


def _writer(tmp_path, **config):
    allowed = tmp_path / "outputs"
    run = allowed / "run-1"
    defaults = {
        "source_batch_size": 2,
        "num_reference_batches": 4,
        "num_random_control_batches": 1,
        "step_scales": [1.0, 0.5, 0.1],
        "numerical_tolerance": 1e-6,
        "zero_step_tolerance": 1e-6,
        "reference_self_warning_threshold": 0.0,
        "ea_mismatch_negative_rate_margin": 0.1,
        "step_sensitivity_negative_rate_drop": 0.2,
        "case_b_negative_rate_min": 0.8,
        "case_b_negative_rate_max": 0.9,
    }
    defaults.update(config)
    return UtilitySourceControlArtifactWriter(
        run,
        allowed_root=allowed,
        forbidden_roots=(),
        run_config=defaults,
    )


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exact_per_source_header():
    assert PER_SOURCE_HEADER == EXPECTED_PER_SOURCE_HEADER


def test_exact_source_status_header():
    assert SOURCE_STATUS_HEADER == EXPECTED_SOURCE_STATUS_HEADER


def test_exact_checkpoint_summary_header():
    assert CHECKPOINT_SUMMARY_HEADER == EXPECTED_CHECKPOINT_HEADER


def test_exact_cross_source_header():
    assert CROSS_SOURCE_HEADER == EXPECTED_CROSS_HEADER


def test_exact_step_scale_header():
    assert STEP_SCALE_HEADER == EXPECTED_STEP_HEADER


def test_exact_reference_stability_header():
    assert REFERENCE_STABILITY_HEADER == EXPECTED_REFERENCE_HEADER


def test_append_checkpoint_writes_formal_and_zero_step_raw_records(tmp_path):
    writer = _writer(tmp_path)
    diagnostic = _diagnostic()
    writer.append_checkpoint(diagnostic)
    rows = _rows(writer.root / "per_source_utility.csv")
    assert sum(row["record_kind"] == "formal" for row in rows) == len(
        diagnostic.records
    )
    assert sum(row["record_kind"] == "zero_step" for row in rows) == len(
        diagnostic.zero_step_results
    )


def test_source_status_preserves_ineligible_rl_own(tmp_path):
    writer = _writer(tmp_path)
    diagnostic = _diagnostic()
    statuses = tuple(
        replace(s, eligible=False, reason="insufficient_rl_own_samples")
        if s.source_type == "RL-own"
        else s
        for s in diagnostic.source_statuses
    )
    writer.append_checkpoint(
        replace(diagnostic, comparison_eligible=False, source_statuses=statuses)
    )
    rl = next(
        row
        for row in _rows(writer.root / "source_status.csv")
        if row["source_type"] == "RL-own"
    )
    assert rl["eligible"] == "False"
    assert rl["reason"] == "insufficient_rl_own_samples"


def test_valid_complete_checkpoint_enters_cross_source_summary(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    writer.finalize_run()
    assert {
        row["source_type"] for row in _rows(writer.root / "cross_source_summary.csv")
    } == {"EA", "RL-own", "Replay-random", "Reference-self"}


def test_valid_incomplete_checkpoint_does_not_enter_case_ab(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic(comparison=False))
    result = writer.finalize_run()
    assert result["classification"] == "INCOMPLETE"


def test_invalid_checkpoint_retains_raw_but_excluded_from_formal_summary(tmp_path):
    writer = _writer(tmp_path)
    diagnostic = _diagnostic(valid=False, comparison=False)
    writer.append_checkpoint(diagnostic)
    writer.finalize_run()
    assert _rows(writer.root / "per_source_utility.csv")
    assert _rows(writer.root / "source_status.csv")
    assert _rows(writer.root / "checkpoint_summary.csv")
    assert _rows(writer.root / "cross_source_summary.csv") == []


def test_ea_rank_uses_cross_reference_median_at_scale_one():
    metrics = compute_ea_rank_metrics(_diagnostic())
    assert metrics["top_f_utility_rank"] == 3.0
    assert metrics["comparable_pair_count"] == 3


def _rank_diagnostic(fitness, utility):
    statuses = tuple(
        SourceControlBatch("EA", f"ea_{i}", None, evolutionary_fitness=f)
        for i, f in enumerate(fitness)
    )
    records = tuple(
        _record(1, "EA", f"ea_{i}", ref, 1.0, u, f)
        for i, (f, u) in enumerate(zip(fitness, utility))
        for ref in range(4)
    )
    return CheckpointDiagnostic(1, "exact", records=records, source_statuses=statuses)


def test_spearman_fixture():
    assert compute_ea_rank_metrics(_rank_diagnostic([1, 2, 3], [1, 2, 3]))[
        "spearman_rho"
    ] == pytest.approx(1.0)


def test_kendall_tau_b_fixture():
    assert compute_ea_rank_metrics(_rank_diagnostic([1, 2, 3], [3, 2, 1]))[
        "kendall_tau_b"
    ] == pytest.approx(-1.0)


def test_inversion_rate_fixture():
    assert (
        compute_ea_rank_metrics(_rank_diagnostic([1, 2, 3], [3, 2, 1]))[
            "inversion_rate"
        ]
        == 1.0
    )


def test_top_f_utility_rank_fixture():
    assert (
        compute_ea_rank_metrics(_rank_diagnostic([1, 2, 3], [3, 2, 1]))[
            "top_f_utility_rank"
        ]
        == 3.0
    )


def test_top_f_equals_top_u_fixture():
    result = compute_ea_rank_metrics(_rank_diagnostic([2, 2, 1], [1, 3, 3]))
    assert result["top_f_equals_top_u"] is True
    assert result["num_top_f_ties"] == 2
    assert result["num_top_u_ties"] == 2


def test_rank_metrics_handle_ties():
    result = compute_ea_rank_metrics(_rank_diagnostic([1, 1, 2], [1, 2, 2]))
    assert result["spearman_rho"] == pytest.approx(0.5)
    assert result["kendall_tau_b"] == pytest.approx(0.5)


def test_rank_metrics_handle_insufficient_sources():
    result = compute_ea_rank_metrics(_rank_diagnostic([1], [2]))
    assert result["spearman_rho"] is None
    assert result["rank_metric_reason"] == "insufficient_eligible_ea_sources"


def test_case_a_fixture():
    result = classify_cases({"EA": 0.5, "RL-own": 0.2, "Replay-random": 0.3}, {}, {})
    assert result["case_a"] is True


def test_case_b_fixture():
    rates = {"EA": 0.85, "RL-own": 0.85, "Replay-random": 0.85}
    assert classify_cases(rates, {}, {})["case_b"] is True


def test_case_c_invalid_fixture():
    assert (
        classify_cases({}, {}, {"valid": False, "reasons": ["zero_step_failure"]})[
            "classification"
        ]
        == "CASE_C_INVALID"
    )


def test_case_d_fixture():
    scales = {1.0: 0.6, 0.5: 0.5, 0.1: 0.3}
    assert classify_cases({}, scales, {})["case_d"] is True


def test_classification_ambiguous_fixture():
    rates = {"EA": 0.85, "RL-own": 0.85, "Replay-random": 0.85}
    scales = {1.0: 0.6, 0.5: 0.5, 0.1: 0.3}
    assert classify_cases(rates, scales, {})["classification_ambiguous"] is True


def test_report_never_uses_proves_language(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    writer.finalize_run()
    report = (writer.root / "diagnostic_report.md").read_text(encoding="utf-8").lower()
    assert "proves" not in report
    assert "proof" not in report
    assert "not a causal" in report


def test_atomic_write_preserves_previous_valid_summary_on_failure(
    tmp_path, monkeypatch
):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    writer.finalize_run()
    target = writer.root / "checkpoint_summary.csv"
    before = target.read_bytes()
    monkeypatch.setattr(
        writer,
        "_replace_atomic",
        lambda temporary, target: (_ for _ in ()).throw(RuntimeError("interrupt")),
    )
    with pytest.raises(RuntimeError, match="interrupt"):
        writer.finalize_run()
    assert target.read_bytes() == before


def test_duplicate_checkpoint_append_is_rejected(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        writer.append_checkpoint(_diagnostic())


def test_reopening_existing_run_preserves_raw_evidence(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    raw = writer.root / "per_source_utility.csv"
    statuses = writer.root / "source_status.csv"
    before = raw.read_bytes()
    statuses_before = statuses.read_bytes()
    reopened = UtilitySourceControlArtifactWriter(
        writer.root,
        allowed_root=tmp_path / "outputs",
        forbidden_roots=(),
        run_config=writer.run_config,
    )
    assert raw.read_bytes() == before
    assert statuses.read_bytes() == statuses_before
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        reopened.append_checkpoint(_diagnostic())


def test_finalize_is_deterministic(tmp_path):
    writer = _writer(tmp_path)
    writer.append_checkpoint(_diagnostic())
    writer.finalize_run()
    first = {p.name: p.read_bytes() for p in writer.root.iterdir() if p.is_file()}
    writer.finalize_run()
    second = {p.name: p.read_bytes() for p in writer.root.iterdir() if p.is_file()}
    assert second == first


def test_run_config_records_common_randomness_policy(tmp_path):
    writer = _writer(tmp_path)
    config = json.loads((writer.root / "run_config.json").read_text())
    assert "reference_self_systemic_negative_threshold" in config
    assert config["source_update_key_scope"] == "checkpoint"
    assert config["reference_eval_key_scope"] == "checkpoint_reference"
    assert config["reference_eval_key_independent_of_source"] is True
    assert config["reference_eval_key_independent_of_step_scale"] is True


def test_run_config_persists_complete_static_and_checkpoint_provenance(tmp_path):
    static = {
        "repository_path": "/repo",
        "worktree_path": "/worktree",
        "branch": "research/utility-source-controls",
        "commit": "abc123",
        "artifact_root": "/outputs/run-1",
        "tmux_session": "runtime-session",
        "tmux_window": "runtime-window",
        "conda_env": "evorl-conflict",
        "python_version": "3.11.16",
        "jax_version": "0.9.0",
        "jaxlib_version": "0.9.0",
        "flax_version": "0.11.2",
        "optax_version": "0.2.8",
        "device": "TFRT_CPU_0",
        "environment": "brax/swimmer",
        "seed": 42,
        "population": 4,
        "timestamp": "2026-10-08T00:00:00+00:00",
    }
    writer = _writer(tmp_path, **static)
    writer.append_checkpoint(
        _diagnostic(1),
        checkpoint_provenance={
            "baseline_prng_fingerprint": "prng-1",
            "replay_metadata_fingerprint": "replay-1",
            "temporal_alignment": "exact",
        },
    )
    writer.append_checkpoint(
        _diagnostic(2),
        checkpoint_provenance={
            "baseline_prng_fingerprint": "prng-2",
            "replay_metadata_fingerprint": "replay-2",
            "temporal_alignment": "delayed",
        },
    )
    config = json.loads((writer.root / "run_config.json").read_text())
    required = {
        "repository_path",
        "worktree_path",
        "branch",
        "commit",
        "artifact_root",
        "tmux_session",
        "tmux_window",
        "conda_env",
        "python_version",
        "jax_version",
        "jaxlib_version",
        "flax_version",
        "optax_version",
        "device",
        "environment",
        "seed",
        "population",
        "source_batch_size",
        "num_reference_batches",
        "num_random_control_batches",
        "step_scales",
        "numerical_tolerance",
        "zero_step_tolerance",
        "timestamp",
        "source_update_key_scope",
        "reference_eval_key_scope",
        "checkpoint_provenance",
    }
    assert required <= config.keys()
    assert {key: config[key] for key in static} == static
    assert config["checkpoint_provenance"] == {
        "1": {
            "baseline_prng_fingerprint": "prng-1",
            "replay_metadata_fingerprint": "replay-1",
            "temporal_alignment": "exact",
        },
        "2": {
            "baseline_prng_fingerprint": "prng-2",
            "replay_metadata_fingerprint": "replay-2",
            "temporal_alignment": "delayed",
        },
    }

    reopened = UtilitySourceControlArtifactWriter(
        writer.root,
        allowed_root=tmp_path / "outputs",
        forbidden_roots=(),
        run_config=static
        | {
            "source_batch_size": 2,
            "num_reference_batches": 4,
            "num_random_control_batches": 1,
            "step_scales": [1.0, 0.5, 0.1],
            "numerical_tolerance": 1e-6,
            "zero_step_tolerance": 1e-6,
            "reference_self_warning_threshold": 0.0,
            "ea_mismatch_negative_rate_margin": 0.1,
            "step_sensitivity_negative_rate_drop": 0.2,
            "case_b_negative_rate_min": 0.8,
            "case_b_negative_rate_max": 0.9,
        },
    )
    assert reopened.run_config["checkpoint_provenance"] == config[
        "checkpoint_provenance"
    ]


def _source_control_config():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "agent"
        / "erl"
        / "erl-utility-source-controls.yaml"
    )
    return OmegaConf.load(path)


def test_source_control_config_defaults_disabled():
    config = _source_control_config()
    assert config.workflow_cls.endswith(".ERLUtilitySourceControlsWorkflow")
    assert config.diagnostic_enabled is False
    assert config.artifact_root is None


def test_config_reference_and_control_counts():
    config = _source_control_config()
    assert config.source_batch_size == 256
    assert config.num_reference_batches == 4
    assert config.num_random_control_batches == 8


def test_config_step_scales():
    config = _source_control_config()
    assert list(config.source_control_step_scales) == [1.0, 0.5, 0.1]
