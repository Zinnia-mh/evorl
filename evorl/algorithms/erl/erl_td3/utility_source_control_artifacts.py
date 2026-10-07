"""Host-only artifact writing for learner-utility source controls."""
from __future__ import annotations

import csv, io, json, os
from collections import defaultdict
from pathlib import Path
from statistics import median
import numpy as np
from scipy import stats
from .utility_source_controls import (
    CheckpointDiagnostic,
    SourceControlBatch,
    SourceUtilityRecord,
    aggregate_reference_utilities,
)

PER_SOURCE_HEADER = (
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
SOURCE_STATUS_HEADER = (
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
CHECKPOINT_SUMMARY_HEADER = (
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
CROSS_SOURCE_HEADER = (
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
STEP_SCALE_HEADER = CROSS_SOURCE_HEADER
REFERENCE_STABILITY_HEADER = (
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
SOURCE_ORDER = {
    v: i for i, v in enumerate(("EA", "RL-own", "Replay-random", "Reference-self"))
}


def validate_artifact_root(candidate, allowed_root, forbidden_roots):
    candidate = Path(candidate)
    if not candidate.is_absolute():
        raise ValueError("artifact root must be absolute")
    candidate = candidate.resolve(strict=False)
    allowed = Path(allowed_root).resolve(strict=False)
    forbidden = tuple(Path(p).resolve(strict=False) for p in forbidden_roots)
    if any(candidate == p or candidate.is_relative_to(p) for p in forbidden):
        raise ValueError("artifact root resolves inside a forbidden root")
    if not (candidate == allowed or candidate.is_relative_to(allowed)):
        raise ValueError("artifact root must resolve under the allowed output root")
    return candidate


def _csv(header, rows):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in header})
    return s.getvalue()


def _join(values):
    return "|".join(sorted(set(values)))


def _status(d):
    if not d.valid:
        return "INVALID"
    return (
        "VALID_COMPLETE"
        if d.measurement_eligible and d.comparison_eligible
        else "VALID_INCOMPLETE"
    )


def compute_ea_rank_metrics(d):
    empty = {
        "spearman_rho": None,
        "kendall_tau_b": None,
        "comparable_pair_count": 0,
        "inversion_count": 0,
        "inversion_rate": None,
        "top_f_utility_rank": None,
        "top_f_equals_top_u": None,
        "num_top_f_ties": 0,
        "num_top_u_ties": 0,
        "rank_metric_reason": None,
    }
    if not (d.valid and d.measurement_eligible and d.comparison_eligible):
        return {**empty, "rank_metric_reason": "checkpoint_not_comparable"}
    eligible = {
        s.source_id for s in d.source_statuses if s.source_type == "EA" and s.eligible
    }
    groups = defaultdict(list)
    for r in d.records:
        if r.source_type == "EA" and r.step_scale == 1.0 and r.source_id in eligible:
            groups[r.source_id].append(r)
    obs = []
    for sid, rs in sorted(groups.items()):
        if rs[0].evolutionary_fitness is not None:
            obs.append(
                (
                    sid,
                    float(rs[0].evolutionary_fitness),
                    median(float(r.utility) for r in rs),
                )
            )
    if len(obs) < 2:
        return {**empty, "rank_metric_reason": "insufficient_eligible_ea_sources"}
    f = np.array([x[1] for x in obs])
    u = np.array([x[2] for x in obs])
    if np.all(f == f[0]):
        return {**empty, "rank_metric_reason": "all_fitness_identical"}
    if np.all(u == u[0]):
        return {**empty, "rank_metric_reason": "all_utility_identical"}
    comparable = inversions = 0
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            p = (f[i] - f[j]) * (u[i] - u[j])
            if p:
                comparable += 1
                inversions += int(p < 0)
    topf = set(np.flatnonzero(f == f.max()))
    topu = set(np.flatnonzero(u == u.max()))
    ranks = stats.rankdata(-u, method="average")
    return {
        "spearman_rho": float(stats.spearmanr(f, u).statistic),
        "kendall_tau_b": float(stats.kendalltau(f, u, variant="b").statistic),
        "comparable_pair_count": comparable,
        "inversion_count": inversions,
        "inversion_rate": inversions / comparable if comparable else None,
        "top_f_utility_rank": float(min(ranks[list(topf)])),
        "top_f_equals_top_u": bool(topf & topu),
        "num_top_f_ties": len(topf),
        "num_top_u_ties": len(topu),
        "rank_metric_reason": None,
    }


def classify_cases(rates, scales, status, config=None):
    c = config or {}
    valid = status.get("valid", True)
    measurement = status.get("measurement_eligible", True)
    comparison = status.get("comparison_eligible", True)
    if not valid:
        return {
            "classification": "CASE_C_INVALID",
            "case_a": False,
            "case_b": False,
            "case_c": True,
            "case_d": False,
            "classification_ambiguous": False,
            "reasons": tuple(status.get("reasons", ())),
        }
    if not (measurement and comparison):
        return {
            "classification": "INCOMPLETE",
            "case_a": False,
            "case_b": False,
            "case_c": False,
            "case_d": False,
            "classification_ambiguous": False,
            "reasons": (),
        }
    required = ("EA", "RL-own", "Replay-random")
    have = all(x in rates for x in required)
    margin = float(c.get("ea_mismatch_negative_rate_margin", 0.1))
    lo = float(c.get("case_b_negative_rate_min", 0.8))
    hi = float(c.get("case_b_negative_rate_max", 0.9))
    drop = float(c.get("step_sensitivity_negative_rate_drop", 0.2))
    a = have and all(rates["EA"] - rates[x] >= margin for x in required[1:])
    b = have and all(lo <= rates[x] <= hi for x in required)
    hs = all(x in scales for x in (1.0, 0.5, 0.1))
    d = (
        hs
        and scales[1.0] >= scales[0.5] >= scales[0.1]
        and scales[1.0] - scales[0.1] >= drop
    )
    conflict = (
        have
        and all(rates[x] >= lo for x in required)
        and any(rates[x] > hi for x in required)
    )
    matches = [n for n, v in (("CASE_A", a), ("CASE_B", b), ("CASE_D", d)) if v]
    ambiguous = len(matches) > 1 or conflict
    return {
        "classification": "AMBIGUOUS"
        if ambiguous
        else (matches[0] if matches else "NO_CASE"),
        "case_a": bool(a),
        "case_b": bool(b),
        "case_c": False,
        "case_d": bool(d),
        "classification_ambiguous": bool(ambiguous),
        "reasons": (),
    }


class UtilitySourceControlArtifactWriter:
    def __init__(self, artifact_root, *, allowed_root, forbidden_roots, run_config):
        self.root = validate_artifact_root(artifact_root, allowed_root, forbidden_roots)
        self.root.mkdir(parents=True, exist_ok=True)
        managed = (
            "run_config.json",
            "per_source_utility.csv",
            "source_status.csv",
            "checkpoint_summary.csv",
            "cross_source_summary.csv",
            "step_scale_summary.csv",
            "reference_stability.csv",
            "diagnostic_report.md",
        )
        requested_config = dict(run_config)
        self.run_config = dict(requested_config)
        defaults = {
            "source_batch_size": None,
            "num_reference_batches": 4,
            "num_random_control_batches": 8,
            "step_scales": [1.0, 0.5, 0.1],
            "numerical_tolerance": 1e-6,
            "zero_step_tolerance": 1e-6,
            "reference_self_warning_threshold": 0.0,
            "reference_self_systemic_negative_threshold": 0.0,
            "ea_mismatch_negative_rate_margin": 0.1,
            "step_sensitivity_negative_rate_drop": 0.2,
            "case_b_negative_rate_min": 0.8,
            "case_b_negative_rate_max": 0.9,
        }
        for key, value in defaults.items():
            self.run_config.setdefault(key, value)
        self.run_config.update(
            {
                "source_update_key_scope": "checkpoint",
                "reference_eval_key_scope": "checkpoint_reference",
                "reference_eval_key_independent_of_source": True,
                "reference_eval_key_independent_of_step_scale": True,
            }
        )
        existing = tuple(name for name in managed if (self.root / name).exists())
        required = {
            "run_config.json",
            "per_source_utility.csv",
            "source_status.csv",
        }
        if existing:
            if not required.issubset(existing):
                raise ValueError("artifact run directory is partially initialized")
            self._restore_existing_run(requested_config)
            return
        self._diagnostics = []
        self._checkpoints = set()
        self._atomic_json("run_config.json", self.run_config)
        self._atomic_text("per_source_utility.csv", _csv(PER_SOURCE_HEADER, ()))
        self._atomic_text("source_status.csv", _csv(SOURCE_STATUS_HEADER, ()))

    @staticmethod
    def _read_rows(path, expected_header):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_header):
                raise ValueError(f"artifact schema mismatch: {path.name}")
            return list(reader)

    @staticmethod
    def _optional_float(value):
        return None if value in (None, "") else float(value)

    @staticmethod
    def _bool(value):
        return str(value).lower() == "true"

    def _restore_existing_run(self, requested_config):
        existing_config = json.loads(
            (self.root / "run_config.json").read_text(encoding="utf-8")
        )
        for key, value in requested_config.items():
            if key not in existing_config or existing_config[key] != value:
                raise ValueError(f"run_config mismatch for {key}")
        for key in (
            "source_update_key_scope",
            "reference_eval_key_scope",
            "reference_eval_key_independent_of_source",
            "reference_eval_key_independent_of_step_scale",
        ):
            if existing_config.get(key) != self.run_config[key]:
                raise ValueError(f"run_config mismatch for {key}")
        raw_rows = self._read_rows(
            self.root / "per_source_utility.csv", PER_SOURCE_HEADER
        )
        status_rows = self._read_rows(
            self.root / "source_status.csv", SOURCE_STATUS_HEADER
        )
        summary_path = self.root / "checkpoint_summary.csv"
        summary_rows = (
            self._read_rows(summary_path, CHECKPOINT_SUMMARY_HEADER)
            if summary_path.exists()
            else []
        )
        summaries = {int(row["checkpoint"]): row for row in summary_rows}
        records = defaultdict(lambda: {"formal": [], "zero_step": []})
        for row in raw_rows:
            record = SourceUtilityRecord(
                checkpoint=int(row["checkpoint"]),
                generation=int(row["generation"]),
                source_type=row["source_type"],
                source_id=row["source_id"],
                reference_id=row["reference_id"],
                reference_fingerprint=row["reference_fingerprint"],
                step_scale=float(row["step_scale"]),
                source_batch_size=int(row["source_batch_size"]),
                loss_ref_before=float(row["loss_ref_before"]),
                loss_ref_after=float(row["loss_ref_after"]),
                utility=float(row["utility"]),
                numerical_tolerance=float(row["numerical_tolerance"]),
                td_error_mean=self._optional_float(row["td_error_mean"]),
                td_error_median=self._optional_float(row["td_error_median"]),
                td_error_std=self._optional_float(row["td_error_std"]),
                evolutionary_fitness=self._optional_float(row["evolutionary_fitness"]),
                eligible=self._bool(row["eligible"]),
                ineligible_reason=row["ineligible_reason"] or None,
                temporal_alignment=row["temporal_alignment"],
            )
            records[record.checkpoint][row["record_kind"]].append(record)
        statuses = defaultdict(list)
        checkpoint_flags = {}
        for row in status_rows:
            checkpoint = int(row["checkpoint"])
            statuses[checkpoint].append(
                SourceControlBatch(
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    batch=None,
                    eligible=self._bool(row["eligible"]),
                    reason=row["reason"] or None,
                    valid_transition_count=int(row["valid_transition_count"]),
                    evolutionary_fitness=self._optional_float(
                        row["evolutionary_fitness"]
                    ),
                )
            )
            checkpoint_flags[checkpoint] = (
                self._bool(row["checkpoint_valid"]),
                self._bool(row["measurement_eligible"]),
                self._bool(row["comparison_eligible"]),
            )
        checkpoints = sorted(set(records) | set(statuses))
        self._diagnostics = []
        for checkpoint in checkpoints:
            summary = summaries.get(checkpoint, {})
            valid, measurement, comparison = checkpoint_flags.get(
                checkpoint, (True, True, False)
            )
            checkpoint_records = records[checkpoint]
            temporal_alignment = (
                checkpoint_records["formal"][0].temporal_alignment
                if checkpoint_records["formal"]
                else summary.get("temporal_alignment", "unknown")
            )
            source_statuses = tuple(statuses[checkpoint])
            self._diagnostics.append(
                CheckpointDiagnostic(
                    checkpoint=checkpoint,
                    temporal_alignment=temporal_alignment,
                    records=tuple(checkpoint_records["formal"]),
                    zero_step_results=tuple(checkpoint_records["zero_step"]),
                    valid=valid,
                    reasons=tuple(filter(None, summary.get("reasons", "").split("|"))),
                    warnings=tuple(
                        filter(None, summary.get("warnings", "").split("|"))
                    ),
                    measurement_eligible=measurement,
                    comparison_eligible=comparison,
                    eligibility_reasons=tuple(
                        sorted({s.reason for s in source_statuses if s.reason})
                    ),
                    source_statuses=source_statuses,
                )
            )
        self._checkpoints = set(checkpoints)
        self.run_config = existing_config

    def _replace_atomic(self, temp, target):
        temp.replace(target)

    def _atomic_text(self, name, text):
        target = self.root / name
        temp = self.root / f".{name}.tmp"
        try:
            with temp.open("w", encoding="utf-8", newline="") as h:
                h.write(text)
                h.flush()
                os.fsync(h.fileno())
            self._replace_atomic(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    def _atomic_json(self, name, value):
        self._atomic_text(name, json.dumps(value, indent=2, sort_keys=True) + "\n")

    def _append(self, name, header, rows):
        with (self.root / name).open("a", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=header, lineterminator="\n")
            for r in rows:
                w.writerow({k: r.get(k) for k in header})
            h.flush()
            os.fsync(h.fileno())

    def append_checkpoint(self, d):
        if d.checkpoint in self._checkpoints:
            raise ValueError(f"duplicate checkpoint append: {d.checkpoint}")
        byid = {s.source_id: s for s in d.source_statuses}
        raw = []
        for kind, records in (
            ("formal", d.records),
            ("zero_step", d.zero_step_results),
        ):
            for r in records:
                s = byid.get(r.source_id)
                row = {k: getattr(r, k, None) for k in PER_SOURCE_HEADER}
                row.update(
                    record_kind=kind,
                    valid_transition_count=None
                    if s is None
                    else s.valid_transition_count,
                    eligible=r.eligible if s is None else s.eligible,
                    ineligible_reason=r.ineligible_reason if s is None else s.reason,
                )
                raw.append(row)
        self._append("per_source_utility.csv", PER_SOURCE_HEADER, raw)
        statuses = (
            {
                "checkpoint": d.checkpoint,
                "source_type": s.source_type,
                "source_id": s.source_id,
                "eligible": s.eligible,
                "reason": s.reason,
                "valid_transition_count": s.valid_transition_count,
                "evolutionary_fitness": s.evolutionary_fitness,
                "measurement_eligible": d.measurement_eligible,
                "comparison_eligible": d.comparison_eligible,
                "checkpoint_valid": d.valid,
            }
            for s in sorted(
                d.source_statuses,
                key=lambda x: (SOURCE_ORDER.get(x.source_type, 99), x.source_id),
            )
        )
        self._append("source_status.csv", SOURCE_STATUS_HEADER, statuses)
        self._diagnostics.append(d)
        self._checkpoints.add(d.checkpoint)

    def _checkpoint_rows(self):
        out = []
        for d in sorted(self._diagnostics, key=lambda x: x.checkpoint):
            count = lambda t: sum(s.source_type == t for s in d.source_statuses)
            eligible = lambda t: sum(
                s.source_type == t and s.eligible for s in d.source_statuses
            )
            rank = compute_ea_rank_metrics(d)
            out.append(
                {
                    "checkpoint": d.checkpoint,
                    "valid": d.valid,
                    "measurement_eligible": d.measurement_eligible,
                    "comparison_eligible": d.comparison_eligible,
                    "checkpoint_status": _status(d),
                    "temporal_alignment": d.temporal_alignment,
                    "reasons": _join(d.reasons),
                    "eligibility_reasons": _join(d.eligibility_reasons),
                    "warnings": _join(d.warnings),
                    "num_ea_sources": count("EA"),
                    "num_eligible_ea": eligible("EA"),
                    "rl_own_eligible": bool(eligible("RL-own")),
                    "num_replay_random": count("Replay-random"),
                    "num_eligible_replay_random": eligible("Replay-random"),
                    "num_reference_self": count("Reference-self"),
                    "num_eligible_reference_self": eligible("Reference-self"),
                    "formal_record_count": len(d.records),
                    "zero_step_record_count": len(d.zero_step_results),
                    "ea_spearman_rho": rank["spearman_rho"],
                    "ea_kendall_tau_b": rank["kendall_tau_b"],
                    "ea_comparable_pair_count": rank["comparable_pair_count"],
                    "ea_inversion_count": rank["inversion_count"],
                    "ea_inversion_rate": rank["inversion_rate"],
                    "top_f_utility_rank": rank["top_f_utility_rank"],
                    "top_f_equals_top_u": rank["top_f_equals_top_u"],
                    "num_top_f_ties": rank["num_top_f_ties"],
                    "num_top_u_ties": rank["num_top_u_ties"],
                    "rank_metric_reason": rank["rank_metric_reason"],
                }
            )
        return out

    def _aggregate(self, complete):
        groups = defaultdict(list)
        for d in self._diagnostics:
            if not (
                d.valid
                and d.measurement_eligible
                and (not complete or d.comparison_eligible)
            ):
                continue
            for r in d.records:
                groups[(r.source_type, float(r.step_scale))].append((d.checkpoint, r))
        out = []
        for key, items in sorted(
            groups.items(), key=lambda x: (SOURCE_ORDER.get(x[0][0], 99), x[0][1])
        ):
            vals = np.array([float(r.utility) for _, r in items])
            tol = np.array([float(r.numerical_tolerance) for _, r in items])
            out.append(
                {
                    "source_type": key[0],
                    "step_scale": key[1],
                    "n_checkpoints": len({c for c, _ in items}),
                    "n_sources": len({(c, r.source_id) for c, r in items}),
                    "n_records": len(items),
                    "utility_mean": float(vals.mean()),
                    "utility_median": float(np.median(vals)),
                    "negative_rate": float(np.mean(vals < -tol)),
                    "near_zero_rate": float(np.mean(np.abs(vals) <= tol)),
                }
            )
        return out

    def _references(self):
        out = []
        expected = int(self.run_config.get("num_reference_batches", 4))
        for d in sorted(self._diagnostics, key=lambda x: x.checkpoint):
            if not (d.valid and d.measurement_eligible):
                continue
            groups = defaultdict(list)
            for r in d.records:
                groups[(r.source_type, r.source_id, float(r.step_scale))].append(r)
            for key, rs in sorted(
                groups.items(),
                key=lambda x: (SOURCE_ORDER.get(x[0][0], 99), x[0][1], x[0][2]),
            ):
                if len(rs) != expected:
                    continue
                a = aggregate_reference_utilities(
                    (float(r.utility) for r in rs),
                    max(float(r.numerical_tolerance) for r in rs),
                )
                out.append(
                    {
                        "checkpoint": d.checkpoint,
                        "source_type": key[0],
                        "source_id": key[1],
                        "step_scale": key[2],
                        "utility_mean": a.mean,
                        "utility_median": a.median,
                        "utility_std": a.std,
                        "negative_fraction": a.negative_fraction,
                        "near_zero_fraction": a.near_zero_fraction,
                        "sign_consistency": a.sign_consistency,
                        "all_near_zero": a.all_near_zero,
                    }
                )
        return out

    def _classification(self, cross, step):
        invalid = [d for d in self._diagnostics if not d.valid]
        if invalid:
            return classify_cases(
                {},
                {},
                {
                    "valid": False,
                    "reasons": sorted({r for d in invalid for r in d.reasons}),
                },
                self.run_config,
            )
        complete = [
            d
            for d in self._diagnostics
            if d.valid and d.measurement_eligible and d.comparison_eligible
        ]
        if not complete:
            return classify_cases(
                {},
                {},
                {"measurement_eligible": False, "comparison_eligible": False},
                self.run_config,
            )
        rates = {
            r["source_type"]: r["negative_rate"]
            for r in cross
            if r["step_scale"] == 1.0
        }
        scales = {
            r["step_scale"]: r["negative_rate"]
            for r in step
            if r["source_type"] == "EA"
        }
        return classify_cases(rates, scales, {}, self.run_config)

    def _report(self, c, checks, cross, refs):
        reasons = _join(r for d in self._diagnostics for r in d.reasons) or "None"
        warnings = _join(w for d in self._diagnostics for w in d.warnings) or "None"
        descriptions = []
        if c["case_a"]:
            descriptions.append(
                "Evidence is consistent with an EA-source / learner mismatch."
            )
        if c["case_b"]:
            descriptions.append(
                "All primary sources descriptively match the configured high-negative-rate band."
            )
        if c["case_c"]:
            descriptions.append(
                "The checkpoint evidence is invalid for formal summaries."
            )
        if c["case_d"]:
            descriptions.append(
                "The scale response is consistent with monotonic step sensitivity."
            )
        if not descriptions:
            descriptions.append("No configured descriptive case matched.")
        case_description = " ".join(descriptions)
        return f"""# Learner-Utility Source-Control Diagnostic

## Run provenance
Run ID: `{self.root.name}`

## Checkpoint status counts
Total checkpoints: {len(checks)}

## Source eligibility counts
See `source_status.csv` for source-local eligibility and reasons.

## Four-source negative-rate summary
Formal complete-checkpoint rows: {len(cross)}

## Step-scale sensitivity
Case D matched: {c['case_d']}

## Reference stability
Stable source/scale rows: {len(refs)}

## EA Fitness-vs-Utility rank diagnostics
Utility is the median across references at step scale 1.0.

## Case A-D descriptive classification
Classification: {c['classification']}
Classification ambiguous: {c['classification_ambiguous']}
{case_description}

## Warnings / invalid reasons
Warnings: {warnings}
Invalid reasons: {reasons}

## Interpretation limits
Utility is one-step frozen-critic learner utility. It is not actor return,
not long-term policy improvement, and not a causal conclusion.
"""

    def finalize_run(self):
        checks = self._checkpoint_rows()
        cross = self._aggregate(True)
        step = self._aggregate(False)
        refs = self._references()
        classification = self._classification(cross, step)
        for name, header, rows in (
            ("checkpoint_summary.csv", CHECKPOINT_SUMMARY_HEADER, checks),
            ("cross_source_summary.csv", CROSS_SOURCE_HEADER, cross),
            ("step_scale_summary.csv", STEP_SCALE_HEADER, step),
            ("reference_stability.csv", REFERENCE_STABILITY_HEADER, refs),
        ):
            self._atomic_text(name, _csv(header, rows))
        self._atomic_text(
            "diagnostic_report.md", self._report(classification, checks, cross, refs)
        )
        return classification
