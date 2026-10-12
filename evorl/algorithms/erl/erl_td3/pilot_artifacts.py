"""Pilot-only artifact writing; no legacy or formal_v1 file can be targeted."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

PILOT_ROOT = Path("/disk1/zmh_projects/evorl_utility_source_controls_outputs/reference_target_bias_pilot")

FAILURE_HEADER = (
    "run_id", "seed", "checkpoint", "source_id", "source_type",
    "reason", "source_batch_fingerprint", "frozen_critic_fingerprint",
)

MECHANISM_HEADER = (
    "run_id", "seed", "checkpoint", "source_id", "source_type",
    "reference_family", "reference_id", "step_scale",
    "source_gradient_norm", "reference_gradient_norm",
    "raw_gradient_cosine", "optax_update_norm",
    "actual_parameter_displacement_norm", "negative_update_reference_cosine",
    "first_order_predicted_utility", "actual_utility", "first_order_residual",
    "cosine_status", "raw_cosine_status", "update_cosine_status",
    "numerical_status", "eligible", "ineligible_reason",
    "loss_ref_before", "frozen_critic_fingerprint",
    "source_gradient_fingerprint", "source_update_fingerprint",
    "reference_gradient_fingerprint", "reference_eval_key_fingerprint",
    "reference_fingerprint",
)


def _mechanism_key(row):
    return (
        str(row["run_id"]), str(row["checkpoint"]), str(row["source_id"]),
        str(row["reference_family"]), str(row["reference_id"]),
        str(row["step_scale"]),
    )


class PilotArtifactWriter:
    def __init__(self, root, run_id):
        allowed = PILOT_ROOT.resolve(strict=False)
        root = Path(root).resolve(strict=False)
        if root != allowed and not root.is_relative_to(allowed):
            raise ValueError("Pilot artifacts must stay under reference_target_bias_pilot")
        if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
            raise ValueError("invalid pilot run ID")
        destination = (root / run_id).resolve(strict=False)
        if destination == allowed or not destination.is_relative_to(allowed):
            raise ValueError("Pilot run path must stay under reference_target_bias_pilot")
        self.root = destination
        self.run_id = run_id

    def append_checkpoint(
        self, *, generation, status, reasons, source_results, aggregates,
        provenance, seed=None, source_types=None, frozen_critic_fingerprint=None,
        require_mechanisms=False, failed_sources=(),
    ):
        source_types = source_types or {}
        rows, zero_rows, mechanisms = [], [], []
        raw_header = (
            "generation", "source_id", "reference_family", "reference_index",
            "step_scale", "loss_ref_before", "loss_ref_after", "utility",
            "source_update_fingerprint",
        )
        for source_id, result in source_results.items():
            raw = {}
            for record in result.records:
                key = (record.reference_family, record.reference_index, record.step_scale)
                if key in raw:
                    raise ValueError("duplicate_utility_key")
                raw[key] = record
                rows.append({
                    "generation": generation, "source_id": source_id,
                    "reference_family": record.reference_family,
                    "reference_index": record.reference_index,
                    "step_scale": record.step_scale,
                    "loss_ref_before": record.loss_ref_before,
                    "loss_ref_after": record.loss_ref_after,
                    "utility": record.utility,
                    "source_update_fingerprint": result.update_fingerprint,
                })
            for record in result.zero_step_results:
                key = (record.reference_family, record.reference_index, 0.0)
                if key in raw:
                    raise ValueError("duplicate_utility_key")
                raw[key] = record
                zero_rows.append({
                    "generation": generation, "source_id": source_id,
                    "reference_family": record.reference_family,
                    "reference_index": record.reference_index,
                    "step_scale": 0.0, "loss_ref_before": record.loss_ref_before,
                    "loss_ref_after": record.loss_ref_after, "utility": record.utility,
                    "source_update_fingerprint": result.update_fingerprint,
                })
            if require_mechanisms and not result.mechanism_records:
                raise ValueError("mechanism_utility_key_mismatch")
            if result.mechanism_records:
                measured = {
                    (record.reference_family, record.reference_index, record.step_scale): record
                    for record in result.mechanism_records
                }
                if len(measured) != len(result.mechanism_records) or set(measured) != set(raw):
                    raise ValueError("mechanism_utility_key_mismatch")
                for key, mechanism in measured.items():
                    utility = raw[key]
                    if not np.isclose(
                        mechanism.actual_utility, utility.utility, rtol=0,
                        atol=0, equal_nan=True,
                    ):
                        raise ValueError("mechanism_utility_value_mismatch")
                    if mechanism.source_update_fingerprint != result.update_fingerprint:
                        raise ValueError("source_update_fingerprint_mismatch")
                    if (frozen_critic_fingerprint is not None
                            and mechanism.frozen_critic_fingerprint
                            != frozen_critic_fingerprint):
                        raise ValueError("frozen_critic_fingerprint_mismatch")
                    mechanisms.append({
                        "run_id": self.run_id, "seed": seed,
                        "checkpoint": generation, "source_id": source_id,
                        "source_type": source_types.get(source_id),
                        "reference_family": mechanism.reference_family,
                        "reference_id": f"reference_{mechanism.reference_index}",
                        "step_scale": mechanism.step_scale,
                        "source_gradient_norm": mechanism.source_gradient_norm,
                        "reference_gradient_norm": mechanism.reference_gradient_norm,
                        "raw_gradient_cosine": mechanism.raw_gradient_cosine,
                        "optax_update_norm": mechanism.optax_update_norm,
                        "actual_parameter_displacement_norm": mechanism.actual_parameter_displacement_norm,
                        "negative_update_reference_cosine": mechanism.negative_update_reference_cosine,
                        "first_order_predicted_utility": mechanism.first_order_predicted_utility,
                        "actual_utility": mechanism.actual_utility,
                        "first_order_residual": mechanism.first_order_residual,
                        "cosine_status": mechanism.cosine_status,
                        "raw_cosine_status": mechanism.raw_cosine_status,
                        "update_cosine_status": mechanism.update_cosine_status,
                        "numerical_status": mechanism.numerical_status,
                        "eligible": mechanism.eligible,
                        "ineligible_reason": mechanism.ineligible_reason,
                        "loss_ref_before": mechanism.loss_ref_before,
                        "frozen_critic_fingerprint": mechanism.frozen_critic_fingerprint,
                        "source_gradient_fingerprint": mechanism.source_gradient_fingerprint,
                        "source_update_fingerprint": mechanism.source_update_fingerprint,
                        "reference_gradient_fingerprint": mechanism.reference_gradient_fingerprint,
                        "reference_eval_key_fingerprint": mechanism.reference_eval_key_fingerprint,
                        "reference_fingerprint": mechanism.reference_fingerprint,
                    })
        failure_rows = []
        for failed in failed_sources:
            if failed["source_id"] in source_results:
                raise ValueError("failed_source_has_utility")
            failure_rows.append({
                "run_id": self.run_id, "seed": seed, "checkpoint": generation,
                "source_id": failed["source_id"],
                "source_type": source_types.get(failed["source_id"]),
                "reason": failed["reason"],
                "source_batch_fingerprint": failed["source_batch_fingerprint"],
                "frozen_critic_fingerprint": frozen_critic_fingerprint,
            })
        if len({row["source_id"] for row in failure_rows}) != len(failure_rows):
            raise ValueError("duplicate_failed_source")
        mechanism_path = self.root / "pilot_gradient_mechanisms.csv"
        existing = set()
        if mechanism_path.exists():
            with mechanism_path.open(newline="") as handle:
                existing = {_mechanism_key(row) for row in csv.DictReader(handle)}
        keys = [_mechanism_key(row) for row in mechanisms]
        if len(set(keys)) != len(keys) or set(keys) & existing:
            raise ValueError("duplicate_mechanism_key")
        self.root.mkdir(parents=True, exist_ok=True)
        for name, header, values in (
            ("reference_level.csv", raw_header, rows),
            ("zero_step.csv", raw_header, zero_rows),
            ("pilot_gradient_mechanisms.csv", MECHANISM_HEADER, mechanisms),
            ("pilot_gradient_failures.csv", FAILURE_HEADER, failure_rows),
        ):
            if name in ("pilot_gradient_mechanisms.csv", "pilot_gradient_failures.csv") and not values:
                continue
            path = self.root / name
            needs_header = not path.exists()
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                if needs_header:
                    writer.writeheader()
                writer.writerows(values)
        with (self.root / "paired_summary.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "generation": generation, "status": status, "reasons": list(reasons),
                "aggregates": aggregates, "provenance": provenance,
            }, sort_keys=True, default=str) + "\n")
