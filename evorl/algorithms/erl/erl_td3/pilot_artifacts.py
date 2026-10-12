"""Pilot-only writer; never targets formal_v1 or legacy Source Controls output."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PILOT_ROOT = Path("/disk1/zmh_projects/evorl_utility_source_controls_outputs/reference_target_bias_pilot")


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

    def append_checkpoint(self, *, generation, status, reasons, source_results, aggregates, provenance):
        self.root.mkdir(parents=True, exist_ok=True)
        rows = []
        zero_rows = []
        for source_id, result in source_results.items():
            for record, target in ((r, rows) for r in result.records):
                target.append({
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
                zero_rows.append({
                    "generation": generation, "source_id": source_id,
                    "reference_family": record.reference_family,
                    "reference_index": record.reference_index,
                    "step_scale": 0.0, "loss_ref_before": record.loss_ref_before,
                    "loss_ref_after": record.loss_ref_after, "utility": record.utility,
                    "source_update_fingerprint": result.update_fingerprint,
                })
        header = ("generation", "source_id", "reference_family", "reference_index",
                  "step_scale", "loss_ref_before", "loss_ref_after", "utility",
                  "source_update_fingerprint")
        for name, values in (("reference_level.csv", rows), ("zero_step.csv", zero_rows)):
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
