"""Host-side artifact safety and writing for source-control diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def validate_artifact_root(
    candidate: str | Path,
    allowed_root: Path,
    forbidden_roots: Sequence[Path],
) -> Path:
    """Return a canonical output path after rejecting every path escape."""
    candidate = Path(candidate)
    if not candidate.is_absolute():
        raise ValueError("artifact root must be absolute")

    resolved_candidate = candidate.resolve(strict=False)
    resolved_allowed = Path(allowed_root).resolve(strict=False)
    resolved_forbidden = tuple(
        Path(path).resolve(strict=False) for path in forbidden_roots
    )
    if any(
        resolved_candidate == forbidden
        or resolved_candidate.is_relative_to(forbidden)
        for forbidden in resolved_forbidden
    ):
        raise ValueError("artifact root resolves inside a forbidden root")
    if not (
        resolved_candidate == resolved_allowed
        or resolved_candidate.is_relative_to(resolved_allowed)
    ):
        raise ValueError("artifact root must resolve under the allowed output root")
    return resolved_candidate


class UtilitySourceControlArtifactWriter:
    """Own host I/O after validating the exclusive run directory."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        allowed_root: Path,
        forbidden_roots: Sequence[Path],
        run_config: Mapping[str, Any],
    ) -> None:
        self.root = validate_artifact_root(
            artifact_root,
            allowed_root=allowed_root,
            forbidden_roots=forbidden_roots,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._atomic_json("run_config.json", dict(run_config))

    def _atomic_json(self, filename: str, value: Mapping[str, Any]) -> None:
        target = self.root / filename
        temporary = self.root / f".{filename}.tmp"
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
