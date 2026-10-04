from pathlib import Path

import pytest

from evorl.algorithms.erl.erl_td3.utility_source_control_artifacts import (
    UtilitySourceControlArtifactWriter,
    validate_artifact_root,
)


def test_allowed_root_and_child_are_accepted(tmp_path):
    allowed = tmp_path / "outputs"
    assert validate_artifact_root(allowed, allowed, ()) == allowed.resolve()
    child = allowed / "smoke" / "run-1"
    assert validate_artifact_root(child, allowed, ()) == child.resolve()


@pytest.mark.parametrize(
    "candidate_name",
    ["repo", "legacy", "replay-pilot"],
)
def test_forbidden_roots_are_rejected(tmp_path, candidate_name):
    allowed = tmp_path / "outputs"
    forbidden = tuple(
        tmp_path / name for name in ("repo", "legacy", "replay-pilot")
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_artifact_root(tmp_path / candidate_name, allowed, forbidden)


def test_relative_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        validate_artifact_root(Path("relative/run"), tmp_path / "outputs", ())


def test_dot_dot_escape_is_rejected(tmp_path):
    allowed = tmp_path / "outputs"
    with pytest.raises(ValueError, match="allowed output root"):
        validate_artifact_root(allowed / ".." / "escape", allowed, ())


def test_symlink_escape_is_rejected(tmp_path):
    allowed = tmp_path / "outputs"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="allowed output root"):
        validate_artifact_root(link / "run", allowed, ())


def test_invalid_artifact_root_fails_before_collector_or_environment_call(tmp_path):
    calls = []
    with pytest.raises(ValueError):
        UtilitySourceControlArtifactWriter(
            tmp_path / "escape",
            allowed_root=tmp_path / "outputs",
            forbidden_roots=(),
            run_config={"seed": 42},
        )
        calls.append("collector")
    assert calls == []
    assert not (tmp_path / "escape").exists()
