"""Unit tests for scripts/run_lodo_ladder.py.

Subprocess execution is mocked — we test the orchestrator's stage planning,
non-overwrite guard, and manifest writing, not the underlying trainers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.run_lodo_ladder import (
    LodoPaths,
    StageResult,
    snapshot_protected_dirs,
    stages_for_holdout,
    verify_no_overwrite,
    write_manifest,
)


def test_lodo_paths_for_each_holdout() -> None:
    for h in ("ctu13", "iot23", "medbiot"):
        p = LodoPaths.for_holdout(h)
        assert p.holdout == h
        assert p.split_config == Path(f"configs/split_lodo_{h}.yaml")
        assert p.ckpt_root == Path("experiments/lodo") / h
        assert p.emb_dir == Path("data/flow_embeddings_lodo") / h
        assert p.manifest == Path("experiments/lodo") / h / "manifest.json"


def test_stages_for_holdout_count_matches_ladder() -> None:
    """Three seeds means: 5 baselines + phase5 + cache + phase6 + 3 phase7 seeds = 11 stages."""
    p = LodoPaths.for_holdout("medbiot")
    plan = stages_for_holdout(p, seeds=[42, 1, 2])
    labels = [s[0] for s in plan]
    assert len(plan) == 11
    assert "phase4_rf" in labels
    assert "phase4_xgb" in labels
    assert "phase4_gat" in labels
    assert "phase4_gin" in labels
    assert "phase4_gine" in labels
    assert "phase5" in labels
    assert "cache" in labels
    assert "phase6" in labels
    assert "phase7_seed42" in labels
    assert "phase7_seed1" in labels
    assert "phase7_seed2" in labels


def test_stages_for_holdout_uses_holdout_split_config() -> None:
    p = LodoPaths.for_holdout("iot23")
    plan = stages_for_holdout(p, seeds=[42])
    for label, cmd, seed in plan:
        if "phase4" in label or "phase5" in label or "phase6" in label or "phase7" in label or "cache" in label:
            # Normalise separators so the test passes on both Windows and POSIX.
            joined = " ".join(cmd).replace("\\", "/")
            assert "configs/split_lodo_iot23.yaml" in joined, \
                f"stage {label} should reference iot23 split config"


def test_verify_no_overwrite_returns_empty_when_unchanged() -> None:
    before = {"a": "2026-01-01T00:00:00Z", "b": "2026-01-02T00:00:00Z"}
    after = dict(before)
    assert verify_no_overwrite(before, after, "phase5") == []


def test_verify_no_overwrite_flags_modifications() -> None:
    before = {"a": "2026-01-01T00:00:00Z"}
    after = {"a": "2026-06-03T18:00:00Z"}
    issues = verify_no_overwrite(before, after, "phase5")
    assert len(issues) == 1
    assert "MODIFIED" in issues[0]


def test_verify_no_overwrite_flags_new_files() -> None:
    before: dict[str, str] = {}
    after = {"x": "2026-06-03T18:00:00Z"}
    issues = verify_no_overwrite(before, after, "cache")
    assert any("CREATED" in i for i in issues)


def test_verify_no_overwrite_flags_deletions() -> None:
    before = {"removed": "2026-01-01T00:00:00Z"}
    after: dict[str, str] = {}
    issues = verify_no_overwrite(before, after, "phase6")
    assert any("DELETED" in i for i in issues)


def test_write_manifest_produces_valid_json(tmp_path: Path) -> None:
    paths = LodoPaths(
        holdout="medbiot",
        split_config=Path("configs/split_lodo_medbiot.yaml"),
        ckpt_root=tmp_path,
        emb_dir=tmp_path / "emb",
        results_dir=tmp_path / "results",
        manifest=tmp_path / "manifest.json",
        log_dir=tmp_path / "logs",
    )
    results = [
        StageResult(name="phase4_rf", cmd=["uv", "run", "rf"],
                     rc=0, started_iso="t0", finished_iso="t1",
                     log_path=tmp_path / "logs/phase4_rf.log", duration_s=12.3),
    ]
    write_manifest(paths, results, seeds=[42, 1, 2])

    assert paths.manifest.exists()
    blob = json.loads(paths.manifest.read_text())
    assert blob["holdout"] == "medbiot"
    assert blob["seeds"] == [42, 1, 2]
    assert len(blob["stages"]) == 1
    assert blob["stages"][0]["name"] == "phase4_rf"
    assert blob["stages"][0]["rc"] == 0
