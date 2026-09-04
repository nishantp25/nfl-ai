"""Contracts for coaching data ownership and point-in-time semantics."""

from pathlib import Path

import pytest

from nfldl.manifest import load_manifest
from nfldl.registry import REGISTRY
from nfldl.sources import coaching


def test_coaching_staff_is_optional_curated_non_head_coach_data():
    spec = load_manifest()["coaching_staff"]

    assert spec.required is False
    assert spec.declared.start is None
    assert spec.declared.end is None
    assert spec.verified.start is None
    assert spec.verified.end is None

    inputs = spec.raw.get("inputs", [])
    assert inputs == ["configs/coaching_staff.csv"]


def test_coaching_staff_adapter_does_not_depend_on_schedules():
    assert REGISTRY["coaching_staff"].needs_local == ()


def test_curated_staff_rejects_head_coach_rows(
    tmp_path: Path,
    monkeypatch,
):
    csv_path = tmp_path / "coaching_staff.csv"
    csv_path.write_text(
        (
            "coach_name,team,role,start_date,end_date,interim,"
            "source_url,source_note\n"
            "Example Coach,KC,HC,2026-02-01,,false,"
            "https://example.com/source,head coach belongs to schedules"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        coaching,
        "CURATED_CSV",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="head coach",
    ):
        coaching.load_curated()
