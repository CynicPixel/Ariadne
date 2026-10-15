"""phase1_gate's comparison rules (design PHASE1_GATE): byte identity, named line exclusions, nothing silent."""
from pathlib import Path

import pytest

from analysis.phase1_gate import Gate, check_determinism, compare_file


def write(p: Path, s: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)
    return p


def test_identical_bytes(tmp_path):
    a, b = write(tmp_path / "a.yaml", "x: 1\n"), write(tmp_path / "b.yaml", "x: 1\n")
    assert compare_file(a, b, [])["status"] == "identical"


def test_a_difference_without_exclusions_is_reported_with_its_line(tmp_path):
    a, b = write(tmp_path / "a.yaml", "x: 1\ny: 2\n"), write(tmp_path / "b.yaml", "x: 1\ny: 3\n")
    r = compare_file(a, b, [])
    assert r["status"] == "DIFFERENT" and r["first_diff"].startswith("line 2")


def test_only_the_named_lines_are_excluded(tmp_path):
    a = write(tmp_path / "a.yaml", "run_id: A\nx: 1\n")
    b = write(tmp_path / "b.yaml", "run_id: B\nx: 1\n")
    r = compare_file(a, b, [{"re": "^run_id: ", "why": "run_identity"}])
    assert r["status"] == "identical_except" and r["lines_excluded"] == 1
    c = write(tmp_path / "c.yaml", "run_id: B\nx: 2\n")
    assert compare_file(a, c, [{"re": "^run_id: ", "why": "run_identity"}])["status"] == "DIFFERENT"


def test_missing_files_fail(tmp_path):
    a = write(tmp_path / "a.yaml", "x\n")
    assert compare_file(tmp_path / "nope", a, [])["status"] == "MISSING"
    assert compare_file(a, tmp_path / "nope", [])["status"] == "MISSING_REFERENCE"


def test_an_exclusion_without_a_named_reason_is_refused(tmp_path):
    cfg = {"reference_root": str(tmp_path), "reference": {"k": "ref"}, "reasons": {"timing": "wall clock"},
           "compare": {"k": [{"file": "a.yaml", "exclude": [{"re": "^t: ", "why": "convenience"}]}]},
           "checksums": []}
    with pytest.raises(SystemExit, match="no named reason"):
        check_determinism(Gate(), cfg, {"k": tmp_path / "run"}, tmp_path)
