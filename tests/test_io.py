"""AC-04: read the example project.yaml and standard CSVs; reject bad CSVs with file name and line number."""

import shutil

import numpy as np
import pytest

from enzopt.core.dataio import DataError, convert, read_series, read_temperature
from enzopt.core.project import ProjectError, load_project
from enzopt.core.yamlio import load_yaml


def test_example_project_yaml(example_dir):
    p = load_project(example_dir)
    assert p.name == "example_ldh_synth"
    assert len(p.network.species) == 9 and len(p.network.steps) == 5
    assert p.Kp == 23.0 and p.haldane_step == 5
    assert [e.name for e in p.by_role("train")] == ["A", "B"]
    assert [e.name for e in p.by_role("holdout")] == ["C", "Dp"]
    assert [e.name for e in p.by_role("design-check")] == ["D1", "D0"]
    a = p.experiments["A"]
    assert a.initial == {"NADH": 0.2, "NAD": 0.0, "Pyr": 0.2, "Lac": 9.8}
    assert a.Et_nM == pytest.approx(24.23) and a.Et == pytest.approx(24.23e-6)
    assert p.observable_weights() == {"NADH": 6.22, "EN": 6.22, "ENP": 6.22}
    assert p.design["dead_end"]["Ki_mM"] == 5.0


def test_vendored_yaml_matches_pyyaml(example_dir):
    a = load_yaml(example_dir / "project.yaml")
    b = load_yaml(example_dir / "project.yaml", force_vendored=True)
    assert a == b
    p = load_project(example_dir, force_vendored_yaml=True)
    assert len(p.experiments) == 6


def test_read_standard_csv(example_dir):
    p = load_project(example_dir)
    for e in p.experiments.values():
        s = read_series(e.file)
        assert len(s.t) >= 60 and np.all(np.diff(s.t) > 0)
        assert s.y.max() <= 2.0
        t = read_temperature(e.temp_file)
        assert abs(t.y.mean() - 25.0) < 0.05


def _write(tmp_path, name, text):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


@pytest.mark.parametrize(
    "body, line, what",
    [
        ("time_s,A340\n0,1.2\n5,1.1\n3,1.0\n10,0.9\n", 4, "시간 역행"),
        ("time_s,A340\n0,1.2\n5,\n10,0.9\n", 3, "빈 값"),
        ("time_s,A340\n0,1.2\n5,1.1\n10,2.5\n", 4, "A340"),
        ("time_s,A340\n0,1.2\n5,abc\n", 3, "숫자가 아님"),
    ],
)
def test_reject_bad_csv(tmp_path, body, line, what):
    f = _write(tmp_path, "setX.csv", body)
    with pytest.raises(DataError) as ei:
        read_series(f)
    msg = str(ei.value)
    assert "setX.csv" in msg
    assert f"{line}번째 줄" in msg
    assert what in msg


def test_bad_csv_stops_run_at_data_stage(tmp_path, example_dir):
    """A bad experiment file stops the run in stage ② with the file and line in the error."""
    from enzopt.pipeline.runner import read_json, run_project

    proj = tmp_path / "projects" / "bad"
    shutil.copytree(example_dir, proj)
    (proj / "data" / "setC.csv").write_text("time_s,A340\n5,0.6\n15,0.61\n10,0.62\n", encoding="utf-8")
    rd = run_project(proj, quick=True, runs_root=tmp_path / "runs", stop_after="2_data")
    m = read_json(rd / "manifest.json")
    assert m["status"] == "failed"
    assert m["stages"]["2_data"]["status"] == "failed"
    assert "setC.csv" in m["error"]["message"] and "4번째 줄" in m["error"]["message"]


def test_missing_header_and_project_errors(tmp_path, example_dir):
    f = _write(tmp_path, "noheader.csv", "0,1.2\n5,1.1\n")
    with pytest.raises(DataError, match="머리글"):
        read_series(f)
    proj = tmp_path / "p"
    shutil.copytree(example_dir, proj)
    text = (proj / "project.yaml").read_text(encoding="utf-8").replace("Et_nM: 24.23, role: train}", "role: train}", 1)
    (proj / "project.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ProjectError, match="Et_nM"):
        load_project(proj)


def test_generic_converter(tmp_path):
    raw = _write(
        tmp_path,
        "export.txt",
        "Instrument X\nTime (min)\tAbs\n0,0\t1,2440\n0,5\t1,2000\n1,0\t1,1600\n\nEnd of data\n",
    )
    t, y = convert(raw, {"delimiter": "tab", "decimal": ",", "skip_rows": 2, "time_unit": "min"})
    assert np.allclose(t, [0, 30, 60]) and np.allclose(y, [1.244, 1.2, 1.16])
