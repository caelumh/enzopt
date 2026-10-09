"""AC-04: read the example project.yaml and standard CSVs; reject bad CSVs with file name and line number."""

import shutil
from pathlib import Path

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


def _proj_copy(tmp_path, example_dir, name="p"):
    proj = tmp_path / name
    shutil.copytree(example_dir, proj)
    return proj


def test_temperature_in_same_file_and_time_offset(tmp_path, example_dir):
    from enzopt.core.dataio import load_experiment

    proj = _proj_copy(tmp_path, example_dir)
    t = np.arange(0.0, 60.0, 5.0)
    rows = "".join(f"{a:.1f},{1.2 - 0.001 * a:.4f},{25.0 + 0.01 * (i % 3):.2f}\n" for i, a in enumerate(t))
    (proj / "data" / "setA.csv").write_text("time_s,A340,T_C\n" + rows, encoding="utf-8")
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    y = y.replace(
        "{file: data/setA.csv,  temp_file: data/setA_temp.csv,  NADH: 0.2,",
        "{file: data/setA.csv,  time_offset_s: 5,  NADH: 0.2,",
    )
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    p = load_project(proj)
    e = p.experiments["A"]
    assert e.temp_file is None and e.time_offset_s == 5.0
    s, temp = load_experiment(e)
    assert temp is not None and len(temp.t) == len(t)
    assert s.t[0] == pytest.approx(5.0) and temp.t[0] == pytest.approx(5.0)


def test_converter_temperature_column(tmp_path):
    from enzopt.core.dataio import load_experiment
    from enzopt.core.project import Experiment

    raw = _write(tmp_path, "export.txt", "Kinetics\ntime\tabs\ttemp\n0\t1,24\t25,0\n0,5\t1,20\t25,1\n1,0\t1,16\t25,0\n")
    e = Experiment(
        "X",
        {"NADH": 0.2},
        24.2,
        "train",
        file=raw,
        format={"delimiter": "tab", "decimal": ",", "skip_rows": 2, "temp_col": 2, "time_unit": "min"},
    )
    s, temp = load_experiment(e, converted_dir=tmp_path)
    assert np.allclose(s.t, [0, 30, 60]) and np.allclose(s.y, [1.24, 1.2, 1.16])
    assert np.allclose(temp.y, [25.0, 25.1, 25.0])
    assert raw.read_text(encoding="utf-8").startswith("Kinetics")  # original untouched


def test_concentration_observable_column(tmp_path, example_dir):
    """A project may observe a concentration (column named after the observable); no A340 > 2 limit then."""
    from enzopt.core.dataio import load_experiment

    proj = _proj_copy(tmp_path, example_dir)
    (proj / "data" / "setA.csv").write_text("time_s,NADH_mM\n5,2.5\n10,2.4\n", encoding="utf-8")
    p = load_project(proj)
    s, _ = load_experiment(p.experiments["A"], value_column="NADH_mM")
    assert s.y[0] == 2.5
    with pytest.raises(DataError, match="머리글"):
        load_experiment(p.experiments["A"], value_column="A340")


def test_equilibrium_observable(example_dir):
    from enzopt.core.dataio import equilibrium_observable

    p = load_project(example_dir)
    assert equilibrium_observable(p, p.experiments["A"].initial) == pytest.approx(6.22 * 0.14851, rel=1e-3)
    assert equilibrium_observable(p, p.experiments["B"].initial) == pytest.approx(6.22 * 0.14851, rel=1e-3)
    assert equilibrium_observable(p, p.experiments["Dp"].initial) == pytest.approx(6.22 * 0.10188, rel=1e-3)


def test_check_command(tmp_path, example_dir):
    from conftest import run_cli

    r = run_cli("check", example_dir)
    assert r.returncode == 0, r.stdout[-2000:]
    assert "예상 평형" in r.stdout and "Dp" in r.stdout and "0.9237" in r.stdout
    proj = _proj_copy(tmp_path, example_dir)
    (proj / "data" / "setC.csv").unlink()
    r = run_cli("check", proj)
    assert r.returncode == 1
    assert "setC.csv" in r.stdout


def test_real_data_templates_load():
    root = Path(__file__).resolve().parents[1] / "projects"
    p = load_project(root / "ldh_ph10")
    assert len(p.by_role("train")) == 4 and len(p.by_role("holdout")) == 4
    h = load_project(root / "ldh_holmes1988")
    assert h.observable["name"] == "NADH_mM" and len(h.experiments) == 23
