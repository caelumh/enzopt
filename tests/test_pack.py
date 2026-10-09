"""AC-14: `enzopt pack <run folder>` bundles input data, settings copy, figures, CSVs, Hopsan .hmf models
and validation reports."""

import zipfile

from conftest import run_cli


def test_pack(quick_run, tmp_path):
    out = tmp_path / "bundle.zip"
    r = run_cli("pack", quick_run, "-o", out)
    assert r.returncode == 0, r.stdout + r.stderr
    with zipfile.ZipFile(out) as z:
        names = [n.split("/", 1)[1] for n in z.namelist()]
    assert "project.yaml" in names and "manifest.json" in names
    assert "inputs/project.yaml" in names
    assert any(n.startswith("inputs/data/") and n.endswith(".csv") for n in names)  # input data
    assert any(n.endswith(".pdf") for n in names) and any(n.endswith(".png") for n in names)  # figures
    assert sum(n.endswith(".csv") for n in names) > 20  # result and figure CSVs
    assert any(n.endswith(".hmf") for n in names)  # Hopsan models
    assert "4_sim/validation.csv" in names and "4_sim/sim_summary.json" in names  # validation report
    assert "1_verdict/report.txt" in names
    assert "5_design/pipette.csv" in names
    assert "PACK_CONTENTS.json" in names
