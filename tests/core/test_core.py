"""AC-10: core modules (verdict, scipy simulation, validation, design, YAML reading) in the KIMM-CAD compatible env.

Run there as ``PYTHONPATH=src python -m pytest tests/core`` (Python 3.12, numpy 1.26.4, scipy 1.13.1,
matplotlib 3.9.1, no torch/streamlit/PyYAML). The same tests also run in the main environment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "projects" / "example_ldh_synth"


def test_core_imports_without_heavy_packages():
    """Importing the core must not pull in torch or streamlit (absent in the KIMM-CAD Python)."""
    import os
    import subprocess

    code = (
        "import sys; import enzopt.core.crnt, enzopt.core.massaction, enzopt.core.validate, enzopt.core.design, "
        "enzopt.core.project, enzopt.core.dataio, enzopt.core.regression; "
        "print([m for m in ('torch', 'streamlit') if m in sys.modules])"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout + r.stderr


def test_core_yaml_project():
    from enzopt.core.project import load_project

    p = load_project(EXAMPLE, force_vendored_yaml=True)
    assert len(p.experiments) == 6 and p.Kp == 23.0
    try:
        import yaml  # noqa: F401

        has_pyyaml = True
    except ImportError:
        has_pyyaml = False
    p2 = load_project(EXAMPLE)  # PyYAML if installed, otherwise the vendored copy
    assert p2.raw == p.raw
    assert isinstance(has_pyyaml, bool)


def test_core_verdict():
    from enzopt.core.crnt import verdict
    from enzopt.core.project import load_project

    p = load_project(EXAMPLE)
    v = verdict(p.network.reactions())
    assert v.weakly_reversible and v.deficiency == 0 and len(v.linkage_classes) == 3 and len(v.conservation_laws) == 4
    d = verdict(p.design_network().reactions())
    assert d.weakly_reversible and len(d.linkage_classes) == 4
    assert not verdict(p.control_network().reactions()).weakly_reversible


def test_core_simulate_and_validate_batch149():
    from enzopt.core.regression import batch149_scipy

    res = batch149_scipy()
    assert res["n_pass"] == 13
    assert res["same_class_final_NADH"] == pytest.approx(0.1485, abs=5e-4)
    assert res["same_class_spread"] < 1e-4
    assert res["other_class_final_NADH"] == pytest.approx(0.1019, abs=5e-4)


def test_core_full_mechanism_simulation():
    from enzopt.core.massaction import MassAction, SolverSettings, atol_vector, haldane_kr, simulate
    from enzopt.core.network import x0_vector
    from enzopt.core.project import load_project
    from enzopt.core.validate import validate_closed

    p = load_project(EXAMPLE)
    net = p.network
    kf = np.array([1e4, 5e3, 300.0, 1000.0, 200.0])
    kr = haldane_kr(kf, np.array([50.0, 500.0, 1000.0, 200.0, 1.0]), 23.0, 5)
    k = net.rate_vector(kf, kr)
    Et = 24.23e-6
    t = np.linspace(0, 3000, 301)
    st = SolverSettings()
    X = simulate(
        MassAction.from_network(net),
        k,
        x0_vector(net, p.experiments["A"].initial, Et),
        t,
        atol=atol_vector(net.species, net.enzyme_species, Et, st),
        settings=st,
    )
    S = {s: X[:, i] for i, s in enumerate(net.species)}
    r = validate_closed(t, S, net.reactions(), k)
    assert r["ok"], r
    assert X[-1, net.index()["NADH"]] == pytest.approx(0.1485, abs=5e-4)


def test_core_design_quick():
    import json

    from enzopt.core.regression import design_regression

    with open(ROOT / "evals" / "baselines.json", encoding="utf-8") as fh:
        ref = json.load(fh)["design"]["quick"]
    res = design_regression(quick=True)
    assert res["rho_star"] == pytest.approx(ref["rho_star"])
    assert res["robust_Et_star_nM"] == pytest.approx(ref["robust_Et_star_nM"], rel=0.02)
    assert res["nominal_fail"] == ref["nominal_fail"] and res["robust_fail"] == 0
