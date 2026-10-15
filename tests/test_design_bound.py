"""design.rho_max: the optimum is taken inside an upper bound of ρ ("train" = largest partner of the training wells)."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_general_design import truth_samples

from enzopt.core.design import AssayDesign, rho_min_of, run_design
from enzopt.pipeline.design import rho_upper_bound
from enzopt.pipeline.runner import StageFailed


def test_rho_max_limits_the_optimum():
    net, tr, samples = truth_samples("ping_pong", n=3)
    d = AssayDesign(net, "A", "B", "E", 0.1, 0.99, 300.0, tr["Kp"][0], overall=tr["overall"][0])
    rmin = rho_min_of(d)
    rhos = [rmin * 1.1, rmin * 2, rmin * 4]
    free = run_design(d, samples, rhos, iters=14, Et_bounds=(1e-7, 1e-2), workers=1)
    bound = run_design(d, samples, rhos, iters=14, Et_bounds=(1e-7, 1e-2), workers=1, rho_max=rmin * 2)
    assert bound.Et_robust == free.Et_robust  # the whole curve is still computed
    allowed = [i for i, r in enumerate(rhos) if r <= rmin * 2 * (1 + 1e-9)]
    j = min(allowed, key=lambda i: free.Et_robust[i])
    assert bound.rho_star == pytest.approx(rhos[j])
    assert bound.Et_star == pytest.approx(free.Et_robust[j])
    assert bound.summary()["rho_max"] == pytest.approx(rmin * 2)
    assert bound.summary()["robust_fail"] == 0
    assert free.summary()["rho_max"] is None


def test_rho_upper_bound_train():
    exps = [SimpleNamespace(initial={"NADH": v, "Pyr": 0.5}) for v in (0.24, 0.458, 0.31)]
    p = SimpleNamespace(by_role=lambda role: exps if role == "train" else [])
    d = SimpleNamespace(partner="NADH", analyte="Pyr", analyte0=0.1)
    rho, why = rho_upper_bound(p, {"rho_max": "train"}, d)
    assert rho == pytest.approx(4.58) and "0.458" in why
    assert rho_upper_bound(p, {"rho_max": 6}, d) == (6.0, "설정값")
    assert rho_upper_bound(p, {"rho_max": None}, d) == (None, None)
    assert rho_upper_bound(p, {"rho_max": "train"}, SimpleNamespace(partner=None, analyte="S", analyte0=0.1)) == (
        None,
        None,
    )
    with pytest.raises(StageFailed):
        rho_upper_bound(p, {"rho_max": "data"}, d)
    assert np.isfinite(rho)
