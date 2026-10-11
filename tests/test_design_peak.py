"""Design goal ``peak``: stop when the product is highest (made and then broken down by the same enzyme)."""

import numpy as np
import pytest

from enzopt.core.design import PEAK_WINDOW, AssayDesign, peak_row, run_peak_design
from enzopt.core.network import Network

STEPS = ["E + S <=> ES", "ES -> E + P", "E + P <=> EP", "EP -> E + Q"]


def toy():
    net = Network.from_strings("toy", ["E", "S", "ES", "P", "EP", "Q"], STEPS, ["E", "ES", "EP"])
    design = AssayDesign(net, "S", None, "E", 1.0, 0.99, 600.0, 1.0, goal="peak", product="P")
    slow = net.rate_vector([100.0, 10.0, 100.0, 1.0], [50.0, 0.0, 50.0, 0.0])
    fast = net.rate_vector([100.0, 10.0, 100.0, 3.0], [50.0, 0.0, 50.0, 0.0])
    return design, [slow, fast]


def test_peak_row_robust_stop_maximises_the_worst_sample():
    t = np.linspace(0.0, 10.0, 101)
    Y = np.array([np.exp(-((t - 4.0) ** 2)), 0.8 * np.exp(-((t - 6.0) ** 2) / 4)])
    r = peak_row(1.0, t, Y)
    worst = Y.min(axis=0)
    assert r["yield_robust"] == pytest.approx(worst.max())
    assert r["stop_robust_s"] == pytest.approx(t[int(np.argmax(worst))])
    assert r["peak_nominal_s"] == pytest.approx(4.0) and r["sample_peak_s"][1] == pytest.approx(6.0)
    assert r["window_lo_s"] <= r["stop_robust_s"] <= r["window_hi_s"]
    j = (t >= r["window_lo_s"]) & (t <= r["window_hi_s"])
    assert np.all(worst[j] >= PEAK_WINDOW * r["yield_robust"] - 1e-12)


def test_run_peak_design_on_a_product_that_is_broken_down():
    design, samples = toy()
    res = run_peak_design(design, samples, [1.0], 1e-3, 3000.0, n_points=601, workers=1)
    s = res.summary()
    own = s["sample_peak_s"]
    assert 0 < own[1] < own[0] < 3000.0  # the faster breakdown peaks earlier
    assert own[1] <= s["robust_stop_s"] <= own[0]  # the robust stop lies between the samples' own peaks
    assert s["robust_yield"] <= min(s["sample_peak_yield"]) + 1e-9
    assert res.yields.shape == (2, 601) and s["window_fraction"] == PEAK_WINDOW
