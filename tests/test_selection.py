"""Criterion-based PINN sample choice (core.selection): fixed starts, likelihood-ratio qualification, maximin."""

import numpy as np
import pytest

from enzopt.core.selection import CHI2_1_95, lr_threshold, maximin_select, start_offsets


def test_start_offsets_deterministic_and_stratified():
    a = start_offsets(32, 10, 1.0)
    b = start_offsets(32, 10, 1.0)
    assert a.shape == (32, 10)
    assert np.array_equal(a, b)
    assert np.all(np.abs(a) < 1.0)
    # every coordinate takes the 32 stratum midpoints of [-1, 1] exactly once
    mids = -1.0 + (2 * np.arange(32) + 1) / 32
    for j in range(10):
        assert np.allclose(np.sort(a[:, j]), mids)


def test_start_offsets_rejects_empty():
    with pytest.raises(ValueError):
        start_offsets(0, 3, 1.0)


def test_lr_threshold():
    assert lr_threshold(1.0, 108) == pytest.approx(np.exp(CHI2_1_95 / 216))
    # more independent evidence → tighter threshold
    assert lr_threshold(1.0, 1000) < lr_threshold(1.0, 10)


def test_maximin_best_first_then_farthest():
    theta = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [0.0, 2.0], [5.0, 5.0]])
    rmse = np.array([1.00, 0.99, 1.01, 1.02, 2.00])
    ok = rmse <= 1.05
    got = maximin_select(theta, rmse, ok, k=10)
    assert [i for i, _ in got] == [1, 2, 3, 0]  # 4 is not qualified; best fit (1) first
    assert got[0][1] is None
    assert got[1][1] == pytest.approx(2.9)


def test_maximin_cap_and_ties():
    theta = np.array([[0.0], [1.0], [-1.0]])
    rmse = np.array([1.0, 1.1, 1.05])
    got = maximin_select(theta, rmse, np.ones(3, bool), k=2)
    assert [i for i, _ in got] == [0, 2]  # equal distance: the lower RMSE wins
    assert maximin_select(theta, rmse, np.zeros(3, bool), k=2) == []
