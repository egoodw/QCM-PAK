"""Tests for _pelt.py: the vectorised PELT must agree with ruptures."""

import numpy as np
import pytest

from qcm_pak._pelt import pelt_l2


def _stepped(n_steps: int, seg: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    levels = np.cumsum(rng.choice([-1.5, 1.5, 2.5], size=n_steps))
    return np.repeat(levels, seg) + rng.normal(0, noise, n_steps * seg)


def test_finds_a_single_step() -> None:
    x = np.concatenate([np.zeros(100), np.full(100, 5.0)])
    x += np.random.default_rng(0).normal(0, 0.1, 200)
    found = pelt_l2(x, pen=10.0, min_size=10)
    assert len(found) == 2 and found[-1] == 200
    assert 95 <= found[0] <= 105


def test_constant_signal_has_no_interior_breakpoint() -> None:
    assert pelt_l2(np.zeros(500), pen=1.0, min_size=10) == [500]


def test_short_signal_returns_end_only() -> None:
    assert pelt_l2(np.zeros(5), pen=1.0, min_size=10) == [5]


@pytest.mark.parametrize("min_size", [10, 15, 20, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_ruptures(seed: int, min_size: int) -> None:
    rpt = pytest.importorskip("ruptures")
    x = _stepped(n_steps=12, seg=157, noise=0.4, seed=seed)
    pen = 12.0
    expected = rpt.Pelt(model="l2", min_size=min_size).fit(x).predict(pen=pen)
    assert pelt_l2(x, pen=pen, min_size=min_size) == expected


def test_matches_ruptures_with_odd_length() -> None:
    rpt = pytest.importorskip("ruptures")
    x = _stepped(n_steps=8, seg=101, noise=0.2, seed=5)[:-3]
    expected = rpt.Pelt(model="l2", min_size=10).fit(x).predict(pen=5.0)
    assert pelt_l2(x, pen=5.0, min_size=10) == expected
