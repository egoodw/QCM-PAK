"""Tests for sauerbrey.py — frequency_to_mass."""

import numpy as np
import pytest

from qcm_pak._types import QCMDataset
from qcm_pak.parameters import SauerbreyConstants
from qcm_pak.sauerbrey import frequency_to_mass


def _make_dataset(delta_f: float, n: int = 10) -> QCMDataset:
    time = np.linspace(0, n - 1, n, dtype=np.float64)
    f0 = 5_000_000.0
    frequency = np.full(n, f0, dtype=np.float64)
    frequency[-1] = f0 + delta_f
    return QCMDataset(time=time, frequency=frequency)


def test_mass_at_zero_first_point() -> None:
    ds = _make_dataset(delta_f=-100.0)
    mass_ds = frequency_to_mass(ds)
    assert mass_ds.mass[0] == pytest.approx(0.0)


def test_frequency_drop_gives_mass_gain() -> None:
    ds = _make_dataset(delta_f=-100.0)   # freq drops → mass gained
    mass_ds = frequency_to_mass(ds)
    assert mass_ds.mass[-1] > 0


def test_frequency_rise_gives_mass_loss() -> None:
    ds = _make_dataset(delta_f=100.0)    # freq rises → mass lost
    mass_ds = frequency_to_mass(ds)
    assert mass_ds.mass[-1] < 0


def test_sauerbrey_factor_applied() -> None:
    c = SauerbreyConstants()
    ds = _make_dataset(delta_f=-1.0)     # 1 Hz drop
    mass_ds = frequency_to_mass(ds, c)
    # C ≈ -12.27 ng/cm²/Hz → 1 Hz drop → ~12.27 ng/cm² gain
    assert mass_ds.mass[-1] == pytest.approx(abs(c.conversion_factor), rel=1e-6)


def test_temperature_passed_through() -> None:
    time = np.array([0.0, 1.0, 2.0])
    freq = np.array([5e6, 5e6 - 1, 5e6 - 2])
    temp = np.array([25.0, 25.1, 25.2])
    ds = QCMDataset(time=time, frequency=freq, temperature=temp)
    mass_ds = frequency_to_mass(ds)
    assert mass_ds.temperature is not None
    np.testing.assert_array_equal(mass_ds.temperature, temp)


def test_custom_crystal_constants() -> None:
    crystal = SauerbreyConstants(fundamental_frequency=5.0e6, overtone=1)
    ds = _make_dataset(delta_f=-1.0)
    mass_ds = frequency_to_mass(ds, crystal)
    default_mass = frequency_to_mass(ds).mass[-1]
    # Different crystal → different mass value
    assert mass_ds.mass[-1] != pytest.approx(default_mass, rel=0.01)
