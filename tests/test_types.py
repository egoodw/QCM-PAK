"""Tests for _types.py — MassDataset.check_dt_sanity."""

import numpy as np
import pytest

from qcm_pak._types import MassDataset


def _clean_dataset(n: int = 1000, dt: float = 0.1) -> MassDataset:
    time = np.arange(n) * dt
    return MassDataset(time=time, frequency=np.full(n, 5e6), mass=np.zeros(n))


def test_check_dt_sanity_passes_for_clean_data() -> None:
    _clean_dataset().check_dt_sanity()  # should not raise


def test_check_dt_sanity_passes_for_realistic_duplicate_rows() -> None:
    # Instruments that write a few duplicate rows per real tick are normal —
    # dt already ignores zero diffs, and the naive/median ratio stays small.
    dt = 0.1
    base = np.arange(1000) * dt
    time = np.repeat(base, 3)  # 3 duplicate timestamps per real sample
    ds = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=np.zeros(len(time)))
    ds.check_dt_sanity()  # should not raise


def test_check_dt_sanity_raises_for_bursty_jitter_column() -> None:
    # Reproduces the real failure mode: most of the file has honest 0.1s
    # spacing, but every burst starts with a handful of near-duplicate
    # microsecond-jitter timestamps (as seen in a real instrument log's
    # "converted_adjusted_time_in_min"-style column) — enough of them to
    # drag the median positive diff down by orders of magnitude.
    rng = np.random.default_rng(0)
    chunks = []
    t = 0.0
    for _ in range(500):
        jitter = t + np.cumsum(rng.uniform(1e-8, 5e-8, size=4))
        chunks.append(jitter)
        t = jitter[-1] + 0.1
        chunks.append([t])
    time = np.concatenate(chunks)
    ds = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=np.zeros(len(time)))

    with pytest.raises(ValueError, match="near-duplicate timestamps"):
        ds.check_dt_sanity()


def test_check_dt_sanity_respects_tolerance() -> None:
    # This pattern's naive/median ratio is ~3.0x (verified directly) — a
    # real but moderate mismatch, unlike the bursty-jitter case's ~500,000x.
    # Exercises the threshold in both directions rather than just the
    # generous default.
    dt = 0.1
    base = np.arange(1000) * dt
    time = np.repeat(base, 3)
    ds = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=np.zeros(len(time)))
    ds.check_dt_sanity(tolerance=5.0)  # 3.0x < 5.0x -> should not raise
    with pytest.raises(ValueError, match="near-duplicate timestamps"):
        ds.check_dt_sanity(tolerance=2.0)  # 3.0x > 2.0x -> should raise
