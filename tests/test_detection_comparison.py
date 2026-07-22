"""
Quantitative comparison of pelt_guided vs hybrid pulse detection.

Tests measure *onset accuracy*: how close each algorithm's detected onset
sample index is to the known ground-truth onset embedded in synthetic data.
Both methods must satisfy a minimum accuracy threshold; the comparison
surfaces relative performance across noise and SNR conditions.

Metrics
-------
mean_error : mean |detected_idx - true_idx| over all events, in samples
max_error  : worst-case offset, in samples
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from qcm_pak._types import MassDataset
from qcm_pak.detection import detect_pulses
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle

# ── Synthetic data helpers ─────────────────────────────────────────────────────


def _make_ald_trace(
    *,
    n_cycles: int = 10,
    pulse_s: float = 0.05,
    purge_s: float = 20.0,
    dt: float = 0.1,
    start_time: float = 30.0,
    mass_per_step: float = 2.0,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> tuple[MassDataset, Recipe, list[int]]:
    """Build a noiseless step-function mass trace and return ground-truth onsets.

    Returns
    -------
    data : MassDataset
    recipe : Recipe
    true_onsets : list[int]
        Ground-truth sample indices of each step onset (chronological order).
    """
    rng = np.random.default_rng(seed)
    step_dur = pulse_s + purge_s

    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", pulse=pulse_s, purge=purge_s),
            PulseStep("B", pulse=pulse_s, purge=purge_s),
        ])],
        repeats=n_cycles,
        start_time=start_time,
    )

    total_s = start_time + 2 * step_dur * n_cycles + 10.0
    n_pts = int(total_s / dt) + 1
    time = np.arange(n_pts) * dt
    mass = np.zeros(n_pts)

    true_onsets: list[int] = []
    t_cur = start_time
    cumulative = 0.0
    for _ in range(n_cycles):
        for _ in ("A", "B"):
            idx = int(round(t_cur / dt))
            true_onsets.append(idx)
            cumulative += mass_per_step
            end_idx = min(int(round((t_cur + step_dur) / dt)), n_pts)
            mass[idx:end_idx] = cumulative
            t_cur += step_dur

    mass = mass + rng.normal(0, noise_sigma, size=n_pts)

    data = MassDataset(
        time=time,
        frequency=np.full(n_pts, 5.0e6),
        mass=mass,
    )
    return data, recipe, true_onsets


def _make_ale_trace(
    *,
    n_cycles: int = 10,
    pulse_s: float = 0.05,
    purge_s: float = 20.0,
    dt: float = 0.1,
    start_time: float = 30.0,
    mass_per_step: float = -1.5,
    noise_sigma: float = 0.05,
    seed: int = 1,
) -> tuple[MassDataset, Recipe, list[int]]:
    """Mixed-sign ALE trace: both steps are mass-loss."""
    rng = np.random.default_rng(seed)
    step_dur = pulse_s + purge_s

    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("HF",  pulse=pulse_s, purge=purge_s, mass_effect="loss"),
            PulseStep("TMA", pulse=pulse_s, purge=purge_s, mass_effect="loss"),
        ])],
        repeats=n_cycles,
        start_time=start_time,
    )

    total_s = start_time + 2 * step_dur * n_cycles + 10.0
    n_pts = int(total_s / dt) + 1
    time = np.arange(n_pts) * dt
    mass = np.zeros(n_pts)

    true_onsets: list[int] = []
    t_cur = start_time
    cumulative = 0.0
    for _ in range(n_cycles):
        for _ in ("HF", "TMA"):
            idx = int(round(t_cur / dt))
            true_onsets.append(idx)
            cumulative += mass_per_step  # negative
            end_idx = min(int(round((t_cur + step_dur) / dt)), n_pts)
            mass[idx:end_idx] = cumulative
            t_cur += step_dur

    mass = mass + rng.normal(0, noise_sigma, size=n_pts)

    data = MassDataset(
        time=time,
        frequency=np.full(n_pts, 5.0e6),
        mass=mass,
    )
    return data, recipe, true_onsets


def _accuracy(
    detected: list[tuple[str, int]],
    true_onsets: list[int],
) -> tuple[float, int]:
    """Return (mean_error, max_error) in samples."""
    errors = [abs(det_idx - true_idx) for (_, det_idx), true_idx
              in zip(detected, true_onsets)]
    return float(np.mean(errors)), int(np.max(errors))


# ── Binary ALD at three noise levels ──────────────────────────────────────────


@pytest.mark.parametrize("noise_sigma,max_allowed_mean,max_allowed_max", [
    (0.01, 3, 10),    # low noise  — PELT L2 boundary placement is ~1-2 samples off
    (0.10, 4, 12),    # medium noise (SNR ~ 20)
    (0.50, 6, 20),    # high noise (SNR ~ 4)
])
def test_pelt_accuracy_binary(
    tmp_path: pathlib.Path,
    noise_sigma: float,
    max_allowed_mean: int,
    max_allowed_max: int,
) -> None:
    data, recipe, true_onsets = _make_ald_trace(noise_sigma=noise_sigma)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="pelt_guided", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)

    mean_err, max_err = _accuracy(index.step_onsets, true_onsets)
    assert mean_err <= max_allowed_mean, (
        f"pelt_guided mean error {mean_err:.2f} samples "
        f"exceeds {max_allowed_mean} at noise={noise_sigma}"
    )
    assert max_err <= max_allowed_max, (
        f"pelt_guided max error {max_err} samples "
        f"exceeds {max_allowed_max} at noise={noise_sigma}"
    )


@pytest.mark.parametrize("noise_sigma,max_allowed_mean,max_allowed_max", [
    (0.01, 2,  5),
    (0.10, 4, 10),
    (0.50, 8, 20),    # hybrid degrades faster under noise
])
def test_hybrid_accuracy_binary(
    tmp_path: pathlib.Path,
    noise_sigma: float,
    max_allowed_mean: int,
    max_allowed_max: int,
) -> None:
    data, recipe, true_onsets = _make_ald_trace(noise_sigma=noise_sigma)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="hybrid", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)

    mean_err, max_err = _accuracy(index.step_onsets, true_onsets)
    assert mean_err <= max_allowed_mean, (
        f"hybrid mean error {mean_err:.2f} samples "
        f"exceeds {max_allowed_mean} at noise={noise_sigma}"
    )
    assert max_err <= max_allowed_max, (
        f"hybrid max error {max_err} samples "
        f"exceeds {max_allowed_max} at noise={noise_sigma}"
    )


# ── ALE (mixed mass-loss signs) ────────────────────────────────────────────────


def test_pelt_accuracy_ale(tmp_path: pathlib.Path) -> None:
    """PELT correctly handles loss-only steps."""
    data, recipe, true_onsets = _make_ale_trace(noise_sigma=0.05)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="pelt_guided", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)

    mean_err, max_err = _accuracy(index.step_onsets, true_onsets)
    assert mean_err <= 4
    assert max_err <= 10


def test_hybrid_accuracy_ale(tmp_path: pathlib.Path) -> None:
    """Hybrid derivative search handles loss-only steps."""
    data, recipe, true_onsets = _make_ale_trace(noise_sigma=0.05)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="hybrid", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)

    mean_err, max_err = _accuracy(index.step_onsets, true_onsets)
    assert mean_err <= 4
    assert max_err <= 10


# ── Head-to-head comparison ────────────────────────────────────────────────────


@pytest.mark.parametrize("noise_sigma", [0.01, 0.05, 0.20, 0.50])
def test_pelt_not_worse_than_hybrid(
    tmp_path: pathlib.Path,
    noise_sigma: float,
) -> None:
    """PELT mean onset error should not exceed hybrid by more than 3 samples.

    At low-to-medium noise, PELT is typically more accurate because it
    searches globally rather than relying on local derivative peaks. This
    test lets hybrid be better in edge cases (high noise + small dt) but
    ensures PELT doesn't catastrophically regress relative to it.
    """
    data, recipe, true_onsets = _make_ald_trace(noise_sigma=noise_sigma, n_cycles=20)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)

    pelt_idx = detect_pulses(
        data, params, DetectionParameters(method="pelt_guided", recipe_tolerance=0.4)
    )
    hyb_idx = detect_pulses(
        data, params, DetectionParameters(method="hybrid", recipe_tolerance=0.4)
    )

    pelt_mean, _ = _accuracy(pelt_idx.step_onsets, true_onsets)
    hyb_mean, _  = _accuracy(hyb_idx.step_onsets, true_onsets)

    assert pelt_mean <= hyb_mean + 3, (
        f"PELT mean error ({pelt_mean:.2f}) is more than 3 samples worse "
        f"than hybrid ({hyb_mean:.2f}) at noise={noise_sigma}"
    )


# ── Timing-offset robustness ───────────────────────────────────────────────────


@pytest.mark.parametrize("start_time_offset", [-10.0, -5.0, 0.0, 5.0, 10.0])
def test_pelt_robust_to_start_time_error(
    tmp_path: pathlib.Path,
    start_time_offset: float,
) -> None:
    """PELT should remain accurate when the recipe start_time is off by up to
    10 seconds (common when the user estimates stabilisation time by eye).

    Uses 30 s purges (a realistic step duration — see README/pyproject
    examples) rather than 20 s: with additive tolerance the window spans
    ±(tol × step_duration), and a ±10 s timing error against a 20 s step is
    close enough to half the step's own duration that a neighboring event's
    true onset becomes equidistant from the (mis-centered) expected
    position — a genuinely ambiguous assignment that no distance-based
    matcher can resolve, not a detection accuracy failure. At 30 s steps,
    ±10 s uncertainty is a comfortable third of a step and the ambiguity
    does not arise.
    """
    true_start = 30.0
    data, _, true_onsets = _make_ald_trace(
        start_time=true_start, n_cycles=5, purge_s=30.0
    )

    # Build recipe with wrong start time
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", pulse=0.05, purge=30.0),
            PulseStep("B", pulse=0.05, purge=30.0),
        ])],
        repeats=5,
        start_time=true_start + start_time_offset,
    )
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    # recipe_tolerance=0.6: ±(0.6 × 300 samples) = ±180 samples = ±18 s
    det = DetectionParameters(method="pelt_guided", recipe_tolerance=0.6)
    index = detect_pulses(data, params, det)

    mean_err, _ = _accuracy(index.step_onsets, true_onsets)
    assert mean_err <= 10, (
        f"PELT failed with start_time offset {start_time_offset:+.0f}s "
        f"(mean error {mean_err:.2f} samples)"
    )
