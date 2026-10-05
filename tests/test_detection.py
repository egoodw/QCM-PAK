"""Tests for detection.py — detect_pulses with both methods."""

import pathlib

import numpy as np
import pytest

from qcm_pak._types import MassDataset
from qcm_pak.detection import _build_timeline, _check_count, _estimate_noise, detect_pulses
from qcm_pak.exceptions import DetectionError
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _dataset_covering(duration: float, dt: float = 0.5) -> MassDataset:
    """A plain evenly-spaced MassDataset spanning at least ``duration`` seconds."""
    time = np.arange(0, duration + 5.0, dt)
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=np.zeros(len(time)))


def _synthetic_ald(
    n_cycles: int = 5,
    cycle_duration: float = 60.0,
    dt: float = 0.5,
    start_time: float = 10.0,
    mass_per_step: float = 1.5,
    noise: float = 0.02,
) -> tuple[MassDataset, Recipe]:
    """Create a synthetic two-step ALD mass trace for testing."""
    rng = np.random.default_rng(42)
    step_dur = cycle_duration / 2.0
    total_s = start_time + cycle_duration * n_cycles + 10.0
    time = np.arange(0, total_s, dt)
    mass = np.zeros(len(time))

    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", pulse=dt, purge=step_dur - dt),
            PulseStep("B", pulse=dt, purge=step_dur - dt),
        ])],
        repeats=n_cycles,
        start_time=start_time,
    )

    cumulative = 0.0
    t_cur = start_time
    for _outer in range(n_cycles):
        for _step_name in ("A", "B"):
            onset_idx = int(round(t_cur / dt))
            end_idx = int(round((t_cur + step_dur) / dt))
            cumulative += mass_per_step
            mass[onset_idx:end_idx] = cumulative
            t_cur += step_dur

    mass += rng.normal(0, noise, size=len(mass))

    dataset = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)
    return dataset, recipe


def test_build_timeline_event_count() -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 29.9), PulseStep("B", 0.1, 29.9)])
        ],
        repeats=10,
        start_time=5.0,
    )
    timeline = _build_timeline(recipe, _dataset_covering(recipe.start_time + 30 * 20))
    assert len(timeline) == 20


def test_build_timeline_super_cycle() -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(
                steps=[PulseStep("A", 0.1, 9.9), PulseStep("B", 0.1, 9.9)], repeats=3
            ),
            SubCycle(
                steps=[PulseStep("A", 0.1, 9.9), PulseStep("C", 0.2, 9.9)], repeats=2
            ),
        ],
        repeats=4,
    )
    timeline = _build_timeline(recipe, _dataset_covering(4 * (3 * 20 + 2 * 20)))
    assert len(timeline) == recipe.total_events


def test_estimate_noise_clean_signal() -> None:
    mass = np.linspace(0, 10, 1000)   # perfectly linear, no noise
    sigma = _estimate_noise(mass)
    assert sigma < 0.05   # MAD estimator has a small positive floor on step signals


def test_pelt_guided_event_count(tmp_path: pathlib.Path) -> None:

    data, recipe = _synthetic_ald(n_cycles=5)

    # Create a dummy file so ALDParameters doesn't fail
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("t,f\n")

    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="pelt_guided", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)
    assert index.n_detected == recipe.total_events


def test_hybrid_event_count(tmp_path: pathlib.Path) -> None:

    data, recipe = _synthetic_ald(n_cycles=5)
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("t,f\n")

    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="hybrid", recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)
    assert index.n_detected == recipe.total_events


def test_cycle_index_step_names(tmp_path: pathlib.Path) -> None:

    data, recipe = _synthetic_ald(n_cycles=3)
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("t,f\n")

    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params)
    names = [name for name, _ in index.step_onsets]
    assert names == ["A", "B"] * 3


def _tiny_recipe(n_cycles: int = 2) -> Recipe:
    return Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10)])],
        repeats=n_cycles,
    )


def test_check_count_accepts_strictly_increasing_onsets() -> None:
    recipe = _tiny_recipe()
    onsets = [("A", 10), ("B", 20), ("A", 30), ("B", 40)]
    _check_count(onsets, recipe)  # should not raise


def test_check_count_rejects_duplicate_onset_index() -> None:
    # Two adjacent events assigned the same sample index — the failure mode
    # that used to crash deep inside extraction.py with a cryptic IndexError
    # (an empty time_abs[onset_idx:end_idx] slice) instead of a clear message
    # here at detection time.
    recipe = _tiny_recipe()
    onsets = [("A", 10), ("B", 10), ("A", 30), ("B", 40)]
    with pytest.raises(DetectionError, match="strictly increasing"):
        _check_count(onsets, recipe)


def test_check_count_rejects_reversed_onset_index() -> None:
    recipe = _tiny_recipe()
    onsets = [("A", 10), ("B", 20), ("A", 15), ("B", 40)]
    with pytest.raises(DetectionError, match="strictly increasing"):
        _check_count(onsets, recipe)


def _synthetic_ald_bursty(
    n_cycles: int = 5,
    cycle_duration: float = 60.0,
    dt: float = 0.5,
    start_time: float = 10.0,
    mass_per_step: float = 1.5,
    noise: float = 0.02,
    burst_size: int = 4,
) -> tuple[MassDataset, Recipe]:
    """Same physical signal as ``_synthetic_ald``, but every real sample is
    preceded by ``burst_size - 1`` near-duplicate jitter timestamps — the
    real-world failure pattern (an instrument logging several readings per
    acquisition tick) that used to corrupt MassDataset.dt's median-based
    estimate by orders of magnitude. Onsets should still be found correctly
    since detection locates them directly in the time array.
    """
    clean, recipe = _synthetic_ald(n_cycles, cycle_duration, dt, start_time, mass_per_step, noise)
    rng = np.random.default_rng(7)
    time_out = [clean.time[0]]
    mass_out = [clean.mass[0]]
    for i in range(1, len(clean.time)):
        jitter = time_out[-1] + np.cumsum(rng.uniform(1e-8, 5e-8, size=burst_size - 1))
        time_out.extend(jitter.tolist())
        mass_out.extend([clean.mass[i - 1]] * (burst_size - 1))
        time_out.append(clean.time[i])
        mass_out.append(clean.mass[i])
    time = np.array(time_out)
    mass = np.array(mass_out)
    bursty = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)
    return bursty, recipe


@pytest.mark.parametrize("method", ["hybrid", "pelt_guided"])
def test_detect_pulses_succeeds_with_bursty_duplicate_timestamps(
    method: str, tmp_path: pathlib.Path
) -> None:
    """Detection must not reject data just because near-duplicate burst
    timestamps corrupt the median-based dt estimate — onsets are located
    directly in the time array (searchsorted), not derived from a single
    global dt, so real duplicate/near-duplicate rows should simply work.
    """
    data, recipe = _synthetic_ald_bursty(n_cycles=5)
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method=method, recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)
    assert index.n_detected == recipe.total_events


def test_detect_pulses_rejects_unsorted_time(tmp_path: pathlib.Path) -> None:
    """The one structural requirement left is that time be non-decreasing —
    detect_pulses locates onsets via binary search into the time array,
    which is meaningless (and silently wrong, not just inaccurate) if time
    isn't sorted.
    """
    data, recipe = _synthetic_ald(n_cycles=2)
    time = data.time.copy()
    time[100], time[101] = time[101], time[100]
    bad = MassDataset(time=time, frequency=data.frequency, mass=data.mass)

    dummy = tmp_path / "dummy.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)

    with pytest.raises(DetectionError, match="not sorted"):
        detect_pulses(bad, params, DetectionParameters(method="hybrid"))
