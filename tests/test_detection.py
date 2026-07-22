"""Tests for detection.py — detect_pulses with both methods."""

import pathlib

import numpy as np

from qcm_pak._types import MassDataset
from qcm_pak.detection import _build_timeline, _estimate_noise, detect_pulses
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


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
    timeline = _build_timeline(recipe, dt=0.5)
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
    timeline = _build_timeline(recipe, dt=0.5)
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
