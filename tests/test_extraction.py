"""Tests for extraction.py — extract_cycles and the 3-level hierarchy."""

import numpy as np
import pytest

from qcm_pak._types import MassDataset
from qcm_pak.detection import detect_pulses
from qcm_pak.extraction import extract_cycles
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _synthetic_ald_full(
    n_cycles: int = 5,
    dt: float = 0.5,
    start_time: float = 10.0,
    step_dur: float = 30.0,
    mass_per_step: float = 1.5,
) -> tuple[MassDataset, Recipe]:
    rng = np.random.default_rng(0)
    total_s = start_time + 2 * step_dur * n_cycles + 5.0
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
    for _ in range(n_cycles):
        for _ in ("A", "B"):
            onset_idx = int(round(t_cur / dt))
            end_idx = int(round((t_cur + step_dur) / dt))
            cumulative += mass_per_step
            mass[onset_idx:end_idx] = cumulative
            t_cur += step_dur

    mass += rng.normal(0, 0.01, size=len(mass))
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass), recipe


def test_cycle_count(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=5)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)
    cycles = extract_cycles(data, index, params)
    assert len(cycles) == 5


def test_sub_cycle_runs_per_cycle(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=3)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)

    for cycle in cycles:
        assert len(cycle.sub_cycle_runs) == 1   # simple A+B → one SubCycleRun
        assert len(cycle.sub_cycle_runs[0].steps) == 2


def test_step_names(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=4)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)

    a_steps = cycles.steps(name="A")
    b_steps = cycles.steps(name="B")
    assert len(a_steps) == 4
    assert len(b_steps) == 4


def test_mass_change_positive_for_gain(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=3, mass_per_step=2.0)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)

    for sr in cycles.steps(name="A"):
        assert sr.mass_change > 0, "gain steps should have positive mass_change"


def test_cycle_collection_iteration(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=3)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)

    count = sum(1 for _ in cycles)
    assert count == 3


def test_time_relative_to_onset(tmp_path) -> None:
    data, recipe = _synthetic_ald_full(n_cycles=2)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)

    for sr in cycles.steps():
        assert sr.time[0] == pytest.approx(0.0, abs=1e-9)


def _synthetic_from_recipe(
    recipe: Recipe, dt: float = 0.5, mass_per_step: float = 1.5
) -> MassDataset:
    """Staircase mass trace with one step-up at every recipe event onset."""
    rng = np.random.default_rng(1)
    time = np.arange(0, recipe.total_duration + 5.0, dt)
    mass = np.zeros(len(time))
    t_cur = recipe.start_time
    cumulative = 0.0
    for _ in range(recipe.repeats):
        for sc in recipe.sub_cycles:
            for _ in range(sc.repeats):
                for step in sc.steps:
                    cumulative += mass_per_step
                    mass[int(round(t_cur / dt)):] = cumulative
                    t_cur += step.duration
    mass += rng.normal(0, 0.01, size=len(mass))
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)


@pytest.mark.parametrize("method", ["hybrid", "pelt_guided"])
def test_three_step_recipe_keeps_steps_distinct(tmp_path, method) -> None:
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", 0.5, 19.5), PulseStep("B", 0.5, 19.5), PulseStep("C", 0.5, 29.5),
        ])],
        repeats=6,
        start_time=10.0,
    )
    data = _synthetic_from_recipe(recipe)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(method=method, recipe_tolerance=0.3))
    cycles = extract_cycles(data, index, params)

    assert [name for name, _ in index.step_onsets] == ["A", "B", "C"] * 6
    true_onsets = 10.0 + np.cumsum([0.0] + [20.0, 20.0, 30.0] * 6)[:-1]
    detected = data.time[[idx for _, idx in index.step_onsets]]
    np.testing.assert_allclose(detected, true_onsets, atol=1.5)
    for name in ("A", "B", "C"):
        steps = cycles.steps(name=name)
        assert len(steps) == 6
        assert [s.outer_cycle for s in steps] == list(range(6))


def test_super_cycle_positions(tmp_path) -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.5, 19.5), PulseStep("B", 0.5, 19.5)], repeats=2),
            SubCycle(steps=[PulseStep("A", 0.5, 19.5), PulseStep("C", 0.5, 29.5)], repeats=3),
        ],
        repeats=3,
        start_time=10.0,
    )
    data = _synthetic_from_recipe(recipe)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.3))
    cycles = extract_cycles(data, index, params)

    a_steps = cycles.steps(name="A")
    assert len(a_steps) == 3 * (2 + 3)
    keys = [(s.outer_cycle, s.sub_cycle_index, s.sub_cycle_run) for s in a_steps]
    assert len(set(keys)) == len(keys)   # every A occurrence is uniquely addressable
    assert len(cycles.steps(name="B")) == 3 * 2
    assert len(cycles.steps(name="C")) == 3 * 3
    assert all(s.sub_cycle_index == 1 for s in cycles.steps(name="C"))
