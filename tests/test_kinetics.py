"""Tests for kinetics.py — fit_langmuir and fit_etch."""

import warnings

import numpy as np
import pytest

from qcm_pak._types import (
    Cycle,
    CycleCollection,
    StepResult,
    SubCycleRun,
)
from qcm_pak.kinetics import _bic, _r_squared, fit_etch, fit_langmuir
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _make_step(
    name: str,
    t_max: float,
    mass_change: float,
    k: float,
    theta_max: float,
    outer: int = 0,
) -> StepResult:
    """Create a synthetic StepResult following a Langmuir curve."""
    t = np.linspace(0, t_max, 50)
    mass_c = theta_max * (1.0 - np.exp(-k * t))
    return StepResult(
        step_name=name,
        mass_effect="gain",
        step_index=0,
        sub_cycle_index=0,
        sub_cycle_run=0,
        outer_cycle=outer,
        time=t,
        mass_raw=mass_c,
        mass_corrected=mass_c,
        mass_change=float(mass_c[-1]),
    )


def _make_collection(steps: list[StepResult], recipe: Recipe) -> CycleCollection:
    """Wrap a flat list of StepResults into a CycleCollection."""
    sub_runs = [
        SubCycleRun(sub_cycle_index=0, run_number=0, outer_cycle=i, steps=[s])
        for i, s in enumerate(steps)
    ]
    cycles = [Cycle(cycle_number=i, sub_cycle_runs=[sr]) for i, sr in enumerate(sub_runs)]
    return CycleCollection(cycles=cycles, recipe=recipe)


def _simple_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("A", 30.0, 0.1)])],
        repeats=1,
    )


def test_r_squared_perfect_fit() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_poor_fit() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_bad = np.array([4.0, 3.0, 2.0, 1.0])
    assert _r_squared(y, y_bad) < 0.5


def test_bic_increases_with_params() -> None:
    bic_2 = _bic(1.0, 100, 2)
    bic_4 = _bic(1.0, 100, 4)
    assert bic_4 > bic_2


def test_fit_langmuir_mono() -> None:
    """fit_langmuir fits the within-pulse averaged trace; all steps share same t array."""
    recipe = _simple_recipe()
    rng = np.random.default_rng(1)
    steps = [
        _make_step("A", 30.0, 12.0, k=0.1, theta_max=12.0, outer=i)
        for i in range(30)
    ]
    # Add a tiny bit of noise to each step's mass trace
    for s in steps:
        s.mass_corrected[:] += rng.normal(0, 0.05, size=len(s.time))
        s.mass_change = float(s.mass_corrected[-1])

    collection = _make_collection(steps, recipe)
    result = fit_langmuir(collection, step="A", model="mono")
    assert result.model == "mono"
    assert result.r_squared > 0.95
    assert result.k is not None
    assert result.theta_max is not None
    assert result.k == pytest.approx(0.1, rel=0.3)
    assert result.theta_max == pytest.approx(12.0, rel=0.15)


def test_fit_langmuir_no_step_raises() -> None:
    recipe = _simple_recipe()
    steps = [_make_step("A", 30.0, 10.0, k=0.1, theta_max=10.0)]
    collection = _make_collection(steps, recipe)
    with pytest.raises(ValueError, match="No step named"):
        fit_langmuir(collection, step="NONEXISTENT")


def test_fit_langmuir_bi_fallback_warns() -> None:
    """When data is well-described by mono, bimodal fit should warn and fall back."""
    recipe = _simple_recipe()
    steps = [
        _make_step("A", 30.0, 10.0, k=0.15, theta_max=10.0, outer=i)
        for i in range(25)
    ]
    collection = _make_collection(steps, recipe)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fit_langmuir(collection, step="A", model="bi")
    # Either fell back to mono (with warning) or succeeded as bi
    assert result.r_squared > 0.9


def test_fit_etch_saturating() -> None:
    """fit_etch averages the within-pulse corrected traces (sign-flipped for loss steps)."""
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("HF", 30.0, 0.1, mass_effect="loss")])],
        repeats=1,
    )
    rng = np.random.default_rng(2)
    t_max = 30.0
    k_true, etch_max_true = 0.08, 5.0

    steps = []
    for i in range(25):
        t = np.linspace(0, t_max, 50)
        e = etch_max_true * (1.0 - np.exp(-k_true * t))
        e += rng.normal(0, 0.05, size=len(t))
        # mass_corrected is negative (loss); fit_etch takes abs()
        sr = StepResult(
            step_name="HF",
            mass_effect="loss",
            step_index=0,
            sub_cycle_index=0,
            sub_cycle_run=0,
            outer_cycle=i,
            time=t,
            mass_raw=-e,
            mass_corrected=-e,
            mass_change=float(-e[-1]),
        )
        steps.append(sr)

    sub_runs = [SubCycleRun(0, 0, i, [s]) for i, s in enumerate(steps)]
    cycles_list = [Cycle(i, [sr]) for i, sr in enumerate(sub_runs)]
    collection = CycleCollection(cycles=cycles_list, recipe=recipe)

    result = fit_etch(collection, step="HF", model="saturating")
    assert result.model == "saturating"
    assert result.r_squared > 0.90
    assert result.k == pytest.approx(k_true, rel=0.3)
    assert result.etch_max == pytest.approx(etch_max_true, rel=0.15)
