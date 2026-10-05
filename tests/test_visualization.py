"""Tests for visualization.py — smoke tests for all plot functions."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")   # non-interactive backend for tests

from matplotlib.figure import Figure  # noqa: E402

from qcm_pak._types import (  # noqa: E402
    Cycle,
    CycleCollection,
    CycleIndex,
    EtchResult,
    LangmuirResult,
    MassDataset,
    StepResult,
    SubCycleRun,
)
from qcm_pak.recipe import PulseStep, Recipe, SubCycle  # noqa: E402
from qcm_pak.visualization import (  # noqa: E402
    plot_cycle_average,
    plot_cycles,
    plot_cycles_batch,
    plot_derivative,
    plot_detailed_cycles,
    plot_etch,
    plot_fit_drift,
    plot_langmuir,
    plot_pulse_timing,
    plot_trace,
)


def _simple_mass() -> MassDataset:
    t = np.linspace(0, 100, 500)
    mass = np.sin(t / 10) * 5 + np.linspace(0, 20, 500)
    return MassDataset(time=t, frequency=np.full(500, 5e6), mass=mass)


def _simple_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", 1.0, 10.0),
            PulseStep("B", 1.0, 10.0),
        ])],
        repeats=3,
    )


def _simple_collection() -> CycleCollection:
    recipe = _simple_recipe()
    t = np.linspace(0, 30, 60)
    mass = np.linspace(0, 5, 60)
    steps_a = [
        StepResult("A", "gain", 0, 0, 0, i, t, mass, mass, float(mass[-1]))
        for i in range(3)
    ]
    steps_b = [
        StepResult("B", "gain", 1, 0, 0, i, t, mass, mass, float(mass[-1]))
        for i in range(3)
    ]
    sub_runs = [SubCycleRun(0, 0, i, [steps_a[i], steps_b[i]]) for i in range(3)]
    cycles = [Cycle(i, [sub_runs[i]]) for i in range(3)]
    return CycleCollection(cycles=cycles, recipe=recipe)


def test_plot_trace_returns_figure() -> None:
    fig = plot_trace(_simple_mass())
    assert isinstance(fig, Figure)


def test_plot_trace_with_index() -> None:
    index = CycleIndex(
        step_onsets=[("A", 10), ("B", 60), ("A", 120), ("B", 170)],
        recipe=_simple_recipe(),
    )
    fig = plot_trace(_simple_mass(), index)
    assert isinstance(fig, Figure)


def test_plot_cycles_returns_figure() -> None:
    fig = plot_cycles(_simple_collection())
    assert isinstance(fig, Figure)


def test_plot_cycles_single_step() -> None:
    fig = plot_cycles(_simple_collection(), step="A")
    assert isinstance(fig, Figure)


def test_plot_langmuir_mono() -> None:
    result = LangmuirResult(
        step_name="A", model="mono", r_squared=0.99,
        k=0.1, theta_max=12.0,
    )
    fig = plot_langmuir(_simple_collection(), result)
    assert isinstance(fig, Figure)


def test_plot_etch_saturating() -> None:
    result = EtchResult(
        step_name="A", model="saturating", r_squared=0.98,
        k=0.05, etch_max=3.0,
    )
    fig = plot_etch(_simple_collection(), result)
    assert isinstance(fig, Figure)


def test_plot_derivative_returns_figure() -> None:
    fig = plot_derivative(_simple_mass())
    assert isinstance(fig, Figure)


def _simple_index() -> CycleIndex:
    # Matches _simple_recipe(): 3 outer cycles x (A, B) = 6 events.
    return CycleIndex(
        step_onsets=[
            ("A", 10), ("B", 60),
            ("A", 120), ("B", 170),
            ("A", 230), ("B", 280),
        ],
        recipe=_simple_recipe(),
    )


def test_plot_detailed_cycles_first() -> None:
    fig = plot_detailed_cycles(_simple_mass(), _simple_index(), location="first", n_cycles=1)
    assert isinstance(fig, Figure)


def test_plot_detailed_cycles_middle() -> None:
    fig = plot_detailed_cycles(_simple_mass(), _simple_index(), location="middle", n_cycles=1)
    assert isinstance(fig, Figure)


def test_plot_detailed_cycles_last() -> None:
    fig = plot_detailed_cycles(_simple_mass(), _simple_index(), location="last", n_cycles=1)
    assert isinstance(fig, Figure)


def test_plot_detailed_cycles_clamps_n_cycles_above_total() -> None:
    # Asking for more cycles than exist shouldn't error — should just show everything.
    fig = plot_detailed_cycles(_simple_mass(), _simple_index(), location="first", n_cycles=999)
    assert isinstance(fig, Figure)


def test_plot_cycles_batch_returns_figure() -> None:
    fig = plot_cycles_batch(_simple_collection(), "A", cycle_range=(0, 2))
    assert isinstance(fig, Figure)


def test_plot_cycles_batch_empty_range() -> None:
    fig = plot_cycles_batch(_simple_collection(), "A", cycle_range=(50, 60))
    assert isinstance(fig, Figure)


def test_plot_cycle_average_returns_figure() -> None:
    fig = plot_cycle_average(_simple_collection(), "A", cycle_range=(0, 2))
    assert isinstance(fig, Figure)


def test_plot_pulse_timing_all_steps() -> None:
    fig = plot_pulse_timing(_simple_index(), _simple_mass())
    assert isinstance(fig, Figure)


def test_plot_pulse_timing_single_step() -> None:
    fig = plot_pulse_timing(_simple_index(), _simple_mass(), step="A")
    assert isinstance(fig, Figure)


def test_plot_pulse_timing_too_few_pulses() -> None:
    index = CycleIndex(step_onsets=[("A", 10)], recipe=_simple_recipe())
    fig = plot_pulse_timing(index, _simple_mass())
    assert isinstance(fig, Figure)


def test_plot_fit_drift_with_dataclass_results() -> None:
    records = [
        LangmuirResult(step_name="A", model="mono", r_squared=0.9, k=0.1 + 0.01 * i, theta_max=10.0, outer_cycle=i)
        for i in range(5)
    ]
    fig = plot_fit_drift(records, "A", param="k")
    assert isinstance(fig, Figure)


def test_plot_fit_drift_with_dict_records() -> None:
    records = [{"outer_cycle": i, "k": 0.1 + 0.01 * i} for i in range(5)]
    fig = plot_fit_drift(records, "A", param="k")
    assert isinstance(fig, Figure)


def test_plot_fit_drift_empty_records() -> None:
    fig = plot_fit_drift([], "A", param="k")
    assert isinstance(fig, Figure)
