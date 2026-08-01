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
    fig_to_png_bytes,
    plot_cycles,
    plot_derivative,
    plot_etch,
    plot_langmuir,
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


def test_plots_do_not_register_with_pyplot() -> None:
    """Figures are built via the OO API, so pyplot's global registry stays empty.

    Regression test for a leak where ``plt.subplots()`` registered every
    figure with pyplot's global state and nothing ever closed it — fatal
    for a long-lived service handling many concurrent analysis requests.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    assert plt.get_fignums() == []

    plot_trace(_simple_mass())
    plot_cycles(_simple_collection())
    plot_derivative(_simple_mass())
    plot_langmuir(
        _simple_collection(),
        LangmuirResult(step_name="A", model="mono", r_squared=0.99, k=0.1, theta_max=12.0),
    )
    plot_etch(
        _simple_collection(),
        EtchResult(step_name="A", model="saturating", r_squared=0.98, k=0.05, etch_max=3.0),
    )

    assert plt.get_fignums() == []


def test_fig_to_png_bytes() -> None:
    fig = plot_trace(_simple_mass())
    png = fig_to_png_bytes(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
