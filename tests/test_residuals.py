"""Tests for residuals.py — fit residual traces and their aggregates."""

import json

import numpy as np
import pytest

from qcm_pak._types import Cycle, CycleCollection, StepResult, SubCycleRun
from qcm_pak.kinetics import fit_etch, fit_langmuir, fit_langmuir_per_cycle
from qcm_pak.recipe import PulseStep, Recipe, SubCycle
from qcm_pak.residuals import (
    ResidualTrace,
    aggregate_traces,
    collect_residuals,
    summarize_residuals,
)

SIGMA = 0.05


def _curve(kind: str, t: np.ndarray) -> np.ndarray:
    if kind == "mono":
        return 10.0 * (1.0 - np.exp(-0.2 * t))
    if kind == "bi":
        return 6.0 * (1.0 - np.exp(-1.5 * t)) + 4.0 * (1.0 - np.exp(-0.05 * t))
    return 8.0 * (1.0 - np.exp(-0.3 * t))  # etch


def _step(
    name: str, kind: str, rng: np.random.Generator, t_max: float, dt: float,
    outer: int, sc: int = 0, run: int = 0, idx: int = 0, effect: str = "gain",
) -> StepResult:
    t = np.arange(0.0, t_max + 1e-9, dt)
    y = _curve(kind, t) + rng.normal(0, SIGMA, t.size)
    sign = 1.0 if effect == "gain" else -1.0
    return StepResult(
        step_name=name, mass_effect=effect, step_index=idx, sub_cycle_index=sc,  # type: ignore[arg-type]
        sub_cycle_run=run, outer_cycle=outer, time=t, mass_raw=sign * y,
        mass_corrected=sign * y, mass_change=float(sign * y[-1]),
    )


def _collection(
    recipe: Recipe, per_outer: list[list[tuple[int, int, StepResult]]]
) -> CycleCollection:
    cycles = []
    for outer, items in enumerate(per_outer):
        runs: dict[tuple[int, int], list[StepResult]] = {}
        for sc, run, s in items:
            runs.setdefault((sc, run), []).append(s)
        scrs = [
            SubCycleRun(sub_cycle_index=sc, run_number=run, outer_cycle=outer, steps=v)
            for (sc, run), v in runs.items()
        ]
        cycles.append(Cycle(cycle_number=outer, sub_cycle_runs=scrs))
    return CycleCollection(cycles=cycles, recipe=recipe)


def _ab_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 20.0), PulseStep("B", 0.1, 20.0)])
        ],
        repeats=6,
    )


def _ab(
    kind: str, effect: str = "gain", windows: tuple[float, float] = (20.0, 30.0)
) -> CycleCollection:
    rng = np.random.default_rng(1)
    per_outer = []
    for o in range(6):
        per_outer.append([
            (0, 0, _step("A", kind, rng, windows[0], 0.5, o, idx=0, effect=effect)),
            (0, 0, _step("B", kind, rng, windows[1], 0.5, o, idx=1, effect=effect)),
        ])
    return _collection(_ab_recipe(), per_outer)


@pytest.mark.parametrize("kind", ["mono", "bi"])
def test_langmuir_residuals_near_zero(kind: str) -> None:
    cc = _ab(kind)
    res = fit_langmuir(cc, "A", model=kind)  # type: ignore[arg-type]
    fr = res.residuals
    assert fr is not None and len(fr.traces) == 6
    for tr in fr.traces:
        assert tr.time[0] == 0.0
        np.testing.assert_allclose(tr.residual, tr.observed - tr.fitted)
        assert tr.rmse < 3 * SIGMA
    assert abs(float(np.mean(fr.stats.mean))) < 2 * SIGMA / np.sqrt(6)
    assert set(fr.stats.n.tolist()) == {6}
    assert fr.rmse < 3 * SIGMA


def test_etch_residuals_near_zero() -> None:
    cc = _ab("etch", effect="loss")
    for model in ("saturating", "linear"):
        res = fit_etch(cc, "A", model=model)  # type: ignore[arg-type]
        assert res.residuals is not None
        assert len(res.residuals.traces) == 6
    sat = fit_etch(cc, "A", model="saturating")
    assert sat.residuals is not None and sat.residuals.rmse < 3 * SIGMA


def test_residuals_do_not_change_fit_and_per_cycle_has_none() -> None:
    cc = _ab("mono")
    res = fit_langmuir(cc, "A")
    assert res.k == pytest.approx(0.2, rel=0.05)
    assert all(r.residuals is None for r in fit_langmuir_per_cycle(cc, "A"))


def test_ab_recipe_collect_and_unequal_windows() -> None:
    cc = _ab("mono", windows=(20.0, 30.0))
    mr = collect_residuals(cc, "langmuir_mono")
    assert [f.step_name for f in mr.fits] == ["A", "B"]
    # each step is fitted on its own window (shortest occurrence)
    assert mr.fits[0].traces[0].time[-1] == pytest.approx(20.0)
    assert mr.fits[1].traces[0].time[-1] == pytest.approx(30.0)
    st = mr.summary.stats
    assert st.time[-1] == pytest.approx(30.0)
    # n = 12 where both steps have data, 6 where only B does
    n_at = dict(zip(np.round(st.time, 6).tolist(), st.n.tolist(), strict=True))
    assert n_at[10.0] == 12 and n_at[20.0] == 12
    assert n_at[20.5] == 6 and n_at[30.0] == 6
    assert mr.summary.n_traces == 12
    assert abs(float(np.average(st.mean, weights=st.n))) < 3 * SIGMA / np.sqrt(6)
    assert [s.n_traces for s in mr.summary.steps] == [6, 6]


def test_super_cycle_keeps_runs_and_sub_cycles_distinct() -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 20.0)], repeats=2),
            SubCycle(steps=[PulseStep("A", 0.1, 20.0)], repeats=1),
        ],
        repeats=3,
    )
    rng = np.random.default_rng(2)
    per_outer = []
    for o in range(3):
        items = [
            (0, 0, _step("A", "bi", rng, 20.0, 0.5, o, sc=0, run=0)),
            (0, 1, _step("A", "bi", rng, 20.0, 0.5, o, sc=0, run=1)),
            (1, 0, _step("A", "bi", rng, 20.0, 0.5, o, sc=1, run=0)),
        ]
        per_outer.append(items)
    cc = _collection(recipe, per_outer)
    mr = collect_residuals(cc, "langmuir_bi", force=True)
    assert [f.sub_cycle for f in mr.fits] == [0, 1]
    keys = [
        (t.outer_cycle, t.sub_cycle_index, t.sub_cycle_run)
        for t in mr.fits[0].traces
    ]
    assert keys == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 0, 1)]
    assert mr.summary.n_traces == 9
    assert set(mr.summary.stats.n.tolist()) == {9}


def test_aggregate_uses_finest_grid_and_stored_residuals() -> None:
    def tr(dt: float, end: float, val: float) -> ResidualTrace:
        t = np.arange(0.0, end + 1e-9, dt)
        r = np.full_like(t, val)
        return ResidualTrace(0, 0, 0, 0, t, r, np.zeros_like(t), r, abs(val))

    st = aggregate_traces([tr(1.0, 10.0, 1.0), tr(0.5, 6.0, 3.0)])
    assert st.time[1] == pytest.approx(0.5)
    assert st.time[-1] == pytest.approx(10.0)
    assert st.mean[0] == pytest.approx(2.0)
    assert st.n[0] == 2 and st.n[-1] == 1
    assert st.std[0] == pytest.approx(np.sqrt(2.0))
    assert st.std[-1] is None
    assert st.mean[-1] == pytest.approx(1.0)


def test_summarize_accepts_results_and_serialises() -> None:
    cc = _ab("mono")
    sm = summarize_residuals([fit_langmuir(cc, "A"), fit_langmuir(cc, "B")])
    assert sm.n_traces == 12
    json.dumps(sm.to_dict(), allow_nan=False)
    json.dumps(fit_langmuir(cc, "A").to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        summarize_residuals([])
    with pytest.raises(ValueError):
        summarize_residuals(fit_langmuir_per_cycle(cc, "A"))


def test_collect_errors() -> None:
    cc = _ab("mono")
    with pytest.raises(KeyError):
        collect_residuals(cc, "nope")
    with pytest.raises(ValueError):
        collect_residuals(cc, "etch_linear")
