"""Tests for growth.py — growth-switch detection on synthetic series only."""

from pathlib import Path

import numpy as np
import pytest

from qcm_pak import (
    ALDParameters,
    AnalysisResult,
    Cycle,
    CycleCollection,
    CycleIndex,
    MassDataset,
    PulseStep,
    Recipe,
    StepResult,
    SubCycle,
    SubCycleRun,
)
from qcm_pak.exceptions import QCMPakError
from qcm_pak.growth import (
    GrowthSwitchResult,
    detect_growth_switch,
    estimate_cycle_noise,
    find_growth_switch,
)

SIGMA = 0.2


def _series(
    n_before: int,
    level_before: float,
    n_after: int,
    level_after: float,
    sigma: float = SIGMA,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.r_[np.full(n_before, level_before), np.full(n_after, level_after)]
    return base + rng.normal(0.0, sigma, base.size)


def test_flat_then_growth() -> None:
    y = _series(20, 0.0, 40, 3.0)
    times = np.arange(y.size) * 60.0 + 100.0
    r = find_growth_switch(y, times)
    assert r.found
    assert r.switch_cycle == 20
    assert r.switch_time == pytest.approx(100.0 + 20 * 60.0)
    assert r.incubation_time == pytest.approx(20 * 60.0)
    assert r.growth_before == pytest.approx(0.0, abs=0.2)
    assert r.growth_after == pytest.approx(3.0, abs=0.2)
    assert r.noise_sigma == pytest.approx(SIGMA, rel=0.4)


def test_loss_then_growth() -> None:
    y = _series(15, -2.0, 30, 2.5)
    r = find_growth_switch(y)
    assert r.found
    assert r.switch_cycle == 15
    assert r.switch_time is None and r.incubation_time is None
    assert r.growth_before == pytest.approx(-2.0, abs=0.2)
    assert r.growth_after == pytest.approx(2.5, abs=0.2)
    assert any("lost" in n for n in r.notes)


def test_immediate_growth_has_no_incubation() -> None:
    y = _series(0, 0.0, 30, 2.0)
    r = find_growth_switch(y, np.arange(30) * 10.0)
    assert r.found
    assert r.switch_cycle == 0
    assert r.incubation_time == 0.0
    assert r.growth_before is None
    assert r.growth_after == pytest.approx(2.0, abs=0.2)


def test_never_grows_flat() -> None:
    r = find_growth_switch(_series(50, 0.0, 0, 0.0))
    assert not r.found
    assert r.switch_cycle is None and r.growth_after is None
    assert any("flat or loses" in n for n in r.notes)


def test_never_grows_loss() -> None:
    r = find_growth_switch(_series(50, -1.5, 0, 0.0))
    assert not r.found
    assert r.growth_before == pytest.approx(-1.5, abs=0.2)
    assert any("flat or loses" in n for n in r.notes)


def test_noisy_series_still_finds_switch() -> None:
    y = _series(25, 0.0, 50, 4.0, sigma=1.0, seed=3)
    r = find_growth_switch(y)
    assert r.found
    assert r.switch_cycle == pytest.approx(25, abs=2)


def test_pure_noise_does_not_trigger() -> None:
    for seed in range(20):
        assert not find_growth_switch(_series(100, 0.0, 0, 0.0, seed=seed)).found


def test_single_outlier_cycle_does_not_trigger() -> None:
    y = _series(40, 0.0, 0, 0.0)
    y[10] = 25.0
    r = find_growth_switch(y)
    assert not r.found


def test_outlier_before_real_switch_is_ignored() -> None:
    y = _series(20, 0.0, 30, 3.0)
    y[5] = 25.0
    assert find_growth_switch(y).switch_cycle == 20


def test_dip_inside_growth_delays_switch_until_run_is_long_enough() -> None:
    y = _series(10, 0.0, 30, 3.0)
    y[13] = 0.0  # breaks the first run; next full run starts at cycle 14
    assert find_growth_switch(y, n_consecutive=5).switch_cycle == 14
    assert find_growth_switch(y, n_consecutive=3).switch_cycle == 10


def test_n_consecutive_is_adjustable() -> None:
    y = np.r_[_series(20, 0.0, 0, 0.0), _series(0, 0.0, 3, 3.0, seed=1), np.zeros(20)]
    y[23:] += np.random.default_rng(2).normal(0, SIGMA, 20)
    assert not find_growth_switch(y, n_consecutive=5).found
    assert find_growth_switch(y, n_consecutive=3).switch_cycle == 20


def test_nan_cycle_breaks_a_run() -> None:
    y = _series(10, 0.0, 30, 3.0)
    y[12] = np.nan
    assert find_growth_switch(y).switch_cycle == 13


def test_too_few_cycles() -> None:
    r = find_growth_switch([1.0, 2.0, 3.0])
    assert not r.found
    assert r.notes


def test_noise_estimate_ignores_level_step_and_outlier() -> None:
    y = _series(30, 0.0, 30, 5.0)
    y[7] = 40.0
    assert estimate_cycle_noise(y) == pytest.approx(SIGMA, rel=0.4)


def test_invalid_arguments() -> None:
    y = _series(10, 0.0, 10, 1.0)
    with pytest.raises(QCMPakError):
        find_growth_switch(y, n_consecutive=0)
    with pytest.raises(QCMPakError):
        find_growth_switch(y, noise_factor=-1.0)
    with pytest.raises(QCMPakError):
        find_growth_switch(y, [0.0, 1.0])


def test_does_not_modify_input() -> None:
    y = _series(10, 0.0, 10, 3.0)
    before = y.copy()
    find_growth_switch(y)
    np.testing.assert_array_equal(y, before)


def _analysis_result(
    tmp_path: Path, per_cycle: list[float], exclude_cycle: int | None = None
) -> AnalysisResult:
    data_file = tmp_path / "synthetic.tsv"
    data_file.write_text("time\tfrequency\n")
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("A", pulse=1.0, purge=9.0)])],
        repeats=len(per_cycle),
        start_time=0.0,
    )
    n = len(per_cycle)
    cycles = [
        Cycle(
            cycle_number=i,
            sub_cycle_runs=[
                SubCycleRun(
                    sub_cycle_index=0,
                    run_number=0,
                    outer_cycle=i,
                    steps=[
                        StepResult(
                            step_name="A",
                            mass_effect="any",
                            step_index=0,
                            sub_cycle_index=0,
                            sub_cycle_run=0,
                            outer_cycle=i,
                            time=np.zeros(2),
                            mass_raw=np.zeros(2),
                            mass_corrected=np.zeros(2),
                            mass_change=m,
                            excluded=(i == exclude_cycle),
                        )
                    ],
                )
            ],
        )
        for i, m in enumerate(per_cycle)
    ]
    time = np.arange(n * 10, dtype=np.float64)
    zeros = np.zeros_like(time)
    mass = MassDataset(time=time, frequency=zeros, mass=zeros)
    index = CycleIndex(step_onsets=[("A", 10 * i) for i in range(n)], recipe=recipe)
    return AnalysisResult(
        cycles=CycleCollection(cycles=cycles, recipe=recipe),
        mass_data=mass,
        cycle_index=index,
        params=ALDParameters(input_file=data_file, recipe=recipe),
    )


def test_detect_growth_switch_on_analysis_result(tmp_path: Path) -> None:
    res = _analysis_result(tmp_path, list(_series(12, 0.0, 30, 3.0)))
    r = detect_growth_switch(res)
    assert isinstance(r, GrowthSwitchResult)
    assert r.switch_cycle == 12
    assert r.switch_time == pytest.approx(120.0)
    assert r.incubation_time == pytest.approx(120.0)


def test_detect_growth_switch_treats_excluded_cycle_as_missing(
    tmp_path: Path,
) -> None:
    res = _analysis_result(
        tmp_path, list(_series(12, 0.0, 30, 3.0)), exclude_cycle=14
    )
    assert detect_growth_switch(res).switch_cycle == 15
