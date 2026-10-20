"""Tests for the ``autocorr`` detection method."""

import pathlib

import numpy as np
import pytest

from qcm_pak._types import MassDataset
from qcm_pak.detection import FLOOR_CONFIDENCE, detect_pulses
from qcm_pak.exceptions import DetectionError
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle

DT = 0.1
STEP_S = 20.0


def _recipe(start: float, n_cycles: int, etch: bool = False) -> Recipe:
    second = "loss" if etch else "gain"
    return Recipe(
        sub_cycles=[
            SubCycle(
                steps=[
                    PulseStep("A", 1.0, 19.0),
                    PulseStep("B", 1.0, 19.0, mass_effect=second),
                ]
            )
        ],
        repeats=n_cycles,
        start_time=start,
    )


def _trace(
    n_cycles: int = 20,
    offset: float = 1.5,
    drift: float = 0.002,
    skip: set[int] | None = None,
    repeat: int = 1,
    seed: int = 4,
    etch: bool = False,
) -> tuple[MassDataset, list[float]]:
    """A 10 Hz trace whose true onsets sit ``offset`` s (plus drift) off the recipe."""
    rng = np.random.default_rng(seed)
    start = 10.0
    n_events = 2 * n_cycles
    time = np.arange(0, start + STEP_S * n_events + 20, DT)
    truth = [start + offset + (1 + drift) * STEP_S * k for k in range(n_events)]
    mass = np.zeros(len(time))
    for k, t_on in enumerate(truth):
        if skip and k in skip:
            continue
        rise = np.clip(time - t_on, 0, None)
        sign = -1.0 if etch and k % 2 else 1.0
        mass += sign * (1.0 - np.exp(-rise / 0.15))
    mass += rng.normal(0, 0.02, len(time))
    if repeat > 1:
        time = np.arange(len(time) * repeat) * (DT / repeat)
        mass = np.repeat(mass, repeat)
    data = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)
    return data, truth


def _params(
    tmp_path: pathlib.Path, n_cycles: int = 20, etch: bool = False
) -> ALDParameters:
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    return ALDParameters(input_file=dummy, recipe=_recipe(10.0, n_cycles, etch))


def test_autocorr_is_selectable_and_not_the_default() -> None:
    assert DetectionParameters().method == "pelt_guided"
    assert DetectionParameters(method="autocorr").method == "autocorr"


def test_finds_onsets_through_a_start_offset_and_drift(
    tmp_path: pathlib.Path,
) -> None:
    data, truth = _trace()
    cycle = detect_pulses(
        data, _params(tmp_path), DetectionParameters(method="autocorr")
    )
    found = np.array([data.time[i] for _, i in cycle.step_onsets])
    err = found - np.array(truth)
    assert len(found) == len(truth)
    # Last baseline sample: at or just before the rise, never after it.
    assert np.all(err <= 0.05)
    assert np.all(err >= -DT - 0.15)
    assert cycle.confidence is not None
    assert np.mean(np.array(cycle.confidence) > FLOOR_CONFIDENCE) >= 0.95


def test_a_missing_pulse_is_flagged_and_the_rest_are_not(
    tmp_path: pathlib.Path,
) -> None:
    data, truth = _trace(skip={17})
    cycle = detect_pulses(
        data, _params(tmp_path), DetectionParameters(method="autocorr")
    )
    assert cycle.confidence is not None
    conf = np.array(cycle.confidence)
    assert conf[17] == FLOOR_CONFIDENCE
    assert np.sum(conf <= FLOOR_CONFIDENCE) <= 3
    # The flagged pulse is parked on the schedule, so the count still holds.
    assert len(cycle.step_onsets) == len(truth)


def test_a_start_error_beyond_capture_range_flags_everything(
    tmp_path: pathlib.Path,
) -> None:
    # One whole step out: the |signal| comb sees nothing wrong, but every
    # class now sits on its neighbour's pulses, which have the other sign.
    data, _ = _trace(offset=20.0, drift=0.0, etch=True)
    cycle = detect_pulses(
        data,
        _params(tmp_path, etch=True),
        DetectionParameters(method="autocorr"),
    )
    assert cycle.confidence is not None
    assert all(c == FLOOR_CONFIDENCE for c in cycle.confidence)


def test_stair_stepped_data_gives_the_same_onsets_as_plain_data(
    tmp_path: pathlib.Path,
) -> None:
    plain, _ = _trace(n_cycles=12)
    stairs, _ = _trace(n_cycles=12, repeat=10)
    params = _params(tmp_path, n_cycles=12)
    det = DetectionParameters(method="autocorr")
    a = detect_pulses(plain, params, det)
    b = detect_pulses(stairs, params, det)
    t_plain = np.array([plain.time[i] for _, i in a.step_onsets])
    t_stair = np.array([stairs.time[i] for _, i in b.step_onsets])
    assert np.max(np.abs(t_plain - t_stair)) <= 0.35


def test_a_trace_without_pulses_does_not_crash_and_flags_more(
    tmp_path: pathlib.Path,
) -> None:
    rng = np.random.default_rng(0)
    time = np.arange(0, 500, DT)
    noise = MassDataset(
        time=time,
        frequency=np.full(len(time), 5e6),
        mass=rng.normal(0, 0.02, len(time)),
    )
    det = DetectionParameters(method="autocorr")
    params = _params(tmp_path, 10)
    try:
        cycle = detect_pulses(noise, params, det)
    except DetectionError:
        return
    clean, _ = _trace(n_cycles=10)
    good = detect_pulses(clean, params, det)
    assert cycle.confidence is not None and good.confidence is not None
    flagged = np.mean(np.array(cycle.confidence) <= FLOOR_CONFIDENCE)
    assert flagged > np.mean(np.array(good.confidence) <= FLOOR_CONFIDENCE)


def test_search_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        DetectionParameters(method="autocorr", autocorr_search_frac=0.6)
    with pytest.raises(ValueError):
        DetectionParameters(method="autocorr", autocorr_iterations=0)
