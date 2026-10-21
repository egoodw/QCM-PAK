"""A recipe that does not fit the data is reported, with a suggested fix."""

import pathlib

import numpy as np
import pytest

from qcm_pak._types import CycleIndex, MassDataset
from qcm_pak.detection import detect_pulses
from qcm_pak.exceptions import DetectionError, RecipeMismatchError
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle
from qcm_pak.recipe_check import check_recipe, expected_onsets

DT = 0.1
STEP_S = 20.0
CYCLES = 30
METHODS = ["hybrid", "pelt_guided", "autocorr"]


def _recipe(cycles: int = CYCLES, purge: float = 19.0, start: float = 10.0) -> Recipe:
    return Recipe(
        sub_cycles=[
            SubCycle(
                steps=[
                    PulseStep("A", 1.0, purge),
                    PulseStep("B", 1.0, purge, mass_effect="loss"),
                ]
            )
        ],
        repeats=cycles,
        start_time=start,
    )


def _data(cycles: int = CYCLES, seed: int = 2) -> MassDataset:
    rng = np.random.default_rng(seed)
    time = np.arange(0, 10.0 + STEP_S * 2 * cycles + 15, DT)
    mass = np.zeros(len(time))
    for k in range(2 * cycles):
        rise = np.clip(time - (11.0 + STEP_S * k), 0, None)
        mass += (-1.0 if k % 2 else 1.0) * (1.0 - np.exp(-rise / 0.15))
    mass += rng.normal(0, 0.02, len(time))
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)


def _params(tmp_path: pathlib.Path, recipe: Recipe) -> ALDParameters:
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    return ALDParameters(input_file=dummy, recipe=recipe)


def test_a_matching_recipe_has_no_warnings(tmp_path: pathlib.Path) -> None:
    data = _data()
    assert check_recipe(data, _recipe()) == []
    for method in METHODS:
        index = detect_pulses(
            data, _params(tmp_path, _recipe()), DetectionParameters(method=method)
        )
        assert index.recipe_warnings == []


def test_a_matching_recipe_on_a_long_trace_has_no_warnings() -> None:
    cycles = 150
    assert check_recipe(_data(cycles=cycles), _recipe(cycles=cycles)) == []


def test_expected_onsets_follow_the_recipe() -> None:
    times = expected_onsets(_recipe(cycles=3))
    assert times.tolist() == [10.0 + STEP_S * k for k in range(6)]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("cycles", "purge"), [(CYCLES + 20, 19.0), (CYCLES, 19.0 + 3.0)]
)
def test_a_recipe_longer_than_the_data_is_refused_with_a_fix(
    tmp_path: pathlib.Path, method: str, cycles: int, purge: float
) -> None:
    data = _data()
    params = _params(tmp_path, _recipe(cycles=cycles, purge=purge))
    with pytest.raises(RecipeMismatchError) as caught:
        detect_pulses(data, params, DetectionParameters(method=method))
    assert isinstance(caught.value, DetectionError)
    error = next(w for w in caught.value.warnings if w.severity == "error")
    assert error.code == "data_ends_early"
    assert error.suggestion is not None
    assert error.suggestion["repeats"] == CYCLES
    if purge != 19.0:
        assert error.suggestion["purge_delta_s"] == pytest.approx(-3.0, abs=0.1)


@pytest.mark.parametrize("method", METHODS)
def test_a_purge_that_is_too_short_is_flagged_with_the_right_length(
    tmp_path: pathlib.Path, method: str
) -> None:
    data = _data()
    params = _params(tmp_path, _recipe(purge=19.0 - 1.0))
    index = detect_pulses(data, params, DetectionParameters(method=method))
    drift = next(w for w in index.recipe_warnings if w.code == "timeline_drift")
    assert drift.suggestion is not None
    assert drift.suggestion["purge_delta_s"] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("method", METHODS)
def test_too_few_cycles_is_flagged_with_the_right_count(
    tmp_path: pathlib.Path, method: str
) -> None:
    data = _data()
    params = _params(tmp_path, _recipe(cycles=CYCLES - 10))
    index = detect_pulses(data, params, DetectionParameters(method=method))
    assert len(index.step_onsets) == 2 * (CYCLES - 10)
    codes = {w.code: w for w in index.recipe_warnings}
    mismatch = codes["pulse_count_mismatch"]
    assert mismatch.suggestion == {"repeats": CYCLES}


def test_a_start_time_far_from_the_first_pulse_is_flagged() -> None:
    found = check_recipe(_data(), _recipe(start=40.0))
    offset = next(w for w in found if w.code == "start_offset")
    assert offset.suggestion is not None
    assert offset.suggestion["start_time"] == pytest.approx(11.0, abs=0.5)


def test_a_small_start_error_is_not_flagged() -> None:
    assert check_recipe(_data(), _recipe(start=12.0)) == []


def test_warnings_are_json_safe_and_survive_corrections(
    tmp_path: pathlib.Path,
) -> None:
    from qcm_pak._types import PulseCorrection
    from qcm_pak.corrections import apply_corrections

    data = _data()
    params = _params(tmp_path, _recipe(cycles=CYCLES - 10))
    index = detect_pulses(data, params, DetectionParameters(method="hybrid"))
    corrected = apply_corrections(index, [PulseCorrection(pulse_id=0, excluded=True)])
    assert corrected.recipe_warnings == index.recipe_warnings
    out = index.to_dict()
    assert out["recipe_warnings"][0]["code"] == "pulse_count_mismatch"


def test_a_cycle_index_without_warnings_has_an_empty_list() -> None:
    assert CycleIndex(step_onsets=[], recipe=_recipe()).recipe_warnings == []
