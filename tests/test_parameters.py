"""Tests for parameters.py."""

import pytest

from qcm_pak.exceptions import DataLoadError
from qcm_pak.parameters import ALDParameters, DetectionParameters, SauerbreyConstants
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _simple_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10)])],
        repeats=1,
    )


def test_sauerbrey_default_conversion_factor() -> None:
    constants = SauerbreyConstants()
    cf = constants.conversion_factor
    # Standard 5 MHz AT-cut quartz: C ≈ -17.7 ng/cm²/Hz (Kanazawa-Gordon)
    assert cf == pytest.approx(-17.7, rel=0.01)


def test_sauerbrey_is_frozen() -> None:
    constants = SauerbreyConstants()
    with pytest.raises((AttributeError, TypeError)):
        constants.overtone = 3  # type: ignore[misc]


def test_ald_parameters_missing_file() -> None:
    with pytest.raises(DataLoadError, match="not found"):
        ALDParameters(input_file="nonexistent_file.csv", recipe=_simple_recipe())


def test_detection_parameters_defaults() -> None:
    dp = DetectionParameters()
    assert dp.method == "pelt_guided"
    assert dp.penalty == "auto"
    assert dp.recipe_tolerance == pytest.approx(0.3)


def test_detection_parameters_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="recipe_tolerance"):
        DetectionParameters(recipe_tolerance=1.5)


def test_detection_parameters_invalid_spacing() -> None:
    with pytest.raises(ValueError, match="min_pulse_spacing"):
        DetectionParameters(min_pulse_spacing=0)
