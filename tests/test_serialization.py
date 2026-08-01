"""Tests for JSON serialization: to_dict()/from_dict() across the public API."""

import json

import numpy as np
import pytest

from qcm_pak._types import AnalysisResult, MassDataset
from qcm_pak.detection import detect_pulses
from qcm_pak.extraction import extract_cycles
from qcm_pak.io import ColumnSpec
from qcm_pak.kinetics import fit_langmuir
from qcm_pak.parameters import ALDParameters, DetectionParameters, SauerbreyConstants
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _super_cycle_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[
            SubCycle(
                steps=[PulseStep("TMA", 0.1, 30.0), PulseStep("H2O", 0.1, 30.0)],
                repeats=3,
                label="AB growth",
            ),
            SubCycle(
                steps=[PulseStep("TMA", 0.1, 30.0, mass_effect="loss")],
                repeats=2,
                label="A passivation",
            ),
        ],
        repeats=4,
        start_time=120.0,
    )


def test_recipe_round_trip() -> None:
    recipe = _super_cycle_recipe()
    payload = recipe.to_dict()
    json.dumps(payload)  # must be plain JSON-safe types

    restored = Recipe.from_dict(payload)
    assert restored.repeats == recipe.repeats
    assert restored.start_time == recipe.start_time
    assert restored.step_names() == recipe.step_names()
    assert restored.sub_cycles[1].steps[0].mass_effect == "loss"
    assert restored.sub_cycles[0].label == "AB growth"


def test_pulse_step_round_trip() -> None:
    step = PulseStep("HF", pulse=0.1, purge=30.0, mass_effect="loss")
    restored = PulseStep.from_dict(step.to_dict())
    assert restored == step


def test_column_spec_round_trip() -> None:
    spec = ColumnSpec(
        freq_col="Sensor 2 Frequency [Hz]",
        time_col="Time",
        time_unit="clock",
        delimiter="\t",
        encoding="latin-1",
        header_rows=1,
    )
    restored = ColumnSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_column_spec_from_dict_ignores_unknown_keys() -> None:
    restored = ColumnSpec.from_dict({"freq_col": "f", "dt": 0.5, "bogus": 123})
    assert restored.freq_col == "f"
    assert restored.dt == 0.5


def test_sauerbrey_constants_round_trip() -> None:
    constants = SauerbreyConstants(overtone=3)
    restored = SauerbreyConstants.from_dict(constants.to_dict())
    assert restored == constants


def test_sauerbrey_constants_from_dict_uses_defaults_for_missing_keys() -> None:
    restored = SauerbreyConstants.from_dict({"overtone": 3})
    assert restored.overtone == 3
    assert restored.fundamental_frequency == SauerbreyConstants().fundamental_frequency


def test_detection_parameters_round_trip() -> None:
    det = DetectionParameters(method="hybrid", recipe_tolerance=0.2)
    restored = DetectionParameters.from_dict(det.to_dict())
    assert restored == det


def test_ald_parameters_round_trip(tmp_path) -> None:
    data_file = tmp_path / "run.csv"
    data_file.write_text("t,f\n0,5000000\n")
    params = ALDParameters(input_file=data_file, recipe=_super_cycle_recipe())

    restored = ALDParameters.from_dict(params.to_dict())
    assert restored.input_file == params.input_file
    assert restored.recipe.repeats == params.recipe.repeats


def _synthetic_result() -> AnalysisResult:
    n_cycles, dt, start_time, step_dur, mass_per_step = 3, 0.5, 10.0, 20.0, 1.5
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
    data = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)
    return recipe, data


@pytest.fixture
def analysis_result(tmp_path) -> AnalysisResult:
    recipe, data = _synthetic_result()
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(recipe_tolerance=0.4)
    index = detect_pulses(data, params, det)
    cycles = extract_cycles(data, index, params)
    return AnalysisResult(cycles=cycles, mass_data=data, cycle_index=index, params=params)


def test_analysis_result_to_dict_is_json_serializable(analysis_result: AnalysisResult) -> None:
    payload = analysis_result.to_dict()
    json.dumps(payload)  # must not raise

    assert len(payload["cycles"]["cycles"]) == 3
    assert payload["params"]["recipe"]["repeats"] == 3
    assert isinstance(payload["mass_data"]["mass"], list)


def test_analysis_result_to_bytes(analysis_result: AnalysisResult) -> None:
    files = analysis_result.to_bytes()

    assert set(files) == {
        "data/cycle_data_A.tsv",
        "data/cycle_data_B.tsv",
        "figures/full_trace.png",
        "figures/derivative_analysis.png",
    }
    assert files["figures/full_trace.png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert b"time_s" in files["data/cycle_data_A.tsv"]


def test_analysis_result_to_bytes_matches_save(analysis_result: AnalysisResult, tmp_path) -> None:
    """The in-memory path and the disk-writing path must produce identical bytes."""
    analysis_result.save(tmp_path)
    files = analysis_result.to_bytes()

    on_disk = (tmp_path / "data" / "cycle_data_A.tsv").read_bytes()
    assert on_disk == files["data/cycle_data_A.tsv"]


def test_langmuir_result_to_dict_is_json_serializable(analysis_result: AnalysisResult) -> None:
    result = fit_langmuir(analysis_result.cycles, step="A", model="mono")
    json.dumps(result.to_dict())
