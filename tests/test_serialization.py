"""Tests for to_dict() / from_dict() and the qcm_pak.serialization helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qcm_pak import serialization
from qcm_pak._types import (
    AnalysisResult,
    CycleIndex,
    EtchResult,
    LangmuirResult,
    MassDataset,
    PulseCorrection,
    QCMDataset,
)
from qcm_pak.detection import detect_pulses
from qcm_pak.extraction import extract_cycles
from qcm_pak.io import ColumnSpec
from qcm_pak.kinetics import fit_langmuir, fit_langmuir_per_cycle
from qcm_pak.models import list_models
from qcm_pak.parameters import ALDParameters, DetectionParameters, SauerbreyConstants
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _through_json(payload: Any) -> Any:  # noqa: ANN401
    """Encode as strict JSON (no NaN/Infinity) and decode again."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _super_cycle_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[
            SubCycle(
                steps=[PulseStep("TMA", 0.1, 30.0), PulseStep("H2O", 0.1, 30.0)],
                repeats=3,
                label="AB growth",
            ),
            SubCycle(
                steps=[PulseStep("HF", 0.5, 20.0, mass_effect="loss")],
                repeats=2,
                label="etch",
            ),
        ],
        repeats=4,
        start_time=120.0,
    )


def _analysis(tmp_path: Path) -> AnalysisResult:
    dt, step_dur, start, n_cycles = 0.5, 20.0, 10.0, 4
    time = np.arange(0, start + 2 * step_dur * n_cycles + 5.0, dt)
    mass = np.zeros(len(time))
    t_cur, cumulative = start, 0.0
    for _ in range(2 * n_cycles):
        cumulative += 1.5
        mass[int(round(t_cur / dt)):int(round((t_cur + step_dur) / dt))] = cumulative
        t_cur += step_dur
    mass += np.random.default_rng(0).normal(0, 0.01, len(mass))
    data = MassDataset(time=time, frequency=np.full(len(time), 6e6), mass=mass)
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", pulse=dt, purge=step_dur - dt),
            PulseStep("B", pulse=dt, purge=step_dur - dt),
        ])],
        repeats=n_cycles,
        start_time=start,
    )
    dummy = tmp_path / "trace.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    return AnalysisResult(extract_cycles(data, index, params), data, index, params)


# ── Configuration types: to_dict() → JSON → from_dict() ─────────────────────


def test_pulse_step_round_trip() -> None:
    step = PulseStep("HF", pulse=0.1, purge=30.0, mass_effect="loss")
    assert PulseStep.from_dict(_through_json(step.to_dict())) == step


def test_sub_cycle_round_trip() -> None:
    sc = SubCycle(steps=[PulseStep("A", 1, 5), PulseStep("B", 1, 5)], repeats=3,
                  label="AB")
    assert SubCycle.from_dict(_through_json(sc.to_dict())) == sc


def test_recipe_round_trip() -> None:
    recipe = _super_cycle_recipe()
    payload = recipe.to_dict()
    assert payload == {
        "sub_cycles": [
            {
                "steps": [
                    {"name": "TMA", "pulse": 0.1, "purge": 30.0, "mass_effect": "gain"},
                    {"name": "H2O", "pulse": 0.1, "purge": 30.0, "mass_effect": "gain"},
                ],
                "repeats": 3,
                "label": "AB growth",
            },
            {
                "steps": [
                    {"name": "HF", "pulse": 0.5, "purge": 20.0, "mass_effect": "loss"},
                ],
                "repeats": 2,
                "label": "etch",
            },
        ],
        "repeats": 4,
        "start_time": 120.0,
    }
    assert Recipe.from_dict(_through_json(payload)) == recipe


def test_recipe_from_dict_uses_defaults_for_missing_optional_keys() -> None:
    recipe = Recipe.from_dict(
        {"sub_cycles": [{"steps": [{"name": "A", "pulse": 1, "purge": 5}]}]}
    )
    assert recipe == Recipe(sub_cycles=[SubCycle(steps=[PulseStep("A", 1, 5)])])


@pytest.mark.parametrize(
    "spec",
    [
        ColumnSpec(freq_col="Sensor 2 Frequency [Hz]", time_col="Time",
                   time_unit="clock", delimiter="\t", encoding="latin-1",
                   header_rows=1, skiprows=8, drop_duplicate_times=False),
        ColumnSpec(freq_col=1, dt=0.1, freq_unit="MHz", temp_col=2),
    ],
)
def test_column_spec_round_trip(spec: ColumnSpec) -> None:
    assert ColumnSpec.from_dict(_through_json(spec.to_dict())) == spec


def test_sauerbrey_constants_round_trip() -> None:
    constants = SauerbreyConstants(fundamental_frequency=5.0e6, overtone=3)
    assert SauerbreyConstants.from_dict(_through_json(constants.to_dict())) == constants


@pytest.mark.parametrize(
    "det",
    [
        DetectionParameters(),
        DetectionParameters(method="hybrid", penalty=12.5, refinement_window=0.0,
                            baseline_window=8.0),
    ],
)
def test_detection_parameters_round_trip(det: DetectionParameters) -> None:
    assert DetectionParameters.from_dict(_through_json(det.to_dict())) == det


def test_ald_parameters_round_trip(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text("t,f\n")
    params = ALDParameters(input_file=trace, recipe=_super_cycle_recipe())
    payload = _through_json(params.to_dict())
    assert payload["input_file"] == str(trace)
    assert ALDParameters.from_dict(payload) == params


def test_pulse_correction_round_trip() -> None:
    for corr in (PulseCorrection(3, new_onset_idx=120),
                 PulseCorrection(5, excluded=True)):
        assert PulseCorrection.from_dict(_through_json(corr.to_dict())) == corr


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="penalt"):
        DetectionParameters.from_dict({"penalt": 5.0})


def test_from_dict_rejects_unknown_keys_in_nested_objects() -> None:
    with pytest.raises(ValueError, match="duration"):
        Recipe.from_dict(
            {"sub_cycles": [{"steps": [{"name": "A", "pulse": 1, "purge": 5,
                                        "duration": 6}]}]}
        )


def test_from_dict_names_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="purge"):
        PulseStep.from_dict({"name": "A", "pulse": 1.0})


def test_from_dict_still_runs_the_class_checks() -> None:
    with pytest.raises(ValueError, match="repeats"):
        Recipe.from_dict({"sub_cycles": [{"steps": [{"name": "A", "pulse": 1,
                                                     "purge": 5}]}], "repeats": 0})


def test_from_dict_rejects_a_non_mapping() -> None:
    with pytest.raises(TypeError, match="PulseStep"):
        PulseStep.from_dict(["A", 1, 5])  # type: ignore[arg-type]


# ── Data and result types: to_dict() is strict JSON ─────────────────────────


def test_qcm_dataset_to_dict(tmp_path: Path) -> None:
    ds = QCMDataset(time=np.array([0.0, 1.0]), frequency=np.array([6e6, 6e6 - 1]),
                    temperature=np.array([150.0, 150.1]),
                    source_path=tmp_path / "trace.csv")
    assert _through_json(ds.to_dict()) == {
        "time": [0.0, 1.0],
        "frequency": [6e6, 6e6 - 1],
        "temperature": [150.0, 150.1],
        "source_path": str(tmp_path / "trace.csv"),
    }


def test_mass_dataset_to_dict_turns_nan_into_null() -> None:
    ds = MassDataset(time=np.array([0.0, 1.0]), frequency=np.array([6e6, np.nan]),
                     mass=np.array([0.0, np.inf]))
    payload = _through_json(ds.to_dict())
    assert payload["frequency"] == [6e6, None]
    assert payload["mass"] == [0.0, None]
    assert payload["temperature"] is None


def test_cycle_index_to_dict() -> None:
    index = CycleIndex(step_onsets=[("A", np.int64(10)), ("B", 60)],
                       recipe=_super_cycle_recipe(),
                       confidence=[np.float64(0.9), 0.5], excluded=[False, True])
    payload = _through_json(index.to_dict())
    assert payload["step_onsets"] == [["A", 10], ["B", 60]]
    assert payload["recipe"] == _super_cycle_recipe().to_dict()
    assert payload["confidence"] == [0.9, 0.5]
    assert payload["excluded"] == [False, True]


def test_fit_results_to_dict() -> None:
    lr = LangmuirResult(step_name="A", model="mono", r_squared=float("nan"), k=0.1,
                        theta_max=12.0, covariance=np.eye(2), outer_cycle=3)
    payload = _through_json(lr.to_dict())
    assert payload["r_squared"] is None
    assert payload["k"] == 0.1
    assert payload["covariance"] == [[1.0, 0.0], [0.0, 1.0]]
    assert payload["outer_cycle"] == 3

    er = EtchResult(step_name="HF", model="linear", r_squared=0.9, rate=0.02)
    assert _through_json(er.to_dict())["rate"] == 0.02


def test_model_spec_to_dict_leaves_out_the_fit_functions() -> None:
    payload = _through_json([m.to_dict() for m in list_models()])
    assert payload[0] == {"id": "langmuir_mono", "display_name": "Monomodal Langmuir",
                          "category": "growth", "mass_effect": "gain"}
    assert [m["id"] for m in payload] == [m.id for m in list_models()]


def test_analysis_result_to_dict(tmp_path: Path) -> None:
    result = _analysis(tmp_path)
    payload = _through_json(result.to_dict())
    assert set(payload) == {"cycles", "mass_data", "cycle_index", "params"}
    assert payload["mass_data"]["mass"] == result.mass_data.mass.tolist()
    assert len(payload["cycles"]["cycles"]) == len(result.cycles)
    step = payload["cycles"]["cycles"][1]["sub_cycle_runs"][0]["steps"][0]
    original = result.cycles.cycles[1].sub_cycle_runs[0].steps[0]
    assert step["step_name"] == original.step_name
    assert step["mass_corrected"] == original.mass_corrected.tolist()
    assert step["mass_change"] == original.mass_change
    assert payload["params"]["recipe"] == result.params.recipe.to_dict()


def test_fit_dicts_work_with_the_full_export(tmp_path: Path) -> None:
    from qcm_pak.export import build_analysis_report, save_full_export

    r = _analysis(tmp_path)
    fits = _through_json(serialization.to_jsonable(
        {"A": {"langmuir_mono": fit_langmuir(r.cycles, "A", model="mono")}}
    ))
    per_cycle = _through_json(serialization.to_jsonable(
        {"A": {"langmuir_mono": fit_langmuir_per_cycle(r.cycles, "A", model="mono")}}
    ))
    assert "langmuir_mono" in build_analysis_report(
        r.mass_data, r.cycle_index, r.cycles, r.params, fits
    )
    out = tmp_path / "out"
    save_full_export(out, r.mass_data, r.cycle_index, r.cycles, r.params, fits,
                     per_cycle, cycle_batch_size=2)
    assert (out / "figures" / "modelling" / "A_langmuir_mono.png").is_file()
    assert (out / "figures" / "modelling" / "A_langmuir_mono_k_drift.png").is_file()


# ── The helper itself ───────────────────────────────────────────────────────


def test_to_jsonable_converts_numpy_scalars_paths_and_tuples() -> None:
    value = {"n": np.int32(4), "x": np.float32(0.5), "ok": np.bool_(True),
             "p": Path("a") / "b.csv", "t": (1, 2), 3: "int key"}
    assert serialization.to_jsonable(value) == {
        "n": 4, "x": 0.5, "ok": True, "p": str(Path("a") / "b.csv"), "t": [1, 2],
        "3": "int key",
    }
    assert type(serialization.to_jsonable(np.int32(4))) is int
    assert type(serialization.to_jsonable(np.bool_(True))) is bool


def test_to_jsonable_uses_a_dataclass_own_to_dict() -> None:
    spec = list_models()[0]
    assert serialization.to_jsonable([spec]) == [spec.to_dict()]


def test_to_jsonable_rejects_values_json_cannot_hold() -> None:
    with pytest.raises(TypeError, match="function"):
        serialization.to_jsonable({"f": len})
