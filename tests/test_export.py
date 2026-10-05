"""Tests for export.py — save_full_export and build_analysis_report."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from qcm_pak._types import MassDataset  # noqa: E402
from qcm_pak.detection import detect_pulses  # noqa: E402
from qcm_pak.export import build_analysis_report, save_full_export  # noqa: E402
from qcm_pak.extraction import extract_cycles  # noqa: E402
from qcm_pak.kinetics import fit_langmuir, fit_langmuir_per_cycle  # noqa: E402
from qcm_pak.parameters import ALDParameters, DetectionParameters  # noqa: E402
from qcm_pak.recipe import PulseStep, Recipe, SubCycle  # noqa: E402


def _synthetic_ald(n_cycles: int = 6, dt: float = 0.5, start_time: float = 10.0,
                    step_dur: float = 20.0, mass_per_step: float = 1.5):
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
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass), recipe


def _built_pipeline(tmp_path, n_cycles=6):
    data, recipe = _synthetic_ald(n_cycles=n_cycles)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    cycles = extract_cycles(data, index, params)
    return data, params, index, cycles


def _fit_dict(result):
    return {f.name: getattr(result, f.name) for f in fields(result) if f.name != "covariance"}


def test_build_analysis_report_without_fits(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path)
    report = build_analysis_report(data, index, cycles, params)
    assert "QCM-PAK Analysis Report" in report
    assert "Fits: (none run yet)" in report
    assert "A, B" in report


def test_build_analysis_report_with_fits(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path)
    result = fit_langmuir(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": _fit_dict(result)}}
    report = build_analysis_report(data, index, cycles, params, fits)
    assert "langmuir_mono" in report
    assert "R²=" in report


def test_save_full_export_writes_core_files(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, cycle_batch_size=3)

    assert (out / "analysis_report.txt").is_file()
    assert (out / "figures" / "full_trace.png").is_file()
    assert (out / "figures" / "derivative_analysis.png").is_file()
    assert (out / "figures" / "detailed_cycles_first.png").is_file()
    assert (out / "figures" / "detailed_cycles_middle.png").is_file()
    assert (out / "figures" / "detailed_cycles_last.png").is_file()
    assert (out / "data" / "cycle_data_A.tsv").is_file()
    assert (out / "data" / "pulse_times_A.tsv").is_file()
    assert (out / "data" / "pulse_confidence.tsv").is_file()


def test_save_full_export_batches_cycle_figures(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, cycle_batch_size=3)

    # 6 cycles / batch_size 3 -> two batches
    assert (out / "figures" / "cycles_0001-0003" / "A_cycles.png").is_file()
    assert (out / "figures" / "cycles_0001-0003" / "A_average.png").is_file()
    assert (out / "figures" / "cycles_0001-0003" / "A_timing.png").is_file()
    assert (out / "figures" / "cycles_0004-0006" / "B_cycles.png").is_file()


def test_save_full_export_with_fits_writes_modelling_figures_and_tsvs(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    per_cycle = fit_langmuir_per_cycle(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": _fit_dict(result)}}
    fits_per_cycle = {"A": {"langmuir_mono": [_fit_dict(r) for r in per_cycle]}}

    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, fits_per_cycle, cycle_batch_size=3)

    assert (out / "figures" / "modelling" / "A_langmuir_mono.png").is_file()
    assert (out / "figures" / "modelling" / "A_langmuir_mono_k_drift.png").is_file()
    assert (out / "data" / "A_langmuir_mono_per_cycle.tsv").is_file()


def test_save_full_export_accepts_dataclass_fits_directly(tmp_path) -> None:
    # fits/fits_per_cycle should also work with the native dataclasses, not
    # just JSON-safe dicts — the SimpleNamespace adapter in save_full_export
    # is only needed for the dict case.
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": result}}

    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, cycle_batch_size=3)
    assert (out / "figures" / "modelling" / "A_langmuir_mono.png").is_file()
