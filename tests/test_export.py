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


def test_saving_leaves_no_figures_registered_with_pyplot(tmp_path) -> None:
    import matplotlib.pyplot as plt

    from qcm_pak._types import AnalysisResult

    data, params, index, cycles = _built_pipeline(tmp_path)
    fits = {"A": {"langmuir_mono": _fit_dict(fit_langmuir(cycles, "A", model="mono"))}}
    before = plt.get_fignums()
    AnalysisResult(cycles, data, index, params).save(tmp_path / "save")
    save_full_export(tmp_path / "full", data, index, cycles, params, fits,
                     cycle_batch_size=3)
    assert plt.get_fignums() == before


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


_OLD_ZIP_FILES = [
    "analysis_report.txt",
    "figures/full_trace.png",
    "figures/derivative_analysis.png",
    "figures/detailed_cycles_first.png",
    "figures/detailed_cycles_middle.png",
    "figures/detailed_cycles_last.png",
    "figures/cycles_0001-0003/A_cycles.png",
    "figures/cycles_0001-0003/A_average.png",
    "figures/cycles_0001-0003/A_timing.png",
    "figures/modelling/A_langmuir_mono.png",
    "figures/modelling/A_langmuir_mono_k_drift.png",
    "figures/modelling/A_langmuir_mono_theta_max_drift.png",
    "data/cycle_data_A.tsv",
    "data/pulse_times_A.tsv",
    "data/pulse_confidence.tsv",
    "data/A_langmuir_mono_per_cycle.tsv",
]


def _export_with_traces(tmp_path, as_dicts: bool):
    import pandas as pd

    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    per_cycle = fit_langmuir_per_cycle(cycles, "A", model="mono")
    assert result.residuals is not None
    fit = _fit_dict(result)
    if as_dicts:  # what the web API passes: JSON-safe dicts
        fit["residuals"] = result.residuals.to_dict()
    fits = {"A": {"langmuir_mono": fit}}
    fits_per_cycle = {"A": {"langmuir_mono": [_fit_dict(r) for r in per_cycle]}}
    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, fits_per_cycle,
                     cycle_batch_size=3)
    return out, result, pd


@pytest.mark.parametrize("as_dicts", [False, True])
def test_export_keeps_old_files_and_adds_fit_loss_traces(tmp_path, as_dicts) -> None:
    out, result, pd = _export_with_traces(tmp_path, as_dicts)
    for name in _OLD_ZIP_FILES:
        assert (out / name).is_file(), name

    loss = pd.read_csv(out / "data" / "fit_loss_A_langmuir_mono.tsv", sep="\t")
    assert list(loss.columns) == [
        "outer_cycle", "sub_cycle_run", "time_s", "data_ng_cm2",
        "fitted_ng_cm2", "residual_fit_minus_data_ng_cm2",
    ]
    assert len(loss) == sum(len(t.time) for t in result.residuals.traces)
    # fit minus data, the opposite sign of the library's data - fit
    np.testing.assert_allclose(
        loss["residual_fit_minus_data_ng_cm2"],
        loss["fitted_ng_cm2"] - loss["data_ng_cm2"], atol=1e-4,
    )
    avg = pd.read_csv(out / "data" / "fit_loss_average.tsv", sep="\t")
    assert list(avg.columns) == [
        "step", "model", "time_s", "mean_fit_minus_data_ng_cm2",
        "sd_ng_cm2", "n_traces",
    ]
    assert set(avg["step"]) == {"A"}
    assert (out / "figures" / "modelling" / "A_langmuir_mono_fit_loss.png").is_file()


def test_export_writes_mass_gain_traces(tmp_path) -> None:
    out, _, pd = _export_with_traces(tmp_path, as_dicts=False)
    trace = pd.read_csv(out / "data" / "mass_gain_traces.tsv", sep="\t")
    assert list(trace.columns) == [
        "outer_cycle", "time_s", "time_in_cycle_s", "mass_ng_cm2",
        "mass_change_in_cycle_ng_cm2", "cumulative_growth_ng_cm2", "partial",
    ]
    assert sorted(trace["outer_cycle"].unique()) == [1, 2, 3, 4, 5, 6]
    growth = pd.read_csv(out / "data" / "cumulative_growth.tsv", sep="\t")
    assert list(growth.columns) == [
        "outer_cycle", "t_start_s", "t_end_s", "delta_ng_cm2",
        "cumulative_ng_cm2", "partial",
    ]
    assert len(growth) == 6
    # the synthetic run adds 3 ng/cm² a cycle (A + B, 1.5 each)
    np.testing.assert_allclose(growth["delta_ng_cm2"].iloc[:5], 3.0, atol=0.2)
    np.testing.assert_allclose(
        growth["cumulative_ng_cm2"], growth["delta_ng_cm2"].cumsum(), atol=1e-6
    )
    assert (out / "figures" / "mass_change_per_cycle.png").is_file()
    assert (out / "figures" / "cumulative_growth.png").is_file()
