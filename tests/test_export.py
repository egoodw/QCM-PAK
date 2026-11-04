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
    assert (out / "README.txt").is_file()
    t = out / "figures" / "1_trace"
    for name in ("full_trace", "derivative_analysis", "detailed_cycles_first",
                 "detailed_cycles_middle", "detailed_cycles_last"):
        assert (t / f"{name}.png").is_file(), name
    assert (out / "data" / "raw" / "full_trace.csv").is_file()
    assert (out / "data" / "traces" / "A_mass.csv").is_file()
    assert (out / "data" / "fit_summary.csv").is_file()


def test_save_full_export_batches_cycle_figures(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, cycle_batch_size=3)

    # 6 cycles / batch_size 3 -> two batches
    assert (out / "figures" / "2_pulse_detection" / "cycles_0001-0003" / "A_cycles.png").is_file()
    assert (out / "figures" / "2_pulse_detection" / "cycles_0001-0003" / "A_average.png").is_file()
    assert (out / "figures" / "2_pulse_detection" / "cycles_0001-0003" / "A_timing.png").is_file()
    assert (out / "figures" / "2_pulse_detection" / "cycles_0004-0006" / "B_cycles.png").is_file()


def test_save_full_export_with_fits_writes_modelling_figures_and_tsvs(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    per_cycle = fit_langmuir_per_cycle(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": _fit_dict(result)}}
    fits_per_cycle = {"A": {"langmuir_mono": [_fit_dict(r) for r in per_cycle]}}

    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, fits_per_cycle, cycle_batch_size=3)

    assert (out / "figures" / "4_monomodal" / "A_langmuir_mono.png").is_file()
    assert (out / "figures" / "4_monomodal" / "A_langmuir_mono_k_drift.png").is_file()


def test_save_full_export_accepts_dataclass_fits_directly(tmp_path) -> None:
    # fits/fits_per_cycle should also work with the native dataclasses, not
    # just JSON-safe dicts — the SimpleNamespace adapter in save_full_export
    # is only needed for the dict case.
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": result}}

    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, cycle_batch_size=3)
    assert (out / "figures" / "4_monomodal" / "A_langmuir_mono.png").is_file()


_OLD_ZIP_FIGURES = [
    "1_trace/full_trace.png",
    "1_trace/derivative_analysis.png",
    "1_trace/detailed_cycles_first.png",
    "1_trace/detailed_cycles_middle.png",
    "1_trace/detailed_cycles_last.png",
    "2_pulse_detection/cycles_0001-0003/A_cycles.png",
    "2_pulse_detection/cycles_0001-0003/A_average.png",
    "2_pulse_detection/cycles_0001-0003/A_timing.png",
    "4_monomodal/A_langmuir_mono.png",
    "4_monomodal/A_langmuir_mono_k_drift.png",
    "4_monomodal/A_langmuir_mono_theta_max_drift.png",
]


def _export(tmp_path, as_dicts: bool = False, **kwargs):
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
                     cycle_batch_size=3, **kwargs)
    return out, result, pd


def _csv(pd, path):
    return pd.read_csv(path, encoding="utf-8-sig")


@pytest.mark.parametrize("as_dicts", [False, True])
def test_export_keeps_old_figures_and_writes_wide_traces(tmp_path, as_dicts) -> None:
    out, result, pd = _export(tmp_path, as_dicts)
    for name in _OLD_ZIP_FIGURES:
        assert (out / "figures" / name).is_file(), name
    assert (out / "figures" / "4_monomodal" / "A_langmuir_mono_fit_loss.png").is_file()

    traces = out / "data" / "traces"
    cols = ["time_s", "1", "2", "3", "4", "5", "6"]
    mass = _csv(pd, traces / "A_mass.csv")
    assert list(mass.columns) == cols
    assert mass["time_s"].iloc[0] == 0

    fit = _csv(pd, traces / "A_fit_mono.csv")
    loss = _csv(pd, traces / "A_fit_loss_mono.csv")
    assert list(loss.columns) == list(fit.columns) == cols
    # fit minus data: the opposite sign of the library's data - fit
    first = result.residuals.traces[0]
    n = len(first.time)
    np.testing.assert_allclose(loss["1"].iloc[:n], -first.residual, atol=1e-4)
    np.testing.assert_allclose(fit["1"].iloc[:n], first.fitted, atol=1e-4)
    avg = _csv(pd, traces / "A_fit_loss_average.csv")
    assert list(avg.columns) == [
        "time_s", "mono_mean_fit_minus_data_ng_cm2", "mono_sd_ng_cm2",
    ]


def test_export_full_trace_is_full_resolution(tmp_path) -> None:
    out, _, pd = _export(tmp_path)
    (tmp_path / "again").mkdir()
    data, _, _, _ = _built_pipeline(tmp_path / "again", n_cycles=6)
    raw = _csv(pd, out / "data" / "raw" / "full_trace.csv")
    assert list(raw.columns)[:2] == ["time_s", "mass_ng_cm2"]
    assert len(raw) == len(data.time)
    np.testing.assert_allclose(raw["time_s"], data.time, atol=1e-6)


def test_fit_summary_has_one_row_per_cycle_with_every_scalar(tmp_path) -> None:
    out, _, pd = _export(tmp_path)
    s = _csv(pd, out / "data" / "fit_summary.csv")
    assert s.columns[0] == "cycle" and list(s["cycle"]) == [1, 2, 3, 4, 5, 6]
    for col in ("A_mono_k_per_s", "A_mono_theta_max_ng_cm2", "A_mono_r2",
                "A_mono_fit_loss_rmse_ng_cm2", "A_pulse_onset_time_s",
                "A_pulse_confidence", "A_excluded", "A_mass_change_ng_cm2",
                "mass_gain_ng_cm2", "cumulative_growth_ng_cm2"):
        assert col in s.columns, col
    # the synthetic run adds 3 ng/cm² a cycle (A + B, 1.5 each)
    np.testing.assert_allclose(s["mass_gain_ng_cm2"].iloc[:5], 3.0, atol=0.2)
    np.testing.assert_allclose(
        s["cumulative_growth_ng_cm2"], s["mass_gain_ng_cm2"].cumsum(), atol=1e-6
    )
    assert s["A_mono_k_per_s"].notna().all()
    assert (s["A_excluded"] == 0).all()


def test_mass_gain_figures_and_cycle_trace(tmp_path) -> None:
    out, _, pd = _export(tmp_path)
    for name in ("mass_change_per_cycle", "cumulative_growth"):
        assert (out / "figures" / "3_mass_gain" / f"{name}.png").is_file()
    t = _csv(pd, out / "data" / "traces" / "cycle_mass_change.csv")
    assert list(t.columns) == ["time_s", "1", "2", "3", "4", "5", "6"]


def test_batch_figures_can_be_switched_off(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    out = tmp_path / "off"
    save_full_export(out, data, index, cycles, params, cycle_batches=False)
    assert not (out / "figures" / "2_pulse_detection").exists()


def test_xlsx_has_readable_headers_and_a_units_row(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    out, _, _ = _export(tmp_path, xlsx=True)
    ws = openpyxl.load_workbook(out / "data" / "fit_summary.xlsx").active
    heads = [c.value for c in ws[1]]
    units = [c.value for c in ws[2]]
    assert heads[0] == "cycle"
    assert units[heads.index("A mono k")] == "1/s"
    assert ws.cell(row=3, column=1).value == 1
    assert ws.max_row == 8  # headers, units, six cycles


def test_wide_table_orders_dotted_labels_and_pads_short_cycles() -> None:
    from qcm_pak.export_traces import wide_table

    t = np.array([0.0, 1.0, 2.0])
    df = wide_table([("7.2", t, t * 2), ("7.1", t, t)])
    assert list(df.columns) == ["time_s", "7.1", "7.2"]
    short = wide_table([("1", t, t), ("2", t[:2], t[:2])])
    assert np.isnan(short["2"].iloc[2])
