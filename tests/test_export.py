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

    for name in [
        "README.txt",
        "analysis_report.txt",
        "figures/1_trace/full_trace.png",
        "figures/1_trace/derivative_analysis.png",
        "figures/1_trace/detailed_cycles_first.png",
        "figures/1_trace/detailed_cycles_middle.png",
        "figures/1_trace/detailed_cycles_last.png",
        "figures/3_mass_gain/mass_change_per_cycle.png",
        "figures/3_mass_gain/cumulative_growth.png",
        "data/raw/full_trace.csv",
        "data/traces/A_mass.csv",
        "data/traces/B_mass.csv",
        "data/traces/cycle_mass_change.csv",
        "data/fit_summary.csv",
    ]:
        assert (out / name).is_file(), name


def test_save_full_export_batches_cycle_figures(tmp_path) -> None:
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, cycle_batch_size=3)

    # 6 cycles / batch_size 3 -> two batches
    batch = out / "figures" / "2_pulse_detection"
    assert (batch / "cycles_0001-0003" / "A_cycles.png").is_file()
    assert (batch / "cycles_0001-0003" / "A_average.png").is_file()
    assert (batch / "cycles_0001-0003" / "A_timing.png").is_file()
    assert (batch / "cycles_0004-0006" / "B_cycles.png").is_file()


def test_save_full_export_accepts_dataclass_fits_directly(tmp_path) -> None:
    # fits/fits_per_cycle should also work with the native dataclasses, not
    # just JSON-safe dicts — the SimpleNamespace adapter in save_full_export
    # is only needed for the dict case.
    data, params, index, cycles = _built_pipeline(tmp_path, n_cycles=6)
    result = fit_langmuir(cycles, "A", model="mono")
    fits = {"A": {"langmuir_mono": result}}

    out = tmp_path / "output"
    save_full_export(out, data, index, cycles, params, fits, cycle_batch_size=3)
    assert (out / "figures" / "4_monomodal" / "langmuir_fit_A.png").is_file()


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
    return out, result, data, pd


def _tree(out) -> list[str]:
    return sorted(
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    )


def test_export_tree_is_one_layout_without_tsv(tmp_path) -> None:
    out, _, _, _ = _export(tmp_path)
    names = _tree(out)
    assert not [n for n in names if n.endswith(".tsv")]
    top = {n.split("/")[0] for n in names}
    assert top == {"README.txt", "analysis_report.txt", "figures", "data"}
    assert {n.split("/")[1] for n in names if n.startswith("figures/")} == {
        "1_trace", "2_pulse_detection", "3_mass_gain", "4_monomodal",
    }
    for name in [
        "figures/4_monomodal/langmuir_fit_A.png",
        "figures/4_monomodal/fit_loss_A.png",
        "figures/4_monomodal/A_k_drift.png",
        "figures/4_monomodal/A_theta_max_drift.png",
        "data/traces/A_fit_mono.csv",
        "data/traces/A_fit_loss_mono.csv",
    ]:
        assert name in names, name
    assert {n for n in names if n.startswith("data/")} == {
        "data/raw/full_trace.csv",
        "data/traces/A_mass.csv", "data/traces/B_mass.csv",
        "data/traces/A_fit_mono.csv", "data/traces/A_fit_loss_mono.csv",
        "data/traces/cycle_mass_change.csv",
        "data/fit_summary.csv",
    }


def test_raw_trace_is_full_resolution(tmp_path) -> None:
    out, _, data, pd = _export(tmp_path)
    raw = pd.read_csv(out / "data" / "raw" / "full_trace.csv")
    assert list(raw.columns) == ["time_s", "mass_ng_cm2", "frequency_hz"]
    assert len(raw) == len(data.time)
    np.testing.assert_allclose(raw["mass_ng_cm2"], data.mass)


def test_mass_trace_is_wide_with_numeric_cycle_headers(tmp_path) -> None:
    out, _, _, pd = _export(tmp_path)
    mass = pd.read_csv(out / "data" / "traces" / "A_mass.csv")
    assert list(mass.columns) == ["time_s", "1", "2", "3", "4", "5", "6"]
    assert mass["time_s"].iloc[0] == 0
    # the synthetic step adds 1.5 ng/cm2 and holds it
    np.testing.assert_allclose(mass["1"].iloc[-1], 1.5, atol=0.2)


@pytest.mark.parametrize("as_dicts", [False, True])
def test_fit_and_fit_loss_traces_are_wide_and_fit_minus_data(
    tmp_path, as_dicts
) -> None:
    out, result, _, pd = _export(tmp_path, as_dicts)
    fit = pd.read_csv(out / "data" / "traces" / "A_fit_mono.csv")
    loss = pd.read_csv(out / "data" / "traces" / "A_fit_loss_mono.csv")
    cols = ["time_s", "1", "2", "3", "4", "5", "6"]
    assert list(fit.columns) == cols
    assert list(loss.columns) == cols + ["mean", "sd", "n"]
    # fit minus data: the opposite sign of the library's data - fit
    for k, tr in enumerate(result.residuals.traces, start=1):
        np.testing.assert_allclose(
            loss[str(k)].dropna().to_numpy(), -np.asarray(tr.residual), atol=1e-6
        )
        np.testing.assert_allclose(
            fit[str(k)].dropna().to_numpy(), np.asarray(tr.fitted), atol=1e-6
        )
    np.testing.assert_allclose(
        loss["mean"].dropna().to_numpy(), -np.asarray(result.residuals.stats.mean),
        atol=1e-6,
    )


def test_cycle_mass_change_trace(tmp_path) -> None:
    out, _, _, pd = _export(tmp_path)
    trace = pd.read_csv(out / "data" / "traces" / "cycle_mass_change.csv")
    assert list(trace.columns) == ["time_s", "1", "2", "3", "4", "5", "6"]
    # the synthetic run adds 3 ng/cm2 a cycle (A + B, 1.5 each)
    np.testing.assert_allclose(trace["1"].dropna().iloc[-1], 3.0, atol=0.3)


@pytest.mark.parametrize("as_dicts", [False, True])
def test_fit_summary_columns_and_rows(tmp_path, as_dicts) -> None:
    out, result, _, pd = _export(tmp_path, as_dicts)
    s = pd.read_csv(out / "data" / "fit_summary.csv", index_col="cycle")
    assert list(s.index) == [1, 2, 3, 4, 5, 6]
    for col in [
        "cycle_t_start_s", "cycle_partial", "mass_gain_ng_cm2",
        "cumulative_growth_ng_cm2",
        "A_pulse_onset_time_s", "A_pulse_confidence", "A_excluded",
        "A_mass_change_ng_cm2", "B_pulse_onset_time_s",
        "A_mono_k_per_s", "A_mono_theta_max_ng_cm2", "A_mono_r2",
        "A_mono_fit_loss_rmse_ng_cm2",
    ]:
        assert col in s.columns, col
    assert not [c for c in s.columns if c.startswith("B_mono")]
    np.testing.assert_allclose(s["mass_gain_ng_cm2"].iloc[:5], 3.0, atol=0.2)
    np.testing.assert_allclose(
        s["cumulative_growth_ng_cm2"], s["mass_gain_ng_cm2"].cumsum(), atol=1e-6
    )
    # rmse is the library's own number for each trace
    np.testing.assert_allclose(
        s["A_mono_fit_loss_rmse_ng_cm2"],
        [t.rmse for t in result.residuals.traces], atol=1e-6,
    )


def test_extra_cycle_columns_and_xlsx_flag(tmp_path) -> None:
    out, _, _, pd = _export(
        tmp_path, extra_cycle_columns={"x_raw_delta_ng_cm2": {1: 1.5, 3: 2.5}},
        fit_summary_xlsx=True,
    )
    s = pd.read_csv(out / "data" / "fit_summary.csv", index_col="cycle")
    assert s["x_raw_delta_ng_cm2"].dropna().to_dict() == {1: 1.5, 3: 2.5}
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        assert not (out / "data" / "fit_summary.xlsx").exists()
    else:
        assert (out / "data" / "fit_summary.xlsx").is_file()


def test_original_file_only_when_asked(tmp_path) -> None:
    out, _, _, _ = _export(tmp_path)
    assert not (out / "data" / "raw" / "d.csv").exists()
    out2 = tmp_path / "second"
    out2.mkdir()
    out2, _, _, _ = _export(out2, include_original=True)
    assert (out2 / "data" / "raw" / "d.csv").is_file()


def test_split_unit_names() -> None:
    from qcm_pak.export_tables import split_unit

    assert split_unit("TMA_mono_k_per_s") == ("TMA mono k", "1/s")
    assert split_unit("TMA_mono_theta_max_ng_cm2") == ("TMA mono theta max", "ng/cm²")
    assert split_unit("TMA_mono_r2") == ("TMA mono r2", "")


def test_occurrence_labels_number_repeated_runs() -> None:
    from qcm_pak.export_tables import occurrence_labels

    assert occurrence_labels([0, 1, 1, 2]) == ["1", "2.1", "2.2", "3"]
