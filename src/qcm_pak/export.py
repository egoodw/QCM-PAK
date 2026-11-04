"""
Full analysis-output export: figures, diagnostic plots, TSVs, and a text
report — everything a researcher would want to archive or hand off,
beyond the lean default written by :meth:`~qcm_pak._types.AnalysisResult.save`.

Split out from ``_types.py`` so ``AnalysisResult`` stays a plain data
container; this module is where the I/O and formatting logic lives.

Public API:

- :func:`save_full_export` — writes the complete bundle described above.
- :func:`build_analysis_report` — the text report, as a string (also
  written to disk by ``save_full_export``, but useful standalone).
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from qcm_pak._types import CycleCollection, CycleIndex, MassDataset
from qcm_pak.models import MODEL_REGISTRY

if TYPE_CHECKING:
    from qcm_pak.parameters import ALDParameters

# Which fit-result field(s) are worth a per-cycle drift plot, per model id.
_DRIFT_PARAMS: dict[str, list[str]] = {
    "langmuir_mono": ["k", "theta_max"],
    "langmuir_bi": ["k1", "k2"],
    "etch_saturating": ["k", "etch_max"],
    "etch_linear": ["rate"],
}


def build_analysis_report(
    mass_data: MassDataset,
    cycle_index: CycleIndex,
    cycles: CycleCollection,
    params: ALDParameters,
    fits: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> str:
    """Plain-text summary of a completed analysis.

    Parameters
    ----------
    mass_data, cycle_index, cycles, params:
        The analysis pipeline's intermediate results.
    fits:
        ``{step_name: {model_id: fit_result_dict}}`` — the ensemble fit for
        each step/model, if modelling has been run yet.

    Returns
    -------
    str
    """
    recipe = cycle_index.recipe
    lines = [
        "QCM-PAK Analysis Report",
        "=" * 40,
        f"Input file: {params.input_file}",
        f"Duration: {mass_data.duration:.1f} s over {mass_data.n_points} samples",
        f"Median sample interval: {mass_data.dt:.4g} s",
        "",
        "Recipe:",
        f"  Steps: {', '.join(recipe.step_names())}",
        f"  Outer repeats: {recipe.repeats}",
        f"  Start time: {recipe.start_time:.1f} s",
        f"  Total expected events: {recipe.total_events}",
        "",
        "Detection:",
        f"  Detected onsets: {cycle_index.n_detected}",
        f"  Manually excluded: {sum(1 for x in cycle_index.excluded if x)}",
    ]
    if cycle_index.confidence:
        mean_conf = statistics.mean(cycle_index.confidence)
        lines.append(f"  Mean confidence: {mean_conf:.2f}")
        low_conf = sum(1 for c in cycle_index.confidence if c < 0.4)
        lines.append(f"  Low-confidence onsets (<0.40): {low_conf}")
    lines.append("")

    lines.append("Extracted cycles:")
    for step_name in recipe.step_names():
        occurrences = cycles.steps(name=step_name, include_excluded=True)
        n = len(occurrences)
        n_excl = sum(1 for s in occurrences if s.excluded)
        lines.append(f"  {step_name}: {n} occurrences ({n_excl} excluded)")
    lines.append("")

    lines.append("Fits:" if fits else "Fits: (none run yet)")
    for step_name, step_fits in (fits or {}).items():
        for model_id, fit in step_fits.items():
            r2 = (
                fit.get("r_squared") if isinstance(fit, Mapping)
                else getattr(fit, "r_squared", None)
            )
            lines.append(
                f"  {step_name} · {model_id}: R²={r2:.4f}" if r2 is not None
                else f"  {step_name} · {model_id}: (fit failed)"
            )

    return "\n".join(lines)


def _model_category(model_id: str) -> str | None:
    spec = MODEL_REGISTRY.get(model_id)
    return spec.category if spec else None


_CSV = {"index": False, "float_format": "%.8g", "encoding": "utf-8-sig"}
_MODEL_FOLDERS = {"langmuir_mono": "4_monomodal", "langmuir_bi": "5_bimodal"}
_OTHER_MODELS = "6_other_models"
_BATCH_LIMIT = 200  # longer runs skip the per-batch cycle figures unless asked

README_TEXT = """QCM-PAK analysis export

  README.txt             this file
  analysis_report.txt    plain-text report of the analysis
  diagnostics/           analysis.log
  figures/               graphs, in numbered folders:
      1_trace/             full trace, derivative, first/middle/last cycles
      2_pulse_detection/   cycle overlay, average and pulse timing, in
                           cycles_NNNN-NNNN/ batches (left out for long runs)
      3_mass_gain/         mass change per cycle, growth over the run
      4_monomodal/         single-rate Langmuir fits, drift and fit loss
      5_bimodal/           two-rate Langmuir fits, drift and fit loss
      6_other_models/      any other model
  data/
      raw/full_trace.csv     the whole run at full resolution
      traces/                one file per step and quantity (wide, see below)
      fit_summary.csv        one row per cycle, every single-number result
      fit_summary.xlsx       the same with readable headers (if asked for)

Trace files: column 1 is time_s, zero at each pulse onset; every other column
is one cycle, headed by its number counting from 1 (7.2 is the second run of
cycle 7 when a recipe repeats a step inside a cycle). Rows are the value at
that time. Empty cells mean the cycle has no sample there.
  <step>_mass.csv             mass after the pulse, baseline subtracted, ng/cm2
  <step>_fit_<model>.csv      the fitted curve, ng/cm2
  <step>_fit_loss_<model>.csv fit minus data, ng/cm2 (above zero = the fit sits
                              above the measurement)
  <step>_fit_loss_average.csv mean and SD of fit minus data over the cycles
  cycle_mass_change.csv       mass change within each cycle against time in it
<model> is mono (one rate) or bi (two rates).

fit_summary.csv: first column cycle. Units are in the column names
(_ng_cm2, _per_s, _s); r2 is R-squared; fit_loss_rmse is the root-mean-square
of fit minus data; excluded and cycle_partial are 1 or 0.
"""


def save_full_export(
    output_dir: str | Path,
    mass_data: MassDataset,
    cycle_index: CycleIndex,
    cycles: CycleCollection,
    params: ALDParameters,
    fits: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    fits_per_cycle: Mapping[str, Mapping[str, list[Mapping[str, Any]]]] | None = None,
    cycle_batch_size: int = 100,
    cycle_batches: bool | None = None,
    xlsx: bool = False,
) -> None:
    """Write the full analysis-output bundle to ``output_dir``.

    Layout (see ``README.txt`` in the output for the file by file guide):
    ``analysis_report.txt``; ``figures/`` in numbered folders
    (``1_trace`` … ``5_bimodal``, ``6_other_models``); ``data/raw/full_trace.csv``;
    ``data/traces/`` with wide per-step files for mass, fits and fit loss
    (fit minus data, ng/cm²); ``data/fit_summary.csv`` with one row per cycle
    and every single-number result, and optionally ``data/fit_summary.xlsx``.
    Fit-loss files need fits that carry ``residuals``.

    ``fits``/``fits_per_cycle`` accept either the native
    ``LangmuirResult``/``EtchResult`` dataclasses or plain JSON-safe dicts
    with the same field names (e.g. from a web API response) — anything
    with attribute- or mapping-style access to ``model``, ``k``,
    ``theta_max``, etc.

    Parameters
    ----------
    output_dir:
        Directory to write outputs into. Created if it does not exist.
    mass_data, cycle_index, cycles, params:
        The analysis pipeline's intermediate results.
    fits:
        ``{step_name: {model_id: fit_result}}`` — ensemble fits.
    fits_per_cycle:
        ``{step_name: {model_id: [fit_result, ...]}}`` — one fit per
        occurrence, for drift/trend plots and fit_summary columns.
    cycle_batch_size:
        How many outer cycles per batch of figures (keeps a long run from
        producing one increasingly-illegible overlay of everything).
    cycle_batches:
        Write the batched cycle figures. ``None`` (default) writes them for
        runs of up to 200 cycles only.
    xlsx:
        Also write ``data/fit_summary.xlsx`` (needs ``openpyxl``).
    """
    from qcm_pak import export_traces as traces
    from qcm_pak import visualization as viz

    fits = fits or {}
    fits_per_cycle = fits_per_cycle or {}
    recipe = cycle_index.recipe
    step_names = recipe.step_names()

    out = Path(output_dir)
    fig_dir = out / "figures"
    data_dir = out / "data"
    trace_dir = data_dir / "traces"
    for d in (fig_dir, data_dir / "raw", trace_dir):
        d.mkdir(parents=True, exist_ok=True)

    def save_fig(fig: Any, folder: str, name: str) -> None:  # noqa: ANN401
        target = fig_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / name, dpi=150, bbox_inches="tight")

    # ── Report and README ───────────────────────────────────────────────
    (out / "analysis_report.txt").write_text(
        build_analysis_report(mass_data, cycle_index, cycles, params, fits)
    )
    (out / "README.txt").write_text(README_TEXT)

    # ── 1_trace: full trace, derivative, first/middle/last cycles ───────
    save_fig(viz.plot_trace(mass_data, cycle_index), "1_trace", "full_trace.png")
    save_fig(viz.plot_derivative(mass_data), "1_trace", "derivative_analysis.png")
    for location in ("first", "middle", "last"):
        fig = viz.plot_detailed_cycles(
            mass_data, cycle_index, location=location, n_cycles=10
        )
        save_fig(fig, "1_trace", f"detailed_cycles_{location}.png")

    # ── 2_pulse_detection: batched overlay / average / timing ────────────
    # The visualization functions don't register figures with pyplot, so each
    # is freed as soon as ``fig`` moves on to the next; nothing to close.
    total_cycles = recipe.repeats
    if cycle_batches is None:
        cycle_batches = total_cycles <= _BATCH_LIMIT
    if cycle_batches:
        size = max(1, cycle_batch_size)
        for lo in range(0, max(1, total_cycles), size):
            hi = min(total_cycles, lo + size)
            batch = (lo, hi)
            folder = f"2_pulse_detection/cycles_{lo + 1:04d}-{hi:04d}"
            for step_name in step_names:
                fig = viz.plot_cycles_batch(cycles, step_name, cycle_range=batch)
                save_fig(fig, folder, f"{step_name}_cycles.png")
                fig = viz.plot_cycle_average(cycles, step_name, cycle_range=batch)
                save_fig(fig, folder, f"{step_name}_average.png")
                fig = viz.plot_pulse_timing(cycle_index, mass_data, step=step_name)
                save_fig(fig, folder, f"{step_name}_timing.png")

    # ── 4_monomodal / 5_bimodal: fit plots, drift plots, fit loss ────────
    for step_name, step_fits in fits.items():
        for model_id, fit in step_fits.items():
            folder = _MODEL_FOLDERS.get(model_id, _OTHER_MODELS)
            category = _model_category(model_id)
            ns = SimpleNamespace(**fit) if isinstance(fit, Mapping) else fit
            try:
                if category == "growth":
                    fig = viz.plot_langmuir(cycles, ns)
                elif category == "etch":
                    fig = viz.plot_etch(cycles, ns)
                else:
                    fig = None
                if fig is not None:
                    save_fig(fig, folder, f"{step_name}_{model_id}.png")
            except Exception:
                # a single failed/degenerate fit shouldn't abort the whole export
                pass
            records = fits_per_cycle.get(step_name, {}).get(model_id)
            if records:
                for param in _DRIFT_PARAMS.get(model_id, []):
                    fig = viz.plot_fit_drift(records, step_name, param=param)
                    save_fig(fig, folder, f"{step_name}_{model_id}_{param}_drift.png")
    for step_name, model_id, res in traces.iter_fit_loss(fits):
        try:
            fig = traces.plot_fit_loss(res, step_name, model_id)
            folder = _MODEL_FOLDERS.get(model_id, _OTHER_MODELS)
            save_fig(fig, folder, f"{step_name}_{model_id}_fit_loss.png")
        except Exception:
            pass  # one degenerate trace set shouldn't abort the export

    # ── data/raw ───────────────────────────────────────────────────────────
    traces.full_trace_table(mass_data).to_csv(
        data_dir / "raw" / "full_trace.csv",
        **{**_CSV, "float_format": "%.10g"},
    )

    # ── data/traces: wide, one file per step and quantity ──────────────────
    labels = traces.occurrence_labels(cycles, step_names)
    for step_name in step_names:
        lab = labels.get(step_name, {})
        mass = traces.mass_traces(cycles, step_name, lab)
        if len(mass.columns) > 1:
            mass.to_csv(trace_dir / f"{step_name}_mass.csv", **_CSV)
        step_fits = fits.get(step_name, {})
        for model_id, fit in step_fits.items():
            res = traces.fit_residuals(fit)
            if res is None:
                continue
            short = traces.short_model(model_id)
            fitted, loss = traces.fit_traces(res, lab)
            if len(fitted.columns) > 1:
                fitted.to_csv(trace_dir / f"{step_name}_fit_{short}.csv", **_CSV)
                loss.to_csv(trace_dir / f"{step_name}_fit_loss_{short}.csv", **_CSV)
        average = traces.fit_loss_average_table(step_fits)
        if len(average.columns) > 1:
            average.to_csv(trace_dir / f"{step_name}_fit_loss_average.csv", **_CSV)

    # ── 3_mass_gain: per-cycle mass change and cumulative growth ───────────
    gain = traces.mass_gain_table(mass_data, cycle_index, cycles)
    if not gain.empty:
        traces.cycle_mass_change_table(gain).to_csv(
            trace_dir / "cycle_mass_change.csv", **_CSV
        )
        save_fig(
            traces.plot_mass_change_per_cycle(gain),
            "3_mass_gain", "mass_change_per_cycle.png",
        )
        save_fig(
            traces.plot_cumulative_growth(gain),
            "3_mass_gain", "cumulative_growth.png",
        )

    # ── data/fit_summary: one row per cycle ────────────────────────────────
    summary = traces.fit_summary_table(
        mass_data, cycle_index, cycles, fits, fits_per_cycle, labels, gain
    )
    summary.to_csv(
        data_dir / "fit_summary.csv", **{**_CSV, "index": True},
    )
    if xlsx:
        traces.write_fit_summary_xlsx(summary, data_dir / "fit_summary.xlsx")
