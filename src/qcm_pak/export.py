"""
Full analysis-output export: figures, diagnostic plots, CSV tables and a text
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

import pandas as pd

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


def _model_folder(model_id: str) -> str:
    """Figure folder of a model, numbered like the web app's bundle."""
    return {"langmuir_mono": "4_monomodal", "langmuir_bi": "5_bimodal"}.get(
        model_id, "6_etch"
    )


def _csv(table: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=index)


README = """QCM-PAK export
==============

Figures are in figures/ (numbered folders), numbers in data/. Masses are in
ng/cm2, times in seconds.

data/raw/full_trace.csv          the whole measured trace at full resolution
data/traces/<step>_mass.csv      mass of each cycle after a pulse of <step>;
                                 time_s is zero at the pulse onset
data/traces/<step>_fit_<m>.csv   the fitted curve, same layout (m = mono, bi)
data/traces/<step>_fit_loss_<m>.csv
                                 fit minus data, same layout, plus mean, sd
                                 and n of the cycles at each time
data/traces/cycle_mass_change.csv
                                 mass change since the cycle started
data/fit_summary.csv             one row per cycle: fit parameters, R2, fit
                                 loss RMSE, mass gain, cumulative growth,
                                 pulse onset time and confidence

In the trace files each column after time_s is one cycle, named by its
number (7, or 7.1 and 7.2 where a step runs twice in a cycle). Columns end
earlier when a cycle has less data. A fit curve is the ensemble fit of all
cycles evaluated on that cycle's time axis; per-cycle parameters are in
fit_summary.csv. The fit loss RMSE is that ensemble fit against the cycle.
Units are part of the column names in fit_summary.csv (_per_s, _ng_cm2).
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
    extra_cycle_columns: Mapping[str, Mapping[int, float]] | None = None,
    include_original: bool = False,
    fit_summary_xlsx: bool = False,
) -> None:
    """Write the full analysis-output bundle to ``output_dir``.

    Layout (see ``README.txt`` in the output)::

        README.txt  analysis_report.txt  diagnostics/ (written by the caller)
        figures/1_trace  2_pulse_detection/cycles_NNNN-NNNN  3_mass_gain
                 4_monomodal  5_bimodal  6_etch
        data/raw/full_trace.csv
        data/traces/<step>_mass.csv, <step>_fit_<model>.csv,
                    <step>_fit_loss_<model>.csv, cycle_mass_change.csv
        data/fit_summary.csv

    Traces are wide CSVs (time, then one column per cycle); fit loss is fit
    minus data. Fit files need fits that carry ``residuals``.

    ``fits``/``fits_per_cycle`` accept either the native
    ``LangmuirResult``/``EtchResult`` dataclasses or plain JSON-safe dicts
    with the same field names (e.g. from a web API response).

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
        occurrence, for drift plots and ``fit_summary.csv``.
    cycle_batch_size:
        How many outer cycles per batch of figures (keeps a long run from
        producing one increasingly-illegible overlay of everything).
    extra_cycle_columns:
        ``{column_name: {cycle (1-based): value}}`` added to
        ``fit_summary.csv`` (the name should carry its unit).
    include_original:
        Copy ``params.input_file`` to ``data/raw/``.
    fit_summary_xlsx:
        Also write ``data/fit_summary.xlsx`` (needs ``openpyxl``; skipped
        silently when it is not installed).
    """
    import shutil

    from qcm_pak import export_tables as tables
    from qcm_pak import export_traces as traces
    from qcm_pak import visualization as viz

    fits = fits or {}
    fits_per_cycle = fits_per_cycle or {}
    recipe = cycle_index.recipe

    out = Path(output_dir)
    fig_dir = out / "figures"
    data_dir = out / "data"

    def save(fig: Any, *parts: str) -> None:  # noqa: ANN401
        path = fig_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Text ────────────────────────────────────────────────────────────
    (out / "analysis_report.txt").write_text(
        build_analysis_report(mass_data, cycle_index, cycles, params, fits)
    )
    (out / "README.txt").write_text(README)

    # ── 1_trace: full trace, derivative, detailed first/middle/last ─────
    save(viz.plot_trace(mass_data, cycle_index), "1_trace", "full_trace.png")
    save(viz.plot_derivative(mass_data), "1_trace", "derivative_analysis.png")
    for location in ("first", "middle", "last"):
        fig = viz.plot_detailed_cycles(
            mass_data, cycle_index, location=location, n_cycles=10
        )
        save(fig, "1_trace", f"detailed_cycles_{location}.png")

    # ── 2_pulse_detection: cycle-batched overlay / average / timing ─────
    # Figures accumulate fast here (steps x batches x 3 plot types). The
    # visualization functions don't register them with pyplot, so each one
    # is freed as soon as ``fig`` moves on to the next; nothing to close.
    total_cycles = recipe.repeats
    step_names = recipe.step_names()
    for batch_lo in range(0, max(1, total_cycles), max(1, cycle_batch_size)):
        batch_hi = min(total_cycles, batch_lo + cycle_batch_size)
        batch = (batch_lo, batch_hi)
        folder = f"cycles_{batch_lo + 1:04d}-{batch_hi:04d}"
        for step_name in step_names:
            figs = {
                "cycles": viz.plot_cycles_batch(cycles, step_name, cycle_range=batch),
                "average": viz.plot_cycle_average(
                    cycles, step_name, cycle_range=batch
                ),
                "timing": viz.plot_pulse_timing(
                    cycle_index, mass_data, step=step_name
                ),
            }
            for kind, fig in figs.items():
                save(fig, "2_pulse_detection", folder, f"{step_name}_{kind}.png")

    # ── Model figures: fit plots, drift plots ───────────────────────────
    for step_name, step_fits in fits.items():
        for model_id, fit in step_fits.items():
            category = _model_category(model_id)
            folder = _model_folder(model_id)
            suffix = f"_{model_id}" if folder == "6_etch" else ""
            ns = SimpleNamespace(**fit) if isinstance(fit, Mapping) else fit
            try:
                if category == "growth":
                    fig = viz.plot_langmuir(cycles, ns)
                    name = f"langmuir_fit_{step_name}"
                elif category == "etch":
                    fig = viz.plot_etch(cycles, ns)
                    name = f"etch_fit_{step_name}{suffix}"
                else:
                    continue
                save(fig, folder, f"{name}.png")
            except Exception:
                # a single failed/degenerate fit shouldn't abort the whole export
                pass

            records = fits_per_cycle.get(step_name, {}).get(model_id)
            if records:
                for param in _DRIFT_PARAMS.get(model_id, []):
                    fig = viz.plot_fit_drift(records, step_name, param=param)
                    save(fig, folder, f"{step_name}{suffix}_{param}_drift.png")

    # ── data/raw ────────────────────────────────────────────────────────
    _csv(tables.raw_trace_table(mass_data), data_dir / "raw" / "full_trace.csv")
    original = Path(params.input_file)
    if include_original and original.is_file():
        shutil.copy2(original, data_dir / "raw" / original.name)

    # ── data/traces: wide CSVs, one file per step and quantity ──────────
    trace_dir = data_dir / "traces"
    for step_name in step_names:
        if cycles.steps(name=step_name):
            _csv(tables.step_mass_table(cycles, step_name),
                 trace_dir / f"{step_name}_mass.csv")
    for step_name, model_id, res in tables.iter_fit_loss(fits):
        short = tables.model_short(model_id)
        fit_t, loss_t = tables.fit_trace_tables(res)
        _csv(fit_t, trace_dir / f"{step_name}_fit_{short}.csv")
        _csv(loss_t, trace_dir / f"{step_name}_fit_loss_{short}.csv")
        folder = _model_folder(model_id)
        suffix = f"_{model_id}" if folder == "6_etch" else ""
        try:
            save(traces.plot_fit_loss(res, step_name, model_id),
                 folder, f"fit_loss_{step_name}{suffix}.png")
        except Exception:
            pass  # one degenerate trace set shouldn't abort the export

    # ── mass gain: figures and the cycle trace ──────────────────────────
    gain = tables.mass_gain_table(mass_data, cycle_index, cycles)
    if not gain.empty:
        _csv(tables.cycle_mass_change_table(gain),
             trace_dir / "cycle_mass_change.csv")
        save(traces.plot_mass_change_per_cycle(gain),
             "3_mass_gain", "mass_change_per_cycle.png")
        save(traces.plot_cumulative_growth(gain),
             "3_mass_gain", "cumulative_growth.png")

    # ── data/fit_summary ────────────────────────────────────────────────
    summary = tables.fit_summary_table(
        mass_data, cycle_index, cycles, fits, fits_per_cycle, gain,
        extra_cycle_columns,
    )
    _csv(summary, data_dir / "fit_summary.csv", index=True)
    if fit_summary_xlsx:
        tables.write_fit_summary_xlsx(
            data_dir / "fit_summary.csv", data_dir / "fit_summary.xlsx"
        )
