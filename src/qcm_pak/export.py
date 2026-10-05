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

from qcm_pak._types import CycleCollection, CycleIndex, MassDataset, _write_step_tsv
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


def save_full_export(
    output_dir: str | Path,
    mass_data: MassDataset,
    cycle_index: CycleIndex,
    cycles: CycleCollection,
    params: ALDParameters,
    fits: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    fits_per_cycle: Mapping[str, Mapping[str, list[Mapping[str, Any]]]] | None = None,
    cycle_batch_size: int = 100,
) -> None:
    """Write the full analysis-output bundle to ``output_dir``.

    Beyond what :meth:`~qcm_pak._types.AnalysisResult.save` writes, this
    adds: detailed first/middle/last-cycle zoom plots, cycle-batched
    overlay/average/timing figures, per-model fit plots and per-cycle drift
    plots, pulse-time and pulse-confidence TSVs, per-cycle fit-parameter
    TSVs, and a text report.

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
        occurrence, for drift/trend plots and TSVs.
    cycle_batch_size:
        How many outer cycles per batch of figures (keeps a long run from
        producing one increasingly-illegible overlay of everything).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from qcm_pak import visualization as viz

    fits = fits or {}
    fits_per_cycle = fits_per_cycle or {}
    recipe = cycle_index.recipe

    out = Path(output_dir)
    fig_dir = out / "figures"
    data_dir = out / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Report ──────────────────────────────────────────────────────────
    (out / "analysis_report.txt").write_text(
        build_analysis_report(mass_data, cycle_index, cycles, params, fits)
    )

    # ── Full trace + derivative (unchanged from AnalysisResult.save) ────
    trace_fig = viz.plot_trace(mass_data, cycle_index)
    trace_fig.savefig(fig_dir / "full_trace.png", dpi=150, bbox_inches="tight")
    plt.close(trace_fig)

    deriv_fig = viz.plot_derivative(mass_data)
    deriv_fig.savefig(fig_dir / "derivative_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(deriv_fig)

    # ── Detailed first/middle/last cycle views ───────────────────────────
    for location in ("first", "middle", "last"):
        fig = viz.plot_detailed_cycles(
            mass_data, cycle_index, location=location, n_cycles=10
        )
        fig.savefig(
            fig_dir / f"detailed_cycles_{location}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # ── Cycle-batched overlay / average / timing figures ─────────────────
    # Figures accumulate fast here (steps x batches x 3 plot types), so each
    # one is closed immediately after saving rather than left registered in
    # pyplot's global figure manager — otherwise a long run can pile up
    # dozens of retained Figure objects before this function returns.
    total_cycles = recipe.repeats
    step_names = recipe.step_names()
    for batch_lo in range(0, max(1, total_cycles), max(1, cycle_batch_size)):
        batch_hi = min(total_cycles, batch_lo + cycle_batch_size)
        batch_dir = fig_dir / f"cycles_{batch_lo + 1:04d}-{batch_hi:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for step_name in step_names:
            batch = (batch_lo, batch_hi)
            fig = viz.plot_cycles_batch(cycles, step_name, cycle_range=batch)
            fig.savefig(
                batch_dir / f"{step_name}_cycles.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            fig = viz.plot_cycle_average(cycles, step_name, cycle_range=batch)
            fig.savefig(
                batch_dir / f"{step_name}_average.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            fig = viz.plot_pulse_timing(cycle_index, mass_data, step=step_name)
            fig.savefig(
                batch_dir / f"{step_name}_timing.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

    # ── Per-model fit plots + per-cycle drift plots ───────────────────────
    if fits:
        model_dir = fig_dir / "modelling"
        model_dir.mkdir(parents=True, exist_ok=True)
        for step_name, step_fits in fits.items():
            for model_id, fit in step_fits.items():
                category = _model_category(model_id)
                ns = SimpleNamespace(**fit) if isinstance(fit, Mapping) else fit
                try:
                    if category == "growth":
                        fig = viz.plot_langmuir(cycles, ns)
                    elif category == "etch":
                        fig = viz.plot_etch(cycles, ns)
                    else:
                        continue
                    fig.savefig(
                        model_dir / f"{step_name}_{model_id}.png",
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)
                except Exception:
                    # a single failed/degenerate fit shouldn't abort the whole export
                    pass

                records = fits_per_cycle.get(step_name, {}).get(model_id)
                if records:
                    for param in _DRIFT_PARAMS.get(model_id, []):
                        fig = viz.plot_fit_drift(records, step_name, param=param)
                        fig.savefig(
                            model_dir / f"{step_name}_{model_id}_{param}_drift.png",
                            dpi=150, bbox_inches="tight",
                        )
                        plt.close(fig)

    # ── TSVs ───────────────────────────────────────────────────────────
    for step_name in step_names:
        step_results = cycles.steps(name=step_name)
        if step_results:
            _write_step_tsv(step_results, data_dir / f"cycle_data_{step_name}.tsv")

        onset_times = [
            mass_data.time[idx]
            for name, idx in cycle_index.step_onsets if name == step_name
        ]
        all_results = cycles.steps(name=step_name, include_excluded=True)
        if onset_times and len(onset_times) == len(all_results):
            rows = [
                {
                    "outer_cycle": sr.outer_cycle,
                    "sub_cycle_run": sr.sub_cycle_run,
                    "onset_time_s": t,
                    "excluded": sr.excluded,
                }
                for sr, t in zip(all_results, onset_times)
            ]
            pd.DataFrame(rows).to_csv(
                data_dir / f"pulse_times_{step_name}.tsv", sep="\t", index=False
            )

        for model_id, records in fits_per_cycle.get(step_name, {}).items():
            if records:
                pd.DataFrame(list(records)).to_csv(
                    data_dir / f"{step_name}_{model_id}_per_cycle.tsv",
                    sep="\t",
                    index=False,
                )

    confidence = cycle_index.confidence
    excluded = cycle_index.excluded
    rows = [
        {
            "pulse_id": i,
            "step_name": name,
            "onset_time_s": mass_data.time[idx],
            "confidence": confidence[i] if i < len(confidence) else None,
            "excluded": excluded[i] if i < len(excluded) else False,
        }
        for i, (name, idx) in enumerate(cycle_index.step_onsets)
    ]
    pd.DataFrame(rows).to_csv(data_dir / "pulse_confidence.tsv", sep="\t", index=False)
