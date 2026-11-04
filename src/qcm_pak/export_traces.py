"""Fit-loss and mass-gain traces for the full analysis export.

Everything here turns data the analysis already holds into tables and
figures; nothing is refitted. :func:`~qcm_pak.export.save_full_export` calls
these helpers.

Sign convention: the library stores residuals as ``data - fit``
(:class:`~qcm_pak.residuals.ResidualTrace`); the exported files use the web
app's ``fit - data`` so above zero means the fit sits above the measurement.
All masses are ng/cm².
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from qcm_pak._types import CycleCollection, CycleIndex, MassDataset

FIT_LOSS_COLUMNS = [
    "outer_cycle",
    "sub_cycle_run",
    "time_s",
    "data_ng_cm2",
    "fitted_ng_cm2",
    "residual_fit_minus_data_ng_cm2",
]
FIT_LOSS_AVERAGE_COLUMNS = [
    "step",
    "model",
    "time_s",
    "mean_fit_minus_data_ng_cm2",
    "sd_ng_cm2",
    "n_traces",
]
MASS_GAIN_COLUMNS = [
    "outer_cycle",
    "time_s",
    "time_in_cycle_s",
    "mass_ng_cm2",
    "mass_change_in_cycle_ng_cm2",
    "cumulative_growth_ng_cm2",
    "partial",
]


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 (dataclass or dict)
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _residuals_of(fit: Any) -> Any:  # noqa: ANN401
    return _get(fit, "residuals")


def _f(values: Any) -> np.ndarray:  # noqa: ANN401
    """List or array to floats; ``None`` (a stored NaN) becomes NaN."""
    return np.array(
        [np.nan if v is None else v for v in values], dtype=float
    )


def iter_fit_loss(
    fits: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[str, str, Any]]:
    """Yield ``(step_name, model_id, residuals)`` for every fit that has traces."""
    for step_name, step_fits in fits.items():
        for model_id, fit in step_fits.items():
            res = _residuals_of(fit)
            if res is not None and _get(res, "traces"):
                yield step_name, model_id, res


def fit_loss_table(residuals: Any) -> pd.DataFrame:  # noqa: ANN401
    """One row per sample of every residual trace (long format)."""
    frames = []
    for tr in _get(residuals, "traces") or []:
        time = _f(_get(tr, "time"))
        frames.append(
            pd.DataFrame(
                {
                    "outer_cycle": int(_get(tr, "outer_cycle")) + 1,
                    "sub_cycle_run": int(_get(tr, "sub_cycle_run")),
                    "time_s": time,
                    "data_ng_cm2": _f(_get(tr, "observed")),
                    "fitted_ng_cm2": _f(_get(tr, "fitted")),
                    # library residual is data - fit; the export is fit - data
                    "residual_fit_minus_data_ng_cm2": -_f(_get(tr, "residual")),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=FIT_LOSS_COLUMNS)
    return pd.concat(frames, ignore_index=True)[FIT_LOSS_COLUMNS]


def fit_loss_average_table(
    fits: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Mean and spread of fit minus data per step and model, from stored stats."""
    frames = []
    for step_name, model_id, res in iter_fit_loss(fits):
        stats = _get(res, "stats")
        if stats is None:
            continue
        time = _f(_get(stats, "time"))
        frames.append(
            pd.DataFrame(
                {
                    "step": step_name,
                    "model": model_id,
                    "time_s": time,
                    "mean_fit_minus_data_ng_cm2": -_f(_get(stats, "mean")),
                    "sd_ng_cm2": _f(_get(stats, "std")),
                    "n_traces": np.asarray(_get(stats, "n"), dtype=int),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=FIT_LOSS_AVERAGE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[FIT_LOSS_AVERAGE_COLUMNS]


def plot_fit_loss(residuals: Any, step_name: str, model_id: str) -> Figure:  # noqa: ANN401
    """Every cycle's fit-minus-data trace, with the mean and ±1 SD band."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    for tr in _get(residuals, "traces") or []:
        ax.plot(
            _f(_get(tr, "time")),
            -_f(_get(tr, "residual")),
            color="steelblue",
            alpha=0.25,
            lw=0.7,
        )
    stats = _get(residuals, "stats")
    if stats is not None:
        t = _f(_get(stats, "time"))
        mean = -_f(_get(stats, "mean"))
        sd = _f(_get(stats, "std"))
        ax.fill_between(t, mean - sd, mean + sd, color="black", alpha=0.12, lw=0)
        ax.plot(t, mean, color="black", lw=1.4, label="mean")
        ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Time since pulse onset (s)")
    ax.set_ylabel("Fit − data (ng/cm²)")
    ax.set_title(f"{step_name} · {model_id} — fit loss")
    fig.tight_layout()
    return fig


def _step_durations(index: CycleIndex) -> dict[str, float]:
    out: dict[str, float] = {}
    for sc in index.recipe.sub_cycles:
        for step in sc.steps:
            out[step.name] = max(out.get(step.name, 0.0), step.pulse + step.purge)
    return out


def mass_gain_table(
    mass_data: MassDataset,
    index: CycleIndex,
    cycles: CycleCollection,
) -> pd.DataFrame:
    """Mass over time for each outer cycle, and the growth built up so far.

    A cycle runs from its first pulse onset (the first cycle from the recipe
    start) to the next cycle's first onset; the last one ends one step length
    after its last onset, or at the end of the data. ``partial`` is ``yes``
    where the data ran out first. Empty when the onsets cannot be matched to
    outer cycles (for example after a failed detection).
    """
    time = np.asarray(mass_data.time, dtype=float)
    mass = np.asarray(mass_data.mass, dtype=float)
    recipe = index.recipe
    first_onset: dict[int, float] = {}
    last_end: dict[int, float] = {}
    durations = _step_durations(index)
    by_name: dict[str, list[float]] = {}
    for name, idx in index.step_onsets:
        by_name.setdefault(name, []).append(float(time[idx]))
    for name, onsets in by_name.items():
        results = cycles.steps(name=name, include_excluded=True)
        if len(results) != len(onsets):
            return pd.DataFrame(columns=MASS_GAIN_COLUMNS)
        for sr, t in zip(results, onsets):
            k = sr.outer_cycle
            first_onset[k] = min(t, first_onset.get(k, t))
            end = t + durations.get(name, 0.0)
            last_end[k] = max(end, last_end.get(k, end))
    if not first_onset or time.size < 2:
        return pd.DataFrame(columns=MASS_GAIN_COLUMNS)

    order = sorted(first_onset)
    starts = {k: first_onset[k] for k in order}
    starts[order[0]] = float(recipe.start_time or time[0])
    t_last = float(time[-1])
    m0 = float(np.interp(starts[order[0]], time, mass))
    frames = []
    for pos, k in enumerate(order):
        t0 = starts[k]
        if pos + 1 < len(order):
            t1, partial = starts[order[pos + 1]], False
        else:
            t1, partial = min(last_end[k], t_last), last_end[k] > t_last
        sel = (time >= t0) & (time <= t1)
        if not sel.any():
            continue
        tt, mm = time[sel], mass[sel]
        m_start = float(np.interp(t0, time, mass))
        frames.append(
            pd.DataFrame(
                {
                    "outer_cycle": k + 1,
                    "time_s": tt,
                    "time_in_cycle_s": tt - t0,
                    "mass_ng_cm2": mm,
                    "mass_change_in_cycle_ng_cm2": mm - m_start,
                    "cumulative_growth_ng_cm2": mm - m0,
                    "partial": "yes" if partial else "",
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=MASS_GAIN_COLUMNS)
    return pd.concat(frames, ignore_index=True)[MASS_GAIN_COLUMNS]


def cumulative_growth_table(trace: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle: growth in the cycle and the running total."""
    cols = [
        "outer_cycle",
        "t_start_s",
        "t_end_s",
        "delta_ng_cm2",
        "cumulative_ng_cm2",
        "partial",
    ]
    if trace.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for k, g in trace.groupby("outer_cycle", sort=True):
        rows.append(
            {
                "outer_cycle": k,
                "t_start_s": g["time_s"].iloc[0],
                "t_end_s": g["time_s"].iloc[-1],
                "delta_ng_cm2": g["mass_change_in_cycle_ng_cm2"].iloc[-1],
                "cumulative_ng_cm2": g["cumulative_growth_ng_cm2"].iloc[-1],
                "partial": g["partial"].iloc[0],
            }
        )
    return pd.DataFrame(rows)[cols]


def plot_mass_change_per_cycle(trace: pd.DataFrame) -> Figure:
    """Mass change within each cycle against time in the cycle, one line each."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    cycles = sorted(trace["outer_cycle"].unique()) if not trace.empty else []
    if not cycles:
        ax.set_title("Mass change per cycle — no cycles")
    for i, k in enumerate(cycles):
        g = trace[trace["outer_cycle"] == k]
        shade = 0.85 - 0.7 * (i / max(1, len(cycles) - 1))
        ax.plot(
            g["time_in_cycle_s"],
            g["mass_change_in_cycle_ng_cm2"],
            color=str(shade),
            lw=0.8,
        )
    if cycles:
        ax.set_title("Mass change per cycle (older lighter, newer darker)")
    ax.set_xlabel("Time in cycle (s)")
    ax.set_ylabel("Mass change in cycle (ng/cm²)")
    fig.tight_layout()
    return fig


def plot_cumulative_growth(trace: pd.DataFrame) -> Figure:
    """Mass relative to the recipe start over the whole run."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(
        trace["time_s"], trace["cumulative_growth_ng_cm2"], color="steelblue", lw=1
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Growth since start (ng/cm²)")
    ax.set_title("Growth over the run")
    fig.tight_layout()
    return fig
