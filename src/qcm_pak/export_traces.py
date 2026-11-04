"""Fit-loss and mass-gain traces for the full analysis export.

Everything here turns data the analysis already holds into tables and
figures; nothing is refitted. Trace files are wide: column 1 time_s (zero at
each pulse onset) and one column per cycle, headed by its 1-based number.
:func:`~qcm_pak.export.save_full_export` calls these helpers.

Sign convention: the library stores residuals as ``data - fit``
(:class:`~qcm_pak.residuals.ResidualTrace`); the exported files use the web
app's ``fit - data`` so above zero means the fit sits above the measurement.
All masses are ng/cm².
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from qcm_pak._types import CycleCollection, CycleIndex, MassDataset

MASS_GAIN_COLUMNS = [
    "outer_cycle",
    "time_s",
    "time_in_cycle_s",
    "mass_ng_cm2",
    "mass_change_in_cycle_ng_cm2",
    "cumulative_growth_ng_cm2",
    "partial",
]

# Fields of a per-cycle fit record that say which occurrence it is, or that are
# not single numbers; everything else becomes a column of fit_summary.
_RECORD_SKIP = {
    "step_name", "model", "outer_cycle", "sub_cycle_index", "sub_cycle_run",
    "residuals", "covariance",
}
_UNITS = {
    "k": "per_s", "k1": "per_s", "k2": "per_s",
    "theta_max": "ng_cm2", "theta1": "ng_cm2", "theta2": "ng_cm2",
    "etch_max": "ng_cm2",
    "r_squared": "r2",
}
# column-name suffix -> unit shown under the header in fit_summary.xlsx
_XLSX_UNITS = [("_ng_cm2", "ng/cm²"), ("_per_s", "1/s"), ("_s", "s")]

OccKey = tuple[int, int, int]  # outer_cycle, sub_cycle_index, sub_cycle_run


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 (dataclass or dict)
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _residuals_of(fit: Any) -> Any:  # noqa: ANN401
    return _get(fit, "residuals")


def _f(values: Any) -> np.ndarray:  # noqa: ANN401
    """List or array to floats; ``None`` (a stored NaN) becomes NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def short_model(model_id: str) -> str:
    """``langmuir_mono`` -> ``mono``, ``langmuir_bi`` -> ``bi``; others as they are."""
    prefix = "langmuir_"
    return model_id[len(prefix):] if model_id.startswith(prefix) else model_id


def fit_residuals(fit: Any) -> Any:  # noqa: ANN401
    """The residuals of one fit if it carries traces, else None."""
    res = _residuals_of(fit)
    return res if res is not None and _get(res, "traces") else None


def iter_fit_loss(
    fits: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[str, str, Any]]:
    """Yield ``(step_name, model_id, residuals)`` for every fit that has traces."""
    for step_name, step_fits in fits.items():
        for model_id, fit in step_fits.items():
            res = _residuals_of(fit)
            if res is not None and _get(res, "traces"):
                yield step_name, model_id, res


def _key(obj: Any) -> OccKey | None:  # noqa: ANN401
    parts = [_get(obj, n) for n in ("outer_cycle", "sub_cycle_index", "sub_cycle_run")]
    if parts[0] is None:
        return None
    return int(parts[0]), int(parts[1] or 0), int(parts[2] or 0)


def occurrence_labels(
    cycles: CycleCollection, step_names: list[str]
) -> dict[str, dict[OccKey, str]]:
    """Column header (and fit_summary row label) of every occurrence of each step.

    Plain 1-based cycle numbers (``7``). When any step is repeated inside one
    outer cycle (sub-cycles), every label gets the occurrence number after a
    dot (``7.1``, ``7.2``) so the labels stay unique and consistent across files.
    """
    keys: dict[str, dict[int, list[OccKey]]] = {}
    for name in step_names:
        per_outer: dict[int, list[OccKey]] = {}
        for sr in cycles.steps(name=name, include_excluded=True):
            k = _key(sr)
            if k is not None and k not in per_outer.setdefault(k[0], []):
                per_outer[k[0]].append(k)
        for lst in per_outer.values():
            lst.sort()
        keys[name] = per_outer
    repeated = any(len(lst) > 1 for per in keys.values() for lst in per.values())
    labels: dict[str, dict[OccKey, str]] = {}
    for name, per_outer in keys.items():
        labels[name] = {}
        for outer, lst in per_outer.items():
            for n, k in enumerate(lst, start=1):
                labels[name][k] = f"{outer + 1}.{n}" if repeated else str(outer + 1)
    return labels


def _label_order(label: str) -> tuple[int, int]:
    head, _, tail = label.partition(".")
    return int(head), int(tail or 0)


def wide_table(series: list[tuple[str, Any, Any]]) -> pd.DataFrame:
    """``time_s`` plus one column per ``(label, time, values)`` series.

    Every series is read onto the time grid of the first one; where a cycle has
    no sample at a time the cell is left empty.
    """
    series = sorted(series, key=lambda s: _label_order(s[0]))
    if not series:
        return pd.DataFrame(columns=["time_s"])
    ref = _f(series[0][1])
    data: dict[str, np.ndarray] = {"time_s": ref}
    for label, t, v in series:
        t, v = _f(t), _f(v)
        if t.shape == ref.shape and np.allclose(t, ref, atol=1e-9):
            data[label] = v
        else:
            data[label] = np.interp(ref, t, v, left=np.nan, right=np.nan)
    return pd.DataFrame(data)


def mass_traces(
    cycles: CycleCollection,
    step_name: str,
    labels: Mapping[OccKey, str],
) -> pd.DataFrame:
    """Baseline-corrected mass after each pulse of one step, wide (ng/cm²)."""
    series = []
    for sr in cycles.steps(name=step_name):
        k = _key(sr)
        if k in labels:
            series.append((labels[k], sr.time, sr.mass_corrected))
    return wide_table(series)


def fit_traces(
    residuals: Any,  # noqa: ANN401
    labels: Mapping[OccKey, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fitted curve and fit loss (fit minus data, ng/cm²) per cycle, wide.

    The library stores residuals as data minus fit, so the loss is negated.
    """
    fitted, loss = [], []
    for tr in _get(residuals, "traces") or []:
        k = _key(tr)
        if k not in labels:
            continue
        fitted.append((labels[k], _get(tr, "time"), _get(tr, "fitted")))
        loss.append((labels[k], _get(tr, "time"), -_f(_get(tr, "residual"))))
    return wide_table(fitted), wide_table(loss)


def fit_loss_average_table(
    step_fits: Mapping[str, Any],
) -> pd.DataFrame:
    """Mean and spread of fit minus data over the cycles, one pair per model."""
    data: dict[str, np.ndarray] = {}
    for model_id, fit in step_fits.items():
        res = _residuals_of(fit)
        stats = _get(res, "stats") if res is not None else None
        if stats is None:
            continue
        if "time_s" not in data:
            data["time_s"] = _f(_get(stats, "time"))
        short = short_model(model_id)
        data[f"{short}_mean_fit_minus_data_ng_cm2"] = -_f(_get(stats, "mean"))
        data[f"{short}_sd_ng_cm2"] = _f(_get(stats, "std"))
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["time_s"])


def full_trace_table(mass_data: MassDataset) -> pd.DataFrame:
    """The whole run at full resolution: time, mass and any other channel."""
    cols: dict[str, Any] = {
        "time_s": mass_data.time,
        "mass_ng_cm2": mass_data.mass,
        "frequency_hz": mass_data.frequency,
    }
    if mass_data.temperature is not None:
        cols["temperature_c"] = mass_data.temperature
    return pd.DataFrame(cols)


def _record_columns(step: str, model_id: str, rec: Any) -> dict[str, float]:  # noqa: ANN401
    items = (
        rec.items() if isinstance(rec, Mapping)
        else ((f.name, getattr(rec, f.name)) for f in dataclasses.fields(rec))
    )
    out: dict[str, float] = {}
    for name, value in items:
        if name in _RECORD_SKIP or isinstance(value, bool):
            continue
        if value is not None and not isinstance(value, (int, float, np.number)):
            continue
        unit = _UNITS.get(name)
        col = f"{step}_{short_model(model_id)}_"
        col += unit if unit == "r2" else f"{name}_{unit}" if unit else name
        out[col] = np.nan if value is None else float(value)
    return out


def fit_summary_table(
    mass_data: MassDataset,
    cycle_index: CycleIndex,
    cycles: CycleCollection,
    fits: Mapping[str, Mapping[str, Any]],
    fits_per_cycle: Mapping[str, Mapping[str, list[Any]]],
    labels: Mapping[str, Mapping[OccKey, str]],
    gain: pd.DataFrame,
) -> pd.DataFrame:
    """One row per cycle with every single-number result, indexed by ``cycle``.

    Columns are ``<step>_<model>_<parameter>_<unit>`` (``TMA_mono_k_per_s``),
    plus R² and fit-loss RMSE per fit, pulse onset time, detection confidence
    and excluded flag per step, and the mass gained in each cycle.
    """
    rows: dict[str, dict[str, float]] = {}

    def row(label: str) -> dict[str, float]:
        return rows.setdefault(label, {})

    growth = cumulative_growth_table(gain)
    by_outer = {int(r.outer_cycle): r for r in growth.itertuples()}
    step_onsets = list(cycle_index.step_onsets)
    conf = cycle_index.confidence
    for step, lab in labels.items():
        results = cycles.steps(name=step, include_excluded=True)
        onsets = [mass_data.time[idx] for n, idx in step_onsets if n == step]
        confs = (
            [conf[i] for i, (n, _) in enumerate(step_onsets) if n == step]
            if len(conf) == len(step_onsets) else []
        )
        for i, sr in enumerate(results):
            k = _key(sr)
            if k not in lab:
                continue
            r = row(lab[k])
            if len(onsets) == len(results):
                r[f"{step}_pulse_onset_time_s"] = float(onsets[i])
            if len(confs) == len(results):
                r[f"{step}_pulse_confidence"] = float(confs[i])
            r[f"{step}_excluded"] = 1.0 if sr.excluded else 0.0
            r[f"{step}_mass_change_ng_cm2"] = float(sr.mass_change)
        for model_id in fits_per_cycle.get(step, {}):
            for rec in fits_per_cycle[step][model_id] or []:
                k = _key(rec)
                if k in lab:
                    row(lab[k]).update(_record_columns(step, model_id, rec))
    for step, model_id, res in iter_fit_loss(fits):
        for tr in _get(res, "traces") or []:
            k = _key(tr)
            if k in labels.get(step, {}) and _get(tr, "rmse") is not None:
                row(labels[step][k])[
                    f"{step}_{short_model(model_id)}_fit_loss_rmse_ng_cm2"
                ] = float(_get(tr, "rmse"))
    for label, r in rows.items():
        g = by_outer.get(_label_order(label)[0])
        if g is not None:
            r["mass_gain_ng_cm2"] = float(g.delta_ng_cm2)
            r["cumulative_growth_ng_cm2"] = float(g.cumulative_ng_cm2)
            r["cycle_start_time_s"] = float(g.t_start_s)
            r["cycle_end_time_s"] = float(g.t_end_s)
            r["cycle_partial"] = 1.0 if g.partial else 0.0
    if not rows:
        return pd.DataFrame(index=pd.Index([], name="cycle"))
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.loc[sorted(df.index, key=_label_order)]
    lead = [c for c in (
        "mass_gain_ng_cm2", "cumulative_growth_ng_cm2", "cycle_start_time_s",
        "cycle_end_time_s", "cycle_partial",
    ) if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.index.name = "cycle"
    return df


def write_fit_summary_xlsx(summary: pd.DataFrame, path: Any) -> None:  # noqa: ANN401
    """fit_summary as a spreadsheet: readable headers, units in the row beneath.

    Needs ``openpyxl`` (``pip install qcm-pak[xlsx]``).
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing fit_summary.xlsx needs openpyxl (pip install openpyxl)"
        ) from exc

    def split(col: str) -> tuple[str, str]:
        if col.endswith("_r2"):
            return col[:-3].replace("_", " ") + " R²", ""
        for suffix, unit in _XLSX_UNITS:
            if col.endswith(suffix):
                return col[: -len(suffix)].replace("_", " "), unit
        return col.replace("_", " "), ""

    wb = Workbook()
    ws = wb.active
    ws.title = "fit_summary"
    heads = [("cycle", "")] + [split(c) for c in summary.columns]
    ws.append([h for h, _ in heads])
    ws.append([u for _, u in heads])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for label, vals in zip(summary.index, summary.itertuples(index=False)):
        first = float(label) if "." not in str(label) else str(label)
        ws.append([first] + [None if pd.isna(v) else float(v) for v in vals])
    ws.freeze_panes = "B3"
    wb.save(path)


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


def cycle_mass_change_table(trace: pd.DataFrame) -> pd.DataFrame:
    """Mass change within each cycle against time in the cycle, wide."""
    series = [
        (str(int(k)), g["time_in_cycle_s"].to_numpy(),
         g["mass_change_in_cycle_ng_cm2"].to_numpy())
        for k, g in trace.groupby("outer_cycle", sort=True)
    ]
    return wide_table(series)


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
