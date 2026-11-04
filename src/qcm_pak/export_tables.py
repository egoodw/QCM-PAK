"""Tables for the full analysis export: raw trace, wide traces, fit summary.

Everything here turns data the analysis already holds into tables; nothing is
refitted. :func:`~qcm_pak.export.save_full_export` calls these helpers.

Conventions
-----------
- Traces are *wide*: column 1 is ``time_s`` (zero at the pulse onset, or at
  the start of the cycle for the cycle-mass-change file) and every other
  column is one cycle, headed by its 1-based number (``7``; ``7.1``/``7.2``
  when a step runs more than once in a cycle). Series shorter than the
  longest one end in empty cells.
- Fit loss is *fit minus data* (ng/cm²); the library stores data - fit
  (:class:`~qcm_pak.residuals.ResidualTrace`), so it is negated here.
- ``fit_summary`` has one row per outer cycle and the unit in every column
  name. All masses are ng/cm², times are seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qcm_pak._types import CycleCollection, CycleIndex, MassDataset

# fit-result field -> unit suffix of its fit_summary column, per model family
_PARAM_UNITS: dict[str, str] = {
    "k": "k_per_s",
    "theta_max": "theta_max_ng_cm2",
    "k1": "k1_per_s",
    "k2": "k2_per_s",
    "theta1": "theta1_ng_cm2",
    "theta2": "theta2_ng_cm2",
    "etch_max": "etch_max_ng_cm2",
    "rate": "rate_ng_cm2_per_s",
}
_MODEL_PARAMS: dict[str, list[str]] = {
    "langmuir_mono": ["k", "theta_max"],
    "langmuir_bi": ["k1", "k2", "theta1", "theta2"],
    "etch_saturating": ["k", "etch_max"],
    "etch_linear": ["rate"],
}


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 (dataclass or dict)
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _f(values: Any) -> np.ndarray:  # noqa: ANN401
    """List or array to floats; ``None`` (a stored NaN) becomes NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def model_short(model_id: str) -> str:
    """``langmuir_mono`` -> ``mono``, ``langmuir_bi`` -> ``bi``; others as is."""
    return model_id.removeprefix("langmuir_")


def occurrence_labels(outer_cycles: Sequence[int]) -> list[str]:
    """Header per occurrence: ``7`` for one run in cycle 7, ``7.1``, ``7.2`` ...

    ``outer_cycles`` are 0-based and in chronological order, one per
    occurrence.
    """
    counts: dict[int, int] = {}
    for k in outer_cycles:
        counts[k] = counts.get(k, 0) + 1
    seen: dict[int, int] = {}
    labels = []
    for k in outer_cycles:
        seen[k] = seen.get(k, 0) + 1
        labels.append(f"{k + 1}.{seen[k]}" if counts[k] > 1 else str(k + 1))
    return labels


def wide_table(
    labels: Sequence[str],
    times: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
) -> pd.DataFrame:
    """``time_s`` plus one column per series, on the longest series' time grid.

    Shorter series are placed on that grid by linear interpolation and are
    empty past their own end.
    """
    if not labels:
        return pd.DataFrame(columns=["time_s"])
    grid = max(times, key=len)
    out: dict[str, np.ndarray] = {"time_s": np.asarray(grid, dtype=float)}
    for label, t, v in zip(labels, times, values):
        if len(t) == len(grid) and np.array_equal(t, grid):
            out[label] = np.asarray(v, dtype=float)
        else:
            out[label] = np.interp(grid, t, v, left=np.nan, right=np.nan)
    return pd.DataFrame(out)


def raw_trace_table(mass_data: MassDataset) -> pd.DataFrame:
    """The whole measured trace at full resolution."""
    cols: dict[str, Any] = {
        "time_s": mass_data.time,
        "mass_ng_cm2": mass_data.mass,
        "frequency_hz": mass_data.frequency,
    }
    if getattr(mass_data, "temperature", None) is not None:
        cols["temperature_c"] = mass_data.temperature
    return pd.DataFrame(cols)


def step_mass_table(cycles: CycleCollection, step_name: str) -> pd.DataFrame:
    """Baseline-corrected mass of every (non-excluded) occurrence of a step."""
    steps = cycles.steps(name=step_name)
    return wide_table(
        occurrence_labels([s.outer_cycle for s in steps]),
        [np.asarray(s.time, dtype=float) for s in steps],
        [np.asarray(s.mass_corrected, dtype=float) for s in steps],
    )


def iter_fit_loss(
    fits: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str, Any]]:
    """``(step_name, model_id, residuals)`` for every fit that has traces."""
    found = []
    for step_name, step_fits in fits.items():
        for model_id, fit in step_fits.items():
            res = _get(fit, "residuals")
            if res is not None and _get(res, "traces"):
                found.append((step_name, model_id, res))
    return found


def fit_trace_tables(residuals: Any) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: ANN401
    """``(fit, fit_loss)`` wide tables of one step and model.

    ``fit`` is the fitted curve on each occurrence's time axis; ``fit_loss``
    is fit minus data. ``fit_loss`` also carries the ``mean``, ``sd`` and
    ``n`` of the occurrences at each time, when the library stored them.
    """
    traces = list(_get(residuals, "traces") or [])
    labels = occurrence_labels([int(_get(t, "outer_cycle")) for t in traces])
    times = [_f(_get(t, "time")) for t in traces]
    fit = wide_table(labels, times, [_f(_get(t, "fitted")) for t in traces])
    loss = wide_table(labels, times, [-_f(_get(t, "residual")) for t in traces])
    stats = _get(residuals, "stats")
    if stats is not None and len(loss):
        grid = loss["time_s"].to_numpy()
        t = _f(_get(stats, "time"))
        loss["mean"] = np.interp(grid, t, -_f(_get(stats, "mean")),
                                 left=np.nan, right=np.nan)
        loss["sd"] = np.interp(grid, t, _f(_get(stats, "std")),
                               left=np.nan, right=np.nan)
        n = np.interp(grid, t, _f(_get(stats, "n")), left=np.nan, right=np.nan)
        loss["n"] = pd.array(
            [None if np.isnan(x) else int(round(x)) for x in n], dtype="Int64"
        )
    return fit, loss


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
    columns = [
        "outer_cycle", "time_s", "time_in_cycle_s", "mass_ng_cm2",
        "mass_change_in_cycle_ng_cm2", "cumulative_growth_ng_cm2", "partial",
    ]
    time = np.asarray(mass_data.time, dtype=float)
    mass = np.asarray(mass_data.mass, dtype=float)
    first_onset: dict[int, float] = {}
    last_end: dict[int, float] = {}
    durations = _step_durations(index)
    by_name: dict[str, list[float]] = {}
    for name, idx in index.step_onsets:
        by_name.setdefault(name, []).append(float(time[idx]))
    for name, onsets in by_name.items():
        results = cycles.steps(name=name, include_excluded=True)
        if len(results) != len(onsets):
            return pd.DataFrame(columns=columns)
        for sr, t in zip(results, onsets):
            k = sr.outer_cycle
            first_onset[k] = min(t, first_onset.get(k, t))
            end = t + durations.get(name, 0.0)
            last_end[k] = max(end, last_end.get(k, end))
    if not first_onset or time.size < 2:
        return pd.DataFrame(columns=columns)

    order = sorted(first_onset)
    starts = {k: first_onset[k] for k in order}
    starts[order[0]] = float(index.recipe.start_time or time[0])
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
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


TOTAL_GROWTH = "cumulative_growth_total_ng_cm2"


def cycle_growth_table(gain: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle (index ``cycle``): start, growth and running total."""
    rows = {}
    for k, g in gain.groupby("outer_cycle", sort=True):
        rows[int(k)] = {
            "cycle_t_start_s": g["time_s"].iloc[0],
            "cycle_partial": g["partial"].iloc[0] == "yes",
            "mass_gain_ng_cm2": g["mass_change_in_cycle_ng_cm2"].iloc[-1],
            TOTAL_GROWTH: g["cumulative_growth_ng_cm2"].iloc[-1],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def step_growth_table(
    cycles: CycleCollection, step_names: Sequence[str], n_cycles: int
) -> pd.DataFrame:
    """Running total of each step's mass change, one column per step.

    Index ``cycle`` (1-based). A step that runs twice in a cycle adds both
    runs. A cycle without a (usable) occurrence adds nothing. The columns
    are named ``cumulative_growth_<step>_ng_cm2``.
    """
    table = pd.DataFrame(index=pd.RangeIndex(1, n_cycles + 1, name="cycle"))
    for step in step_names:
        per_cycle = pd.Series(0.0, index=table.index)
        for sr in cycles.steps(name=step, include_excluded=True):
            if sr.excluded or sr.mass_change is None:
                continue
            value = float(sr.mass_change)
            if np.isfinite(value) and sr.outer_cycle + 1 in per_cycle.index:
                per_cycle[sr.outer_cycle + 1] += value
        table[f"cumulative_growth_{step}_ng_cm2"] = per_cycle.cumsum()
    return table


def cycle_mass_change_table(gain: pd.DataFrame) -> pd.DataFrame:
    """Mass change since the cycle's start, one column per cycle (wide)."""
    ks = sorted(gain["outer_cycle"].unique()) if not gain.empty else []
    groups = [gain[gain["outer_cycle"] == k] for k in ks]
    return wide_table(
        [str(int(k)) for k in ks],
        [g["time_in_cycle_s"].to_numpy(float) for g in groups],
        [g["mass_change_in_cycle_ng_cm2"].to_numpy(float) for g in groups],
    )


def _ranked(keys: Sequence[int]) -> list[int]:
    """0, 1, ... for repeated cycle numbers in order of appearance."""
    seen: dict[int, int] = {}
    ranks = []
    for k in keys:
        ranks.append(seen.get(k, 0))
        seen[k] = seen.get(k, 0) + 1
    return ranks


def fit_summary_table(
    mass_data: MassDataset,
    index: CycleIndex,
    cycles: CycleCollection,
    fits: Mapping[str, Mapping[str, Any]],
    fits_per_cycle: Mapping[str, Mapping[str, Sequence[Any]]],
    gain: pd.DataFrame,
    extra_cycle_columns: Mapping[str, Mapping[int, float]] | None = None,
) -> pd.DataFrame:
    """One row per cycle (index ``cycle``, 1-based) with every per-cycle number.

    Columns: cycle start, mass gain, ``cumulative_growth_total_ng_cm2`` (the
    measured mass since the recipe start) and ``cumulative_growth_<step>_ng_cm2``
    (the running sum of that step's own mass change); per step the pulse
    onset, confidence, excluded flag and mass change of its occurrence in the
    cycle; per fitted model the per-cycle parameters, ``r2`` and the RMSE of
    that occurrence's fit loss. A step that runs again within a cycle gets a
    second set with the prefix ``<step>_run2_``. ``extra_cycle_columns`` adds
    caller-supplied ``{column: {cycle (1-based): value}}`` columns.
    """
    n_cycles = max(
        [index.recipe.repeats]
        + [s.outer_cycle + 1 for s in cycles.steps(include_excluded=True)]
    )
    table = pd.DataFrame(index=pd.RangeIndex(1, n_cycles + 1, name="cycle"))
    new: dict[str, dict[int, Any]] = {}
    if not gain.empty:
        for col, ser in cycle_growth_table(gain).items():
            new[str(col)] = ser.to_dict()
    steps_growth = step_growth_table(cycles, index.recipe.step_names(), n_cycles)
    for col, ser in steps_growth.items():
        new[str(col)] = ser.to_dict()

    time = np.asarray(mass_data.time, dtype=float)
    ordinals: dict[str, list[int]] = {}
    for i, (name, _idx) in enumerate(index.step_onsets):
        ordinals.setdefault(name, []).append(i)
    onset_t: dict[str, list[float]] = {}
    for name, idx in index.step_onsets:
        onset_t.setdefault(name, []).append(float(time[idx]))

    def put(col: str, cyc: int, value: Any) -> None:  # noqa: ANN401
        new.setdefault(col, {})[cyc + 1] = value

    def prefix(step: str, rank: int) -> str:
        return step if rank == 0 else f"{step}_run{rank + 1}"

    for step in index.recipe.step_names():
        occ = cycles.steps(name=step, include_excluded=True)
        ranks = _ranked([s.outer_cycle for s in occ])
        aligned = len(onset_t.get(step, [])) == len(occ)
        for pos, (sr, rk) in enumerate(zip(occ, ranks)):
            pre = prefix(step, rk)
            put(f"{pre}_mass_change_ng_cm2", sr.outer_cycle, sr.mass_change)
            put(f"{pre}_excluded", sr.outer_cycle, bool(sr.excluded))
            if aligned:
                put(f"{pre}_pulse_onset_time_s", sr.outer_cycle,
                    onset_t[step][pos])
                ordinal = ordinals[step][pos]
                if ordinal < len(index.confidence):
                    put(f"{pre}_pulse_confidence", sr.outer_cycle,
                        index.confidence[ordinal])

        for model_id in {**fits.get(step, {}), **fits_per_cycle.get(step, {})}:
            short = model_short(model_id)
            records = list(fits_per_cycle.get(step, {}).get(model_id) or [])
            rk = _ranked([int(_get(r, "outer_cycle")) for r in records])
            for r, rank in zip(records, rk):
                pre = f"{prefix(step, rank)}_{short}"
                cyc = int(_get(r, "outer_cycle"))
                for p in _MODEL_PARAMS.get(model_id, []):
                    put(f"{pre}_{_PARAM_UNITS[p]}", cyc, _get(r, p))
                put(f"{pre}_r2", cyc, _get(r, "r_squared"))
            res = _get(fits.get(step, {}).get(model_id), "residuals")
            traces = list(_get(res, "traces") or []) if res is not None else []
            rk = _ranked([int(_get(t, "outer_cycle")) for t in traces])
            for t, rank in zip(traces, rk):
                put(f"{prefix(step, rank)}_{short}_fit_loss_rmse_ng_cm2",
                    int(_get(t, "outer_cycle")), _get(t, "rmse"))

    for col, values in (extra_cycle_columns or {}).items():
        for cyc, v in values.items():
            put(col, int(cyc) - 1, v)

    for col, values in new.items():
        table[col] = pd.Series(values).reindex(table.index)
    return table


_UNIT_TEXT = {
    "per_s_per_ng_cm2": "1/s",
    "ng_cm2_per_s": "ng/cm²/s",
    "ng_cm2": "ng/cm²",
    "per_s": "1/s",
    "s": "s",
}


def split_unit(column: str) -> tuple[str, str]:
    """``TMA_mono_k_per_s`` -> (``TMA mono k``, ``1/s``); no unit -> ``""``."""
    for suffix, text in _UNIT_TEXT.items():
        if column.endswith("_" + suffix):
            return column[: -len(suffix) - 1].replace("_", " "), text
    return column.replace("_", " "), ""


def write_fit_summary_xlsx(csv_path: str | Path, xlsx_path: str | Path) -> bool:
    """Write ``fit_summary.xlsx`` from ``fit_summary.csv``: readable names with
    the unit on a row below.

    Needs ``openpyxl`` (not a dependency of qcm-pak). Returns ``False`` without
    writing anything when it is not installed.
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        return False
    table = pd.read_csv(csv_path)
    wb = Workbook()
    ws = wb.active
    ws.title = "fit_summary"
    names, units = zip(*(split_unit(str(c)) for c in table.columns))
    ws.append(list(names))
    ws.append(list(units))
    for row in table.itertuples(index=False):
        ws.append([None if pd.isna(v) else v for v in row])
    wb.save(xlsx_path)
    return True
