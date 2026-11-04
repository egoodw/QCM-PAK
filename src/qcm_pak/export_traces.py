"""Fit-loss and mass-gain figures for the full analysis export.

The tables live in :mod:`qcm_pak.export_tables`; nothing is refitted.
:func:`~qcm_pak.export.save_full_export` calls these helpers.

Sign convention: the library stores residuals as ``data - fit``
(:class:`~qcm_pak.residuals.ResidualTrace`); the exported files use the web
app's ``fit - data`` so above zero means the fit sits above the measurement.
All masses are ng/cm².
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 (dataclass or dict)
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _f(values: Any) -> np.ndarray:  # noqa: ANN401
    """List or array to floats; ``None`` (a stored NaN) becomes NaN."""
    return np.array(
        [np.nan if v is None else v for v in values], dtype=float
    )


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


def plot_cumulative_growth(growth: pd.DataFrame) -> Figure:
    """Growth built up cycle by cycle: the total and each step's own share.

    ``growth`` has one row per cycle (index ``cycle``) and columns
    ``cumulative_growth_total_ng_cm2`` and ``cumulative_growth_<step>_ng_cm2``.
    """
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    prefix, suffix = "cumulative_growth_", "_ng_cm2"
    for col in growth.columns:
        name = col[len(prefix):-len(suffix)]
        if name == "total":
            ax.plot(growth.index, growth[col], color="black", lw=2, label="total")
        else:
            ax.plot(growth.index, growth[col], lw=1.2, label=name)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Growth since start (ng/cm²)")
    ax.set_title("Growth over the run: total and per step")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
