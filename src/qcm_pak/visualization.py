"""
Matplotlib-based visualizations for QCM analysis results.

All functions return a :class:`matplotlib.figure.Figure` and do not call
``plt.show()`` — the caller decides whether to display or save.

Public API:

- :func:`plot_trace` — full mass vs. time trace, with detected pulses highlighted
- :func:`plot_cycles` — per-step mass-change overlay across all cycles
- :func:`plot_langmuir` — Langmuir kinetics fit
- :func:`plot_etch` — etch kinetics fit
- :func:`plot_derivative` — smoothed first derivative of the mass signal
- :func:`plot_detailed_cycles` — zoomed trace over a span of outer cycles
- :func:`plot_cycles_batch` — per-step overlay restricted to an outer-cycle range
- :func:`plot_cycle_average` — mean ± std envelope of a step's traces
- :func:`plot_pulse_timing` — histogram of onset-to-onset intervals
- :func:`plot_fit_drift` — a per-cycle fit parameter plotted across cycles
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from qcm_pak._types import (
    CycleCollection,
    CycleIndex,
    EtchResult,
    LangmuirResult,
    MassDataset,
)


def plot_trace(
    data: MassDataset,
    index: CycleIndex | None = None,
) -> Figure:
    """Plot mass vs. time for the full experiment.

    Parameters
    ----------
    data:
        Sauerbrey-converted mass dataset.
    index:
        If provided, vertical lines are drawn at each detected pulse onset.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.time / 60.0, data.mass, lw=0.8, color="steelblue", label="Mass")

    if index is not None:
        unique_names = list(dict.fromkeys(n for n, _ in index.step_onsets))
        colors = plt.cm.tab10(np.linspace(0, 0.9, max(1, len(unique_names))))  # type: ignore[attr-defined]
        color_map = dict(zip(unique_names, colors))

        first_seen: set[str] = set()
        for step_name, idx in index.step_onsets:
            t_min = data.time[idx] / 60.0
            label = step_name if step_name not in first_seen else None
            ax.axvline(
                t_min,
                color=color_map[step_name],
                lw=0.6,
                alpha=0.7,
                linestyle="--",
                label=label,
            )
            first_seen.add(step_name)
        ax.legend(fontsize=8, ncol=min(4, len(unique_names)))

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Δm (ng/cm²)")
    ax.set_title("QCM Mass Trace")
    fig.tight_layout()
    return fig


def plot_cycles(
    cycles: CycleCollection,
    step: str | None = None,
) -> Figure:
    """Overlay mass-change time traces for all step occurrences.

    Each trace is baseline-corrected (``mass_corrected``) and aligned to its
    pulse onset (time[0] = 0).

    Parameters
    ----------
    cycles:
        Extracted cycle collection.
    step:
        If given, only traces for this step name are shown. If ``None``,
        all steps are plotted on separate axes.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    if step is not None:
        step_names = [step]
    else:
        step_names = cycles.recipe.step_names()

    n = len(step_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for col, name in enumerate(step_names):
        ax = axes[0, col]
        step_results = cycles.steps(name=name)
        if not step_results:
            ax.set_title(f"{name} (no data)")
            continue
        n_traces = len(step_results)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, n_traces)))  # type: ignore[attr-defined]
        for i, sr in enumerate(step_results):
            ax.plot(
                sr.time,
                sr.mass_corrected,
                lw=0.8,
                alpha=0.6,
                color=colors[i],
            )
        ax.set_xlabel("Time in step (s)")
        ax.set_ylabel("Δm (ng/cm²)")
        ax.set_title(f"{name}")

    fig.suptitle("Per-step mass traces")
    fig.tight_layout()
    return fig


def plot_langmuir(
    cycles: CycleCollection,
    result: LangmuirResult,
) -> Figure:
    """Plot Langmuir kinetics fit alongside the experimental data points.

    Parameters
    ----------
    cycles:
        Extracted cycle collection (used to retrieve the data points).
    result:
        Fit result from :func:`~qcm_pak.kinetics.fit_langmuir`.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    step_results = cycles.steps(name=result.step_name)
    min_len = min(len(s.mass_corrected) for s in step_results)
    t_exp = step_results[0].time[:min_len]
    theta_exp = np.mean(
        np.vstack([np.abs(s.mass_corrected[:min_len]) for s in step_results]), axis=0
    )

    t_fit = np.linspace(0, float(np.max(t_exp)) * 1.1, 300)

    if result.model == "mono":
        theta_fit = result.theta_max * (1.0 - np.exp(-result.k * t_fit))  # type: ignore[operator]
        label = (
            f"Mono fit: θ_max={result.theta_max:.2f} ng/cm², k={result.k:.4f} s⁻¹\n"  # type: ignore[operator]
            f"R²={result.r_squared:.4f}"
        )
    else:
        theta_fit = (
            result.theta1 * (1.0 - np.exp(-result.k1 * t_fit))  # type: ignore[operator]
            + result.theta2 * (1.0 - np.exp(-result.k2 * t_fit))  # type: ignore[operator]
        )
        label = (
            f"Bi fit: θ₁={result.theta1:.2f}, k₁={result.k1:.4f} s⁻¹\n"  # type: ignore[operator]
            f"θ₂={result.theta2:.2f}, k₂={result.k2:.4f} s⁻¹  R²={result.r_squared:.4f}"  # type: ignore[operator]
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_exp, theta_exp, lw=0.8, color="steelblue", alpha=0.7, label="Avg data")
    ax.plot(t_fit, theta_fit, lw=1.5, color="tomato", label=label)
    ax.set_xlabel("Pulse time (s)")
    ax.set_ylabel("Mass change (ng/cm²)")
    ax.set_title(f"Langmuir fit — {result.step_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_etch(
    cycles: CycleCollection,
    result: EtchResult,
) -> Figure:
    """Plot etch kinetics fit alongside experimental data points.

    Parameters
    ----------
    cycles:
        Extracted cycle collection.
    result:
        Fit result from :func:`~qcm_pak.kinetics.fit_etch`.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    step_results = cycles.steps(name=result.step_name)
    min_len = min(len(s.mass_corrected) for s in step_results)
    t_exp = step_results[0].time[:min_len]
    etch_exp = np.mean(
        np.vstack([np.abs(s.mass_corrected[:min_len]) for s in step_results]), axis=0
    )

    t_fit = np.linspace(0, float(np.max(t_exp)) * 1.1, 300)

    if result.model == "saturating":
        etch_fit = result.etch_max * (1.0 - np.exp(-result.k * t_fit))  # type: ignore[operator]
        label = (
            f"Saturating: E_max={result.etch_max:.3f} ng/cm², k={result.k:.4f} s⁻¹\n"  # type: ignore[operator]
            f"R²={result.r_squared:.4f}"
        )
    else:
        etch_fit = result.rate * t_fit  # type: ignore[operator]
        label = f"Linear: rate={result.rate:.4f} ng/cm²/s  R²={result.r_squared:.4f}"  # type: ignore[operator]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_exp, etch_exp, lw=0.8, color="steelblue", alpha=0.7, label="Avg data")
    ax.plot(t_fit, etch_fit, lw=1.5, color="tomato", label=label)
    ax.set_xlabel("Pulse time (s)")
    ax.set_ylabel("Etch per step (ng/cm²)")
    ax.set_title(f"Etch kinetics — {result.step_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_derivative(data: MassDataset) -> Figure:
    """Plot the smoothed first derivative of the mass signal.

    Useful for diagnosing pulse detection quality: each ALD pulse should
    appear as a sharp positive (gain) or negative (loss) spike.

    Parameters
    ----------
    data:
        Sauerbrey-converted mass dataset.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    mass = data.mass
    n = len(mass)

    win = min(51, n // 4 * 2 + 1)   # always odd: n//4*2 is even, +1 makes odd
    poly = min(3, win - 1)
    smooth = savgol_filter(mass, window_length=win, polyorder=poly)
    deriv = np.gradient(smooth, data.time)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(data.time / 60.0, mass, lw=0.6, color="steelblue")
    axes[0].set_ylabel("Δm (ng/cm²)")
    axes[0].set_title("Mass trace")

    axes[1].plot(data.time / 60.0, deriv, lw=0.6, color="darkorange")
    axes[1].axhline(0, color="k", lw=0.4, linestyle="--")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("dΔm/dt (ng/cm²/s)")
    axes[1].set_title("Smoothed first derivative")

    fig.tight_layout()
    return fig


def plot_detailed_cycles(
    data: MassDataset,
    index: CycleIndex,
    location: str = "first",
    n_cycles: int = 10,
) -> Figure:
    """Zoomed view of the raw mass trace over a span of outer cycles.

    Useful for spot-checking detection quality at different points in a
    long run — pulse edges that look clean in the first few cycles can
    drift or degrade by the end.

    Parameters
    ----------
    data:
        Sauerbrey-converted mass dataset.
    index:
        Detected pulse onsets — also carries the ``Recipe`` used to
        determine how many events make up one outer cycle.
    location:
        Which span of the experiment to zoom into: ``"first"``,
        ``"middle"``, or ``"last"`` ``n_cycles`` outer repeats.
    n_cycles:
        How many outer cycles to show.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    recipe = index.recipe
    steps_per_cycle = recipe.steps_per_cycle
    total_cycles = recipe.repeats
    n = max(1, min(n_cycles, total_cycles))

    if location == "last":
        c0 = max(0, total_cycles - n)
    elif location == "middle":
        c0 = max(0, (total_cycles - n) // 2)
    else:
        c0 = 0
    c1 = min(total_cycles, c0 + n)

    lo_event = c0 * steps_per_cycle
    hi_event = min(len(index.step_onsets), c1 * steps_per_cycle)

    fig, ax = plt.subplots(figsize=(12, 4))
    if hi_event <= lo_event:
        ax.set_title(f"No events in cycles [{c0}, {c1})")
        fig.tight_layout()
        return fig

    onsets_window = index.step_onsets[lo_event:hi_event]
    idx_lo = onsets_window[0][1]
    idx_hi = onsets_window[-1][1]
    pad = max(1, (idx_hi - idx_lo) // 20)  # a little context on each side
    idx_lo = max(0, idx_lo - pad)
    idx_hi = min(len(data.time) - 1, idx_hi + pad)

    window = slice(idx_lo, idx_hi + 1)
    ax.plot(data.time[window], data.mass[window], lw=0.8, color="steelblue")

    unique_names = list(dict.fromkeys(name for name, _ in onsets_window))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(1, len(unique_names))))  # type: ignore[attr-defined]
    color_map = dict(zip(unique_names, colors))
    first_seen: set[str] = set()
    for step_name, onset_idx in onsets_window:
        label = step_name if step_name not in first_seen else None
        ax.axvline(
            data.time[onset_idx], color=color_map[step_name],
            lw=0.8, alpha=0.8, linestyle="--", label=label,
        )
        first_seen.add(step_name)
    ax.legend(fontsize=8, ncol=min(4, len(unique_names)))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δm (ng/cm²)")
    ax.set_title(f"Detailed view — cycles {c0 + 1}-{c1} of {total_cycles} ({location})")
    fig.tight_layout()
    return fig


def plot_cycles_batch(
    cycles: CycleCollection,
    step: str,
    cycle_range: tuple[int, int] | None = None,
) -> Figure:
    """Overlay mass-change traces for one step, restricted to an outer-cycle range.

    Same idea as :func:`plot_cycles` for a single step, but only includes
    occurrences whose ``outer_cycle`` falls in ``cycle_range`` (inclusive
    lower, exclusive upper) — splits a long run into manageable batches of
    figures instead of one increasingly-illegible overlay of everything.

    Parameters
    ----------
    cycles:
        Extracted cycle collection.
    step:
        Step name to plot.
    cycle_range:
        ``(lo, hi)`` outer-cycle bounds, or ``None`` for every cycle.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    step_results = cycles.steps(name=step)
    if cycle_range is not None:
        lo, hi = cycle_range
        step_results = [s for s in step_results if lo <= s.outer_cycle < hi]

    fig, ax = plt.subplots(figsize=(7, 4))
    label = (
        f"cycles {cycle_range[0] + 1}-{cycle_range[1]}" if cycle_range else "all cycles"
    )
    if not step_results:
        ax.set_title(f"{step} — {label} (no data)")
        fig.tight_layout()
        return fig

    n_traces = len(step_results)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, n_traces)))  # type: ignore[attr-defined]
    for i, sr in enumerate(step_results):
        ax.plot(sr.time, sr.mass_corrected, lw=0.8, alpha=0.6, color=colors[i])
    ax.set_xlabel("Time in step (s)")
    ax.set_ylabel("Δm (ng/cm²)")
    ax.set_title(f"{step} — {label}")
    fig.tight_layout()
    return fig


def plot_cycle_average(
    cycles: CycleCollection,
    step: str,
    cycle_range: tuple[int, int] | None = None,
) -> Figure:
    """Mean ± 1σ envelope of a step's baseline-corrected mass trace.

    Parameters
    ----------
    cycles:
        Extracted cycle collection.
    step:
        Step name to plot.
    cycle_range:
        ``(lo, hi)`` outer-cycle bounds, or ``None`` for every cycle.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    step_results = cycles.steps(name=step)
    if cycle_range is not None:
        lo, hi = cycle_range
        step_results = [s for s in step_results if lo <= s.outer_cycle < hi]

    fig, ax = plt.subplots(figsize=(7, 4))
    label = (
        f"cycles {cycle_range[0] + 1}-{cycle_range[1]}" if cycle_range else "all cycles"
    )
    if not step_results:
        ax.set_title(f"{step} average — {label} (no data)")
        fig.tight_layout()
        return fig

    min_len = min(len(s.mass_corrected) for s in step_results)
    t = step_results[0].time[:min_len]
    stack = np.vstack([s.mass_corrected[:min_len] for s in step_results])
    mean, std = stack.mean(axis=0), stack.std(axis=0)

    ax.plot(t, mean, lw=1.2, color="steelblue", label="mean")
    ax.fill_between(
        t, mean - std, mean + std, color="steelblue", alpha=0.25, label="±1σ"
    )
    ax.set_xlabel("Time in step (s)")
    ax.set_ylabel("Δm (ng/cm²)")
    ax.set_title(f"{step} average — {label} (n={len(step_results)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pulse_timing(
    index: CycleIndex,
    data: MassDataset,
    step: str | None = None,
) -> Figure:
    """Histogram of onset-to-onset intervals — a drift/jitter diagnostic.

    A tight, single-peaked histogram means pulse timing is consistent
    across the run; a spread-out or multi-modal one usually means either
    genuine process drift or a detection issue.

    Parameters
    ----------
    index:
        Detected pulse onsets.
    data:
        Sauerbrey-converted mass dataset (for converting onset indices to
        times).
    step:
        If given, only intervals between consecutive onsets of this step
        are used; otherwise all onset-to-onset intervals are used.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    onsets = [
        (name, idx) for name, idx in index.step_onsets if step is None or name == step
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(onsets) < 2:
        ax.set_title("Not enough pulses to plot timing")
        fig.tight_layout()
        return fig

    times = np.array([data.time[idx] for _, idx in onsets])
    intervals = np.diff(times)

    # Near-uniform timing (every interval identical or equal up to float
    # noise, e.g. a tiny or synthetic run) gives a near-zero-width range —
    # np.histogram can't lay out >1 bin edge across that without adjacent
    # edges colliding at float precision. A relative-tolerance check (not
    # an exact-zero one, since two "equal" floats computed via different
    # paths rarely compare bit-identical) catches that case; a single bin
    # is the correct rendering of "no meaningful spread" either way.
    spread = np.ptp(intervals)
    scale = max(1e-12, float(np.mean(np.abs(intervals))))
    n_bins = min(40, max(5, len(intervals) // 2)) if spread > 1e-9 * scale else 1
    ax.hist(intervals, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Onset-to-onset interval (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Pulse timing — {step}" if step else "Pulse timing — all steps")
    fig.tight_layout()
    return fig


def plot_fit_drift(
    records: list[Mapping[str, Any]],
    step_name: str,
    param: str = "k",
) -> Figure:
    """Plot a fit parameter's value across cycles — a drift/trend diagnostic.

    Parameters
    ----------
    records:
        Per-cycle fit results, as produced by
        :func:`~qcm_pak.kinetics.fit_langmuir_per_cycle` /
        :func:`~qcm_pak.kinetics.fit_etch_per_cycle` (or the equivalent
        JSON-safe dicts) — each needs an ``outer_cycle`` key/attribute and
        the ``param`` key/attribute being plotted.
    step_name:
        For the plot title.
    param:
        Which field to plot on the y-axis (e.g. ``"k"``, ``"theta_max"``,
        ``"etch_max"``, ``"rate"``).

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    def _get(r: Mapping[str, Any], key: str) -> Any:  # noqa: ANN401 (dict or object)
        return r.get(key) if isinstance(r, Mapping) else getattr(r, key, None)

    pts = [(_get(r, "outer_cycle"), _get(r, param)) for r in records]
    pts = [(c, v) for c, v in pts if c is not None and v is not None]
    pts.sort(key=lambda cv: cv[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    if not pts:
        ax.set_title(f"{step_name} — no per-cycle {param} data")
        fig.tight_layout()
        return fig
    cycles_x, values_y = zip(*pts)
    ax.plot(cycles_x, values_y, marker="o", ms=3, lw=0.8, color="steelblue")
    ax.set_xlabel("Outer cycle")
    ax.set_ylabel(param)
    ax.set_title(f"{step_name} — {param} vs cycle")
    fig.tight_layout()
    return fig
