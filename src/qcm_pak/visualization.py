"""
Matplotlib-based visualizations for QCM analysis results.

All functions build a :class:`matplotlib.figure.Figure` directly via the
object-oriented API (``Figure(...)`` + ``fig.subplots(...)``) rather than
``pyplot.subplots()``. This deliberately avoids pyplot's global figure
registry: a figure built this way is never tracked by pyplot, so nothing
leaks and there's nothing to ``plt.close()``. That matters when this module
runs inside a long-lived, concurrent service (e.g. an HTTP API handling many
requests) rather than a one-shot script — pyplot's global state is not
meant to be shared across threads/requests, and figures registered with it
but never closed accumulate for the life of the process.

No function calls ``plt.show()`` — the caller decides whether to display or
save the returned Figure.

Public API:

- :func:`plot_trace` — full mass vs. time trace, with detected pulses highlighted
- :func:`plot_cycles` — per-step mass-change overlay across all cycles
- :func:`plot_langmuir` — Langmuir kinetics fit
- :func:`plot_etch` — etch kinetics fit
- :func:`plot_derivative` — smoothed first derivative of the mass signal
- :func:`fig_to_png_bytes` — render a Figure to PNG bytes without touching disk
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from qcm_pak._types import (
    CycleCollection,
    CycleIndex,
    EtchResult,
    LangmuirResult,
    MassDataset,
)


def fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    """Render a Figure to PNG bytes without touching disk.

    Parameters
    ----------
    fig:
        Figure to render, typically returned by one of this module's
        ``plot_*`` functions.
    dpi:
        Resolution in dots per inch.

    Returns
    -------
    bytes
        PNG image data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


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
    fig = Figure(figsize=(12, 4))
    ax = fig.subplots()
    ax.plot(data.time / 60.0, data.mass, lw=0.8, color="steelblue", label="Mass")

    if index is not None:
        unique_names = list(dict.fromkeys(n for n, _ in index.step_onsets))
        cmap = colormaps["tab10"]
        colors = cmap(np.linspace(0, 0.9, max(1, len(unique_names))))
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
    if step is not None:
        step_names = [step]
    else:
        step_names = cycles.recipe.step_names()

    n = len(step_names)
    fig = Figure(figsize=(6 * n, 4))
    axes = fig.subplots(1, n, squeeze=False)

    cmap = colormaps["viridis"]
    for col, name in enumerate(step_names):
        ax = axes[0, col]
        step_results = cycles.steps(name=name)
        if not step_results:
            ax.set_title(f"{name} (no data)")
            continue
        n_traces = len(step_results)
        colors = cmap(np.linspace(0.1, 0.9, max(1, n_traces)))
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

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
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

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
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
    from scipy.signal import savgol_filter

    dt = data.dt
    mass = data.mass
    n = len(mass)

    win = min(51, n // 4 * 2 + 1)   # always odd: n//4*2 is even, +1 makes odd
    poly = min(3, win - 1)
    smooth = savgol_filter(mass, window_length=win, polyorder=poly)
    deriv = np.gradient(smooth, dt)

    fig = Figure(figsize=(12, 6))
    axes = fig.subplots(2, 1, sharex=True)
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
