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
"""

from __future__ import annotations

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

    dt = data.dt
    mass = data.mass
    n = len(mass)

    win = min(51, n // 4 * 2 + 1)   # always odd: n//4*2 is even, +1 makes odd
    poly = min(3, win - 1)
    smooth = savgol_filter(mass, window_length=win, polyorder=poly)
    deriv = np.gradient(smooth, dt)

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
