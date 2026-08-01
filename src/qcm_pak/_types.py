"""
All data-transfer and result dataclasses.

Results are organised in a three-level hierarchy that mirrors the Recipe structure:

    StepResult      — one detected pulse event (finest grain)
    SubCycleRun     — one execution of a SubCycle (n A+B runs, m A+C runs, …)
    Cycle           — one outer Recipe repeat (what researchers call "one cycle")
    CycleCollection — all results, queryable at any level

Kinetics results:

    LangmuirResult  — returned by fit_langmuir()
    EtchResult      — returned by fit_etch()

Top-level container:

    AnalysisResult  — returned by run_analysis(); call .save() to write files
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from qcm_pak.recipe import Recipe
from qcm_pak.serialization import to_jsonable

if TYPE_CHECKING:
    from qcm_pak.parameters import ALDParameters


# ── Raw and converted data ────────────────────────────────────────────────────


@dataclass
class QCMDataset:
    """Raw QCM measurement loaded from a CSV file.

    All arrays share the same length (one entry per measurement point).
    Time is always in seconds; frequency is always in Hz regardless of the
    units in the source file.

    Parameters
    ----------
    time:
        Elapsed time in seconds from the first measurement.
    frequency:
        Crystal resonance frequency in Hz.
    temperature:
        Sample temperature in °C, if a temperature column was provided.
    source_path:
        Path to the source file, for traceability.
    """

    time: NDArray[np.float64]
    frequency: NDArray[np.float64]
    temperature: NDArray[np.float64] | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if len(self.time) != len(self.frequency):
            raise ValueError(
                f"time and frequency arrays must have the same length, "
                f"got {len(self.time)} and {len(self.frequency)}"
            )
        if self.temperature is not None and len(self.temperature) != len(self.time):
            raise ValueError(
                f"temperature array length {len(self.temperature)} does not "
                f"match time array length {len(self.time)}"
            )

    @property
    def n_points(self) -> int:
        """Number of measurement points."""
        return len(self.time)

    @property
    def duration(self) -> float:
        """Total measurement duration in seconds."""
        return float(self.time[-1] - self.time[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class MassDataset:
    """QCM data with Sauerbrey-converted areal mass density.

    Produced by :func:`~qcm_pak.sauerbrey.frequency_to_mass`. The ``mass``
    array represents cumulative areal mass change (ng/cm²) relative to the
    first measurement point.

    Parameters
    ----------
    time:
        Elapsed time in seconds.
    frequency:
        Crystal resonance frequency in Hz.
    mass:
        Areal mass density change in ng/cm² (positive = mass gain,
        negative = mass loss relative to first point).
    temperature:
        Sample temperature in °C, if available.
    """

    time: NDArray[np.float64]
    frequency: NDArray[np.float64]
    mass: NDArray[np.float64]
    temperature: NDArray[np.float64] | None = None

    @property
    def n_points(self) -> int:
        """Number of measurement points."""
        return len(self.time)

    @property
    def dt(self) -> float:
        """Median positive sampling interval in seconds.

        Ignores zero-difference rows (duplicate timestamps from instruments
        that write multiple identical rows per measurement interval).
        """
        diffs = np.diff(self.time)
        pos = diffs[diffs > 0]
        return float(np.median(pos)) if len(pos) > 0 else float(np.max(np.abs(diffs)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class CycleIndex:
    """Detected pulse onset positions, structured to mirror the Recipe.

    Produced by :func:`~qcm_pak.detection.detect_pulses` and consumed by
    :func:`~qcm_pak.extraction.extract_cycles`.

    Parameters
    ----------
    step_onsets:
        Chronologically ordered list of ``(step_name, array_index)`` tuples,
        where ``array_index`` is a 0-based index into the :class:`MassDataset`
        arrays. The order follows the recipe step traversal order.
    recipe:
        The Recipe used for detection, kept for reference during extraction.
    """

    step_onsets: list[tuple[str, int]]
    recipe: Recipe

    @property
    def n_detected(self) -> int:
        """Number of detected pulse events."""
        return len(self.step_onsets)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return cast(dict[str, Any], to_jsonable(self))


# ── Analysis results (three levels) ──────────────────────────────────────────


@dataclass
class StepResult:
    """Result for one detected pulse event — the finest-grained output.

    Time is relative to the step onset (``time[0] == 0``). ``mass_corrected``
    has the pre-pulse baseline subtracted so ``mass_corrected[0] ≈ 0``.

    Parameters
    ----------
    step_name:
        Name of the precursor that was pulsed (e.g., ``"TMA"``).
    mass_effect:
        Expected direction from the Recipe (``"gain"``, ``"loss"``, ``"any"``).
    step_index:
        0-based position of this step within its SubCycle's ``steps`` list.
    sub_cycle_index:
        0-based index of the SubCycle in ``recipe.sub_cycles``.
    sub_cycle_run:
        0-based repeat number of this SubCycle within the outer cycle.
    outer_cycle:
        0-based outer repeat index of the Recipe.
    time:
        Time array in seconds, relative to this step's onset.
    mass_raw:
        Raw mass values in ng/cm² (not baseline-corrected).
    mass_corrected:
        Baseline-corrected mass in ng/cm².
    mass_change:
        Net mass change during this step in ng/cm². Negative for loss steps.
    """

    step_name: str
    mass_effect: Literal["gain", "loss", "any"]
    step_index: int
    sub_cycle_index: int
    sub_cycle_run: int
    outer_cycle: int
    time: NDArray[np.float64]
    mass_raw: NDArray[np.float64]
    mass_corrected: NDArray[np.float64]
    mass_change: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class SubCycleRun:
    """One execution of a SubCycle — a group of consecutive StepResults.

    For a super cycle ``[(A+B)*n + (A+C)*m] * l``, each outer ``Cycle``
    contains ``n`` SubCycleRun objects for the first SubCycle and ``m`` for
    the second.

    Parameters
    ----------
    sub_cycle_index:
        0-based index of the SubCycle in ``recipe.sub_cycles``.
    run_number:
        0-based repeat counter within the outer Cycle.
    outer_cycle:
        0-based outer cycle index.
    steps:
        Ordered list of StepResults for each pulse in this SubCycle run.
    """

    sub_cycle_index: int
    run_number: int
    outer_cycle: int
    steps: list[StepResult]

    @property
    def total_mass_change(self) -> float:
        """Net mass change across all steps in this SubCycle run (ng/cm²)."""
        return sum(s.mass_change for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class Cycle:
    """One outer repeat of the Recipe — what most ALD researchers call 'one cycle'.

    For simple A+B, each Cycle contains one SubCycleRun.
    For super cycles, each Cycle contains ``n + m`` SubCycleRuns.

    Parameters
    ----------
    cycle_number:
        0-based outer repeat index.
    sub_cycle_runs:
        All SubCycleRun objects belonging to this outer repeat, in order.
    """

    cycle_number: int
    sub_cycle_runs: list[SubCycleRun]

    @property
    def net_mass_change(self) -> float:
        """Net mass change across all sub-cycles in this outer cycle (ng/cm²)."""
        return sum(scr.total_mass_change for scr in self.sub_cycle_runs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class CycleCollection:
    """All analysis results, queryable at three levels of granularity.

    Iteration over a ``CycleCollection`` yields :class:`Cycle` objects (outer
    repeats). Use :meth:`sub_cycle_runs` and :meth:`steps` for finer access.

    Parameters
    ----------
    cycles:
        Ordered list of outer-cycle results.
    recipe:
        The Recipe used for this analysis, for reference.
    """

    cycles: list[Cycle]
    recipe: Recipe

    # ── Iteration and length ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.cycles)

    def __iter__(self) -> Iterator[Cycle]:
        return iter(self.cycles)

    # ── Multi-level queries ───────────────────────────────────────────────────

    def sub_cycle_runs(self, index: int) -> list[SubCycleRun]:
        """All runs of ``sub_cycle[index]`` across all outer cycles.

        Parameters
        ----------
        index:
            0-based index into ``recipe.sub_cycles``.

        Returns
        -------
        list[SubCycleRun]
            One entry per outer cycle (or per repeat for inner repeats).
        """
        return [
            scr
            for c in self.cycles
            for scr in c.sub_cycle_runs
            if scr.sub_cycle_index == index
        ]

    def steps(
        self,
        name: str | None = None,
        sub_cycle: int | None = None,
    ) -> list[StepResult]:
        """Flat list of all StepResults, with optional filters.

        Parameters
        ----------
        name:
            If given, return only steps whose ``step_name`` matches.
        sub_cycle:
            If given, return only steps from ``sub_cycle[sub_cycle]``.

        Returns
        -------
        list[StepResult]
            Chronologically ordered step results matching the filters.
        """
        all_steps: list[StepResult] = [
            s
            for c in self.cycles
            for scr in c.sub_cycle_runs
            for s in scr.steps
        ]
        if name is not None:
            all_steps = [s for s in all_steps if s.step_name == name]
        if sub_cycle is not None:
            all_steps = [s for s in all_steps if s.sub_cycle_index == sub_cycle]
        return all_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


# ── Kinetics results ──────────────────────────────────────────────────────────


@dataclass
class LangmuirResult:
    """Result returned by :func:`~qcm_pak.kinetics.fit_langmuir`.

    The ``model`` field records which model was *actually* fitted — it may be
    ``"mono"`` even when ``"bi"`` was requested, if BIC analysis detected a
    degenerate two-site fit.

    Parameters
    ----------
    step_name:
        The precursor name this fit applies to.
    model:
        Which model was used: ``"mono"`` or ``"bi"``.
    r_squared:
        Coefficient of determination (R²) for the fit.
    k, theta_max:
        Monomodal parameters. Present when ``model == "mono"``.
    k1, k2, theta1, theta2:
        Bimodal parameters. Present when ``model == "bi"``.
    covariance:
        Parameter covariance matrix from the fit, if available.

    Notes
    -----
    Monomodal model: θ(t) = θ_max · (1 − exp(−k · t))

    Bimodal model:   θ(t) = θ₁ · (1 − exp(−k₁ · t)) + θ₂ · (1 − exp(−k₂ · t))
    """

    step_name: str
    model: Literal["mono", "bi"]
    r_squared: float
    k: float | None = None
    theta_max: float | None = None
    k1: float | None = None
    k2: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    covariance: NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


@dataclass
class EtchResult:
    """Result returned by :func:`~qcm_pak.kinetics.fit_etch`.

    Fits etch kinetics to ``"loss"`` steps. The fit is performed on the
    absolute value of mass change, so ``etch_max`` and ``rate`` are always
    positive; the negative direction is implied by the ``"loss"`` mass_effect.

    Parameters
    ----------
    step_name:
        The precursor name this fit applies to.
    model:
        Which model was used: ``"saturating"`` or ``"linear"``.
    r_squared:
        Coefficient of determination (R²).
    k:
        Rate constant in s⁻¹ (saturating model only).
    etch_max:
        Maximum etch per step in ng/cm² (positive; saturating model only).
    rate:
        Etch rate in ng/cm²/s (linear model only).
    covariance:
        Parameter covariance matrix, if available.

    Notes
    -----
    Saturating model: E(t) = E_max · (1 − exp(−k · t))  [self-limiting ALE]

    Linear model:     E(t) = rate · t                    [non-saturating regime]
    """

    step_name: str
    model: Literal["saturating", "linear"]
    r_squared: float
    k: float | None = None
    etch_max: float | None = None
    rate: float | None = None
    covariance: NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become lists)."""
        return cast(dict[str, Any], to_jsonable(self))


# ── Top-level result container ────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Container returned by :func:`~qcm_pak.pipeline.run_analysis`.

    All intermediate results are accessible as plain Python objects for
    downstream computation. Call :meth:`save` to write files to disk, or
    :meth:`to_bytes` / :meth:`to_dict` for in-memory output (e.g. behind an
    HTTP service handling concurrent requests).

    Parameters
    ----------
    cycles:
        All extracted cycle data at three levels of granularity.
    mass_data:
        The full Sauerbrey-converted mass time series.
    cycle_index:
        Detected pulse onset positions (intermediate result from detection).
    params:
        The :class:`~qcm_pak.parameters.ALDParameters` used for this run.
    """

    cycles: CycleCollection
    mass_data: MassDataset
    cycle_index: CycleIndex
    params: ALDParameters

    def save(self, output_dir: str | Path) -> None:
        """Write analysis outputs to ``output_dir``.

        Creates the following files (``output_dir`` is created if absent):

        ``data/cycle_data_{step_name}.tsv``
            One file per unique step name. Each column is one occurrence of
            that step (in chronological order); rows are time points within
            the extraction window.

        ``figures/full_trace.png``
            Mass vs. time for the full experiment with detected pulse regions
            highlighted.

        ``figures/derivative_analysis.png``
            Smoothed first derivative of the mass signal — diagnostic for
            pulse detection quality.

        Parameters
        ----------
        output_dir:
            Directory to write outputs into. Created if it does not exist.
        """
        from qcm_pak import visualization

        out = Path(output_dir)
        data_dir = out / "data"
        fig_dir = out / "figures"
        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Write per-step TSV files
        for step_name in self.cycles.recipe.step_names():
            step_results = self.cycles.steps(name=step_name)
            if not step_results:
                continue
            path = data_dir / f"cycle_data_{step_name}.tsv"
            path.write_bytes(_step_tsv_bytes(step_results))

        # Write diagnostic figures
        trace_fig = visualization.plot_trace(self.mass_data, self.cycle_index)
        trace_fig.savefig(fig_dir / "full_trace.png", dpi=150, bbox_inches="tight")

        deriv_fig = visualization.plot_derivative(self.mass_data)
        deriv_fig.savefig(
            fig_dir / "derivative_analysis.png", dpi=150, bbox_inches="tight"
        )

    def to_bytes(self) -> dict[str, bytes]:
        """Render all analysis outputs in memory, without touching disk.

        Mirrors :meth:`save`'s layout — ``data/cycle_data_{step_name}.tsv``
        and ``figures/*.png`` — as a flat mapping of relative path to file
        bytes. Use this instead of :meth:`save` when running behind a
        service that handles concurrent requests and shouldn't write to a
        shared filesystem path per request.

        Returns
        -------
        dict[str, bytes]
            Relative output path (e.g. ``"data/cycle_data_TMA.tsv"``) mapped
            to its file contents.
        """
        from qcm_pak import visualization

        out: dict[str, bytes] = {}
        for step_name in self.cycles.recipe.step_names():
            step_results = self.cycles.steps(name=step_name)
            if not step_results:
                continue
            out[f"data/cycle_data_{step_name}.tsv"] = _step_tsv_bytes(step_results)

        trace_fig = visualization.plot_trace(self.mass_data, self.cycle_index)
        out["figures/full_trace.png"] = visualization.fig_to_png_bytes(trace_fig)

        deriv_fig = visualization.plot_derivative(self.mass_data)
        out["figures/derivative_analysis.png"] = visualization.fig_to_png_bytes(
            deriv_fig
        )

        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize cycles, mass data, detection index, and params to a
        JSON-compatible dict (arrays become lists). Figures are not
        included — use :meth:`to_bytes` for those.
        """
        return cast(dict[str, Any], to_jsonable(self))


def _step_tsv_bytes(steps: list[StepResult]) -> bytes:
    """Render a collection of StepResults as TSV bytes.

    Each column is one step occurrence; rows are time-point samples aligned
    to the step onset (time[0] = 0). The header row lists each occurrence as
    ``cycle_{outer}_sc{sub_cycle}_run{run}``.
    """
    import pandas as pd

    # Align on a common time axis using the shortest window
    min_len = min(len(s.time) for s in steps)
    ref_time = steps[0].time[:min_len]

    data: dict[str, NDArray[np.float64]] = {"time_s": ref_time}
    for s in steps:
        col = f"cycle_{s.outer_cycle}_sc{s.sub_cycle_index}_run{s.sub_cycle_run}"
        data[col] = s.mass_corrected[:min_len]

    tsv_text = cast(str, pd.DataFrame(data).to_csv(sep="\t", index=False))
    return tsv_text.encode("utf-8")
