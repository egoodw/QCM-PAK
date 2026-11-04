"""
Fit residual traces: what each kinetics model leaves unexplained.

The ensemble fits (:func:`~qcm_pak.kinetics.fit_langmuir`,
:func:`~qcm_pak.kinetics.fit_etch`) fit one curve to the average of every
occurrence of a step. This module returns, for every occurrence, the data, that
ensemble curve and the difference between them, so a fit can be judged by eye
and by number.

For each occurrence (a :class:`ResidualTrace`)::

    time      s from that step's pulse onset, out to the end of the fitted window
    observed  |mass_corrected|, the quantity the fit targets (ng/cm²)
    fitted    the ensemble curve at ``time`` (ng/cm²)
    residual  observed - fitted (ng/cm²)
    rmse      sqrt(mean(residual²)) (ng/cm²)

The ensemble fit functions attach a :class:`FitResiduals` as
``result.residuals``. Per-cycle fits are untouched and carry none. Nothing
here changes the fitting: the curve is evaluated from the reported
parameters.

Aggregates are built by :func:`aggregate_traces` from the *stored* traces on a
common time grid (nothing is refitted or recomputed):

- ``FitResiduals.stats``: per-step mean, σ and n over its occurrences.
- :func:`summarize_residuals`: the same pooled across every step of a model.
- :func:`collect_residuals`: fit every step of a recipe for one registry model
  and return the per-step sets plus the cross-step summary.

All classes have ``to_dict()`` giving dicts of lists of floats (NaN becomes
``None``), ready for JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from qcm_pak.serialization import dataclass_to_dict

#: Most points the common residual grid may have (see :func:`aggregate_traces`).
MAX_GRID_POINTS = 5000

if TYPE_CHECKING:
    from qcm_pak._types import CycleCollection, EtchResult, LangmuirResult, StepResult


@dataclass
class ResidualTrace:
    """Observed, fitted and residual values for one occurrence of a step.

    Time is in seconds from this step's pulse onset and runs to the end of the
    window the ensemble fit used. ``observed`` is the quantity the fit targets,
    ``|mass_corrected|`` in ng/cm², and ``fitted`` is the *ensemble* curve (the
    one whose parameters are the reported result) evaluated on this
    occurrence's own time axis. ``residual = observed - fitted`` (ng/cm²).

    Parameters
    ----------
    outer_cycle, sub_cycle_index, sub_cycle_run, step_index:
        Which occurrence this is, as on :class:`~qcm_pak._types.StepResult`.
        Sub-cycle runs of a super cycle stay distinguishable by these.
    time, observed, fitted, residual:
        Equal-length arrays (lists once serialised).
    rmse:
        Root-mean-square of ``residual``, in ng/cm².
    """

    outer_cycle: int
    sub_cycle_index: int
    sub_cycle_run: int
    step_index: int
    time: NDArray[np.float64]
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    residual: NDArray[np.float64]
    rmse: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict (arrays become lists)."""
        return dataclass_to_dict(self)


@dataclass
class ResidualStats:
    """Mean, spread and count of residual traces on a common time grid.

    Built by aggregating stored :class:`ResidualTrace` objects. The grid
    starts at 0 (pulse onset) and its spacing is the finest sampling step among
    the traces; each trace is linearly interpolated onto it and contributes
    only where it has data, so longer traces are not cut to the shortest and
    ``n`` falls towards the end of the grid where fewer traces reach.

    Parameters
    ----------
    time:
        Common grid in seconds from pulse onset.
    mean:
        Mean residual at each grid point, ng/cm².
    std:
        Sample standard deviation (ddof=1), ng/cm²; ``None`` where ``n < 2``.
    n:
        Number of traces that have data at each grid point.
    """

    time: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: list[float | None]
    n: NDArray[np.int64]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict (arrays become lists)."""
        return dataclass_to_dict(self)


@dataclass
class FitResiduals:
    """Residual traces for one ensemble fit (one step, one model).

    Attached to ``LangmuirResult.residuals`` and ``EtchResult.residuals`` by
    the ensemble fit functions.

    Parameters
    ----------
    step_name:
        Step the fit applies to.
    model:
        The model actually fitted (``"mono"``, ``"bi"``, ``"saturating"`` or
        ``"linear"``).
    sub_cycle:
        The ``sub_cycle`` the fit was restricted to, or ``None``.
    traces:
        One :class:`ResidualTrace` per step occurrence, chronological.
    stats:
        Per-step mean/σ/n over ``traces``.
    rmse:
        RMSE over every residual point of every trace, ng/cm².
    """

    step_name: str
    model: str
    sub_cycle: int | None
    traces: list[ResidualTrace]
    stats: ResidualStats
    rmse: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict."""
        return dataclass_to_dict(self)


@dataclass
class StepResidualSummary:
    """Per-step line of a :class:`ResidualSummary`: label, RMSE and stats."""

    step_name: str
    sub_cycle: int | None
    n_traces: int
    rmse: float
    stats: ResidualStats

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict."""
        return dataclass_to_dict(self)


@dataclass
class ResidualSummary:
    """Cross-step residual aggregate for one model.

    ``stats`` pools every stored per-occurrence trace of every step onto one
    grid (see :class:`ResidualStats`), so its ``n`` counts traces.

    Parameters
    ----------
    model:
        Model id, e.g. ``"langmuir_mono"`` (or the fitted model name when
        built straight from results).
    rmse:
        RMSE over all pooled residual points, ng/cm².
    n_traces:
        Total number of traces pooled.
    stats:
        Pooled mean/σ/n.
    steps:
        One :class:`StepResidualSummary` per step.
    """

    model: str
    rmse: float
    n_traces: int
    stats: ResidualStats
    steps: list[StepResidualSummary]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict."""
        return dataclass_to_dict(self)


@dataclass
class ModelResiduals:
    """All residuals for one model across a recipe (one set per step).

    Returned by :func:`collect_residuals`.

    Parameters
    ----------
    model_id:
        Registry id, e.g. ``"langmuir_bi"``.
    fits:
        One :class:`FitResiduals` per step (per sub-cycle where a step name
        occurs in several sub-cycles).
    summary:
        Cross-step aggregate of ``fits``.
    """

    model_id: str
    fits: list[FitResiduals]
    summary: ResidualSummary

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict."""
        return dataclass_to_dict(self)


# ── Building traces from a fit ────────────────────────────────────────────────


def model_curve(
    result: LangmuirResult | EtchResult, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the fitted model of ``result`` at times ``t`` (ng/cm²).

    Raises
    ------
    ValueError
        If ``result`` lacks the parameters its ``model`` needs.
    """
    model = result.model
    if model == "mono" or model == "saturating":
        amp = result.theta_max if model == "mono" else result.etch_max  # type: ignore[union-attr]
        k = result.k
        if amp is None or k is None:
            raise ValueError(f"{model!r} result is missing k or its amplitude.")
        return np.asarray(amp * (1.0 - np.exp(-k * t)), dtype=np.float64)
    if model == "bi":
        r = result
        vals = (r.k1, r.k2, r.theta1, r.theta2)  # type: ignore[union-attr]
        if any(v is None for v in vals):
            raise ValueError("'bi' result is missing k1, k2, theta1 or theta2.")
        k1, k2, th1, th2 = vals
        return np.asarray(
            th1 * (1.0 - np.exp(-k1 * t)) + th2 * (1.0 - np.exp(-k2 * t)),  # type: ignore[operator]
            dtype=np.float64,
        )
    if model == "linear":
        if result.rate is None:  # type: ignore[union-attr]
            raise ValueError("'linear' result is missing rate.")
        return np.asarray(result.rate * t, dtype=np.float64)  # type: ignore[union-attr]
    raise ValueError(f"Unknown model {model!r}.")


def residuals_for_fit(
    result: LangmuirResult | EtchResult,
    steps: list[StepResult],
    sub_cycle: int | None = None,
) -> FitResiduals:
    """Residual traces of an ensemble ``result`` against ``steps``.

    ``steps`` are the occurrences the fit used. Each trace is cut to the
    window the fit used (the shortest occurrence), on its own time axis.
    Called by the ensemble fit functions; call it directly to recompute traces
    for a result you already hold.
    """
    min_len = min(len(s.mass_corrected) for s in steps)
    traces: list[ResidualTrace] = []
    for s in steps:
        t = np.asarray(s.time[:min_len], dtype=np.float64)
        t = t - t[0]
        observed = np.abs(np.asarray(s.mass_corrected[:min_len], dtype=np.float64))
        fitted = model_curve(result, t)
        resid = observed - fitted
        traces.append(
            ResidualTrace(
                outer_cycle=s.outer_cycle,
                sub_cycle_index=s.sub_cycle_index,
                sub_cycle_run=s.sub_cycle_run,
                step_index=s.step_index,
                time=t,
                observed=observed,
                fitted=fitted,
                residual=resid,
                rmse=_rmse([resid]),
            )
        )
    return FitResiduals(
        step_name=result.step_name,
        model=result.model,
        sub_cycle=sub_cycle,
        traces=traces,
        stats=aggregate_traces(traces),
        rmse=_rmse([tr.residual for tr in traces]),
    )


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_traces(traces: list[ResidualTrace]) -> ResidualStats:
    """Mean, σ and n of stored residual traces on a common time grid.

    No refitting: the ``residual`` arrays are used as stored. The grid runs
    from 0 to the longest trace in steps of the finest sampling interval among
    the traces, but never more than ``MAX_GRID_POINTS`` points: a log whose
    time axis has a near-zero gap (repeated clock stamps nudged apart by
    microseconds, say) would otherwise ask for tens of millions of points.
    Each trace is linearly interpolated onto the grid and counts only where
    it has data (no truncation to the shortest trace).

    Raises
    ------
    ValueError
        If ``traces`` is empty or a trace has fewer than two points.
    """
    if not traces:
        raise ValueError("Cannot aggregate zero residual traces.")
    if any(len(tr.time) < 2 for tr in traces):
        raise ValueError("Residual traces need at least two points to aggregate.")
    dt = min(float(np.min(np.diff(tr.time))) for tr in traces)
    if not dt > 0:
        raise ValueError("Residual trace time axes must be strictly increasing.")
    t_end = max(float(tr.time[-1] - tr.time[0]) for tr in traces)
    if t_end / dt + 1.0 > MAX_GRID_POINTS:
        dt = t_end / (MAX_GRID_POINTS - 1)
    n_pts = int(np.floor(t_end / dt + 1e-9)) + 1
    grid = np.arange(n_pts, dtype=np.float64) * dt

    total = np.zeros(n_pts)
    total_sq = np.zeros(n_pts)
    count = np.zeros(n_pts, dtype=np.int64)
    tol = dt * 1e-6
    for tr in traces:
        rel = tr.time - tr.time[0]
        have = grid <= rel[-1] + tol
        vals = np.interp(grid[have], rel, tr.residual)
        total[have] += vals
        total_sq[have] += vals**2
        count[have] += 1

    mean = total / count
    enough = count >= 2
    var = (total_sq - count * mean**2) / np.where(enough, count - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    std: list[float | None] = [float(v) if ok else None for v, ok in zip(sd, enough)]
    return ResidualStats(time=grid, mean=mean, std=std, n=count)


def summarize_residuals(
    fits: list[FitResiduals] | list[LangmuirResult | EtchResult],
    model: str | None = None,
) -> ResidualSummary:
    """Cross-step aggregate of the residuals stored on ``fits``.

    Takes :class:`FitResiduals` or ensemble results carrying ``.residuals``,
    pools every stored per-occurrence trace and aggregates it on a common grid
    with :func:`aggregate_traces`. Nothing is refitted.

    Parameters
    ----------
    fits:
        One entry per step (A, B, …) of one model.
    model:
        Label for the summary; defaults to the fitted model name.

    Raises
    ------
    ValueError
        If ``fits`` is empty, or a result has no residuals (per-cycle fits
        carry none).
    """
    sets: list[FitResiduals] = []
    for f in fits:
        fr = f if isinstance(f, FitResiduals) else f.residuals
        if fr is None:
            raise ValueError(
                "Result has no residuals; use an ensemble fit "
                "(fit_langmuir / fit_etch), not a per-cycle fit."
            )
        sets.append(fr)
    if not sets:
        raise ValueError("Need at least one fit to summarize.")
    pooled = [tr for fr in sets for tr in fr.traces]
    return ResidualSummary(
        model=model if model is not None else sets[0].model,
        rmse=_rmse([tr.residual for tr in pooled]),
        n_traces=len(pooled),
        stats=aggregate_traces(pooled),
        steps=[
            StepResidualSummary(
                step_name=fr.step_name,
                sub_cycle=fr.sub_cycle,
                n_traces=len(fr.traces),
                rmse=fr.rmse,
                stats=fr.stats,
            )
            for fr in sets
        ],
    )


def collect_residuals(
    cycles: CycleCollection, model_id: str, **fit_kwargs: Any  # noqa: ANN401
) -> ModelResiduals:
    """Fit every matching step of a recipe with one model; gather residuals.

    Uses the model registry (``"langmuir_mono"``, ``"langmuir_bi"``,
    ``"etch_saturating"``, ``"etch_linear"``): each step whose ``mass_effect``
    matches the model is fitted with the ordinary ensemble fit, so fit results
    are exactly those of calling it yourself. A step name that occurs in
    several sub-cycles is fitted once per sub-cycle (``sub_cycle`` set on its
    :class:`FitResiduals`); otherwise once over all its occurrences. Sub-cycle
    runs stay distinguishable through each trace's ``sub_cycle_run``.

    Parameters
    ----------
    cycles:
        Extracted cycle data.
    model_id:
        Key of :data:`~qcm_pak.models.MODEL_REGISTRY`.
    **fit_kwargs:
        Passed to the fit function (bounds, ``force``).

    Raises
    ------
    KeyError
        If ``model_id`` is not registered.
    ValueError
        If the recipe has no step the model applies to.
    """
    from qcm_pak.models import MODEL_REGISTRY

    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_id!r}. Choose from: {', '.join(MODEL_REGISTRY)}."
        )
    spec = MODEL_REGISTRY[model_id]
    seen: dict[str, set[int]] = {}
    order: list[str] = []
    for s in cycles.steps():
        if s.mass_effect != spec.mass_effect:
            continue
        if s.step_name not in seen:
            seen[s.step_name] = set()
            order.append(s.step_name)
        seen[s.step_name].add(s.sub_cycle_index)

    fits: list[FitResiduals] = []
    for name in order:
        subs = sorted(seen[name])
        targets: list[int | None] = [None] if len(subs) == 1 else list(subs)
        for sc in targets:
            res = spec.fit(cycles, name, sub_cycle=sc, **fit_kwargs)
            assert res.residuals is not None  # noqa: S101 (ensemble fits set it)
            fits.append(res.residuals)
    if not fits:
        raise ValueError(
            f"No '{spec.mass_effect}' steps to fit with model {model_id!r}."
        )
    return ModelResiduals(
        model_id=model_id,
        fits=fits,
        summary=summarize_residuals(fits, model=model_id),
    )


def _rmse(arrays: list[NDArray[np.float64]]) -> float:
    """Root-mean-square over all values of ``arrays`` (NaN if there are none)."""
    flat = np.concatenate([a.ravel() for a in arrays]) if arrays else np.empty(0)
    if flat.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(flat**2)))
