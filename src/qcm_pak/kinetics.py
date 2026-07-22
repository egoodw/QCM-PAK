"""
Langmuir adsorption and etch kinetics fitting.

Two public functions:

:func:`fit_langmuir`
    Fit Langmuir adsorption kinetics to ALD gain steps.
    Monomodal: θ(t) = θ_max · (1 − exp(−k · t))
    Bimodal:   θ(t) = θ₁ · (1 − exp(−k₁ · t)) + θ₂ · (1 − exp(−k₂ · t))

:func:`fit_etch`
    Fit etch kinetics to ALE loss steps.
    Saturating: E(t) = E_max · (1 − exp(−k · t))
    Linear:     E(t) = rate · t

Both use a two-stage lmfit fitting strategy:
  1. ``differential_evolution`` global search for robust initial parameters.
  2. Levenberg-Marquardt local polish for precise covariance estimation.

Bimodal degeneracy handling in ``fit_langmuir``:
  If BIC prefers the monomodal fit, or if |ρ(k₁, k₂)| > 0.95 (indicating
  the two rate constants are collinear), a :class:`UserWarning` is issued and
  the function returns a monomodal :class:`~qcm_pak._types.LangmuirResult`.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qcm_pak._types import CycleCollection, EtchResult, LangmuirResult
from qcm_pak.exceptions import ConvergenceError

# ── Langmuir fitting ──────────────────────────────────────────────────────────


def fit_langmuir(
    cycles: CycleCollection,
    step: str,
    sub_cycle: int | None = None,
    model: Literal["mono", "bi"] = "mono",
    k_bounds: tuple[float, float] = (0.001, 10.0),
    theta_bounds: tuple[float, float] = (0.0, 300.0),
    force: bool = False,
) -> LangmuirResult:
    """Fit Langmuir adsorption kinetics to a named gain step.

    The fit is performed on the *average* baseline-corrected mass-change value
    per step occurrence vs. pulse exposure time.

    Parameters
    ----------
    cycles:
        Extracted cycle data from :func:`~qcm_pak.pipeline.run_analysis`.
    step:
        Name of the PulseStep to fit (e.g., ``"TMA"``).
    sub_cycle:
        If given, restrict the fit to steps from ``sub_cycle[sub_cycle]`` only.
        Use this when a step name appears in multiple SubCycles.
    model:
        ``"mono"`` for single-site Langmuir; ``"bi"`` for two-site. If the
        bimodal fit is degenerate (see module docstring), falls back to
        ``"mono"`` with a :class:`UserWarning`.
    k_bounds:
        ``(min, max)`` bounds for rate constant(s) k in s⁻¹.
    theta_bounds:
        ``(min, max)`` bounds for coverage parameter(s) θ in ng/cm².
    force:
        When ``True`` and ``model="bi"``, bypass the BIC evidence gate so the
        bimodal result is always returned as long as the fit converged and the
        two rate constants are not collinear (|ρ(k₁,k₂)| < 0.95). Useful when
        you have prior physical knowledge that two sites exist even if the data
        do not statistically prefer the extra parameters. A :class:`UserWarning`
        is still issued when the BIC favours the monomodal fit.

    Returns
    -------
    LangmuirResult
        Fit parameters. ``result.model`` may be ``"mono"`` even if ``"bi"``
        was requested (auto-fallback on collinearity even with ``force=True``).

    Raises
    ------
    ConvergenceError
        If the monomodal fit fails to converge.
    ValueError
        If no steps named ``step`` exist in ``cycles``.
    """
    step_results = cycles.steps(name=step, sub_cycle=sub_cycle)
    if not step_results:
        raise ValueError(
            f"No step named '{step}' found in CycleCollection"
            + (f" for sub_cycle={sub_cycle}" if sub_cycle is not None else "")
        )

    # Build (pulse_time, mass_change) dataset: one point per step occurrence
    t_vals, theta_vals = _langmuir_dataset(step_results)

    if model == "bi":
        return _fit_langmuir_bi(
            step, t_vals, theta_vals, k_bounds, theta_bounds, force=force
        )
    return _fit_langmuir_mono(step, t_vals, theta_vals, k_bounds, theta_bounds)


def _langmuir_dataset(
    steps: list,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build an averaged within-pulse coverage trace from a list of StepResults.

    Each StepResult contains a time array (0 → step end) and a baseline-corrected
    mass array. We align all occurrences on the shortest common window and average
    the mass traces, then fit the Langmuir equation to this ensemble average.

    This is the physically meaningful approach for fixed-pulse ALD experiments:
    the averaged trace shows how coverage builds during a single pulse, and the
    rate constant reflects how quickly the surface saturates.

    Returns
    -------
    t:
        Common time axis in seconds (starts at 0, length = shortest step window).
    theta:
        Ensemble-averaged baseline-corrected mass in ng/cm² at each time point.
    """
    min_len = min(len(s.mass_corrected) for s in steps)
    t = steps[0].time[:min_len].astype(np.float64)
    theta = np.mean(
        np.vstack([np.abs(s.mass_corrected[:min_len]) for s in steps]),
        axis=0,
    )
    return t, theta


def _fit_langmuir_mono(
    step_name: str,
    t: NDArray[np.float64],
    theta: NDArray[np.float64],
    k_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
) -> LangmuirResult:
    """Two-stage lmfit monomodal Langmuir fit."""
    try:
        import lmfit
    except ImportError as exc:
        raise ImportError(
            "lmfit is required for kinetics fitting. "
            "Install it with: pip install lmfit"
        ) from exc

    def _model(params: lmfit.Parameters, t: NDArray[np.float64]) -> NDArray[np.float64]:
        k = params["k"].value
        theta_max = params["theta_max"].value
        return theta_max * (1.0 - np.exp(-k * t))

    def _residual(params: lmfit.Parameters) -> NDArray[np.float64]:
        return _model(params, t) - theta

    params = lmfit.Parameters()
    params.add("k", value=1.0, min=k_bounds[0], max=k_bounds[1])
    tmax = float(np.max(theta))
    params.add("theta_max", value=tmax, min=theta_bounds[0], max=theta_bounds[1])

    # Stage 1: global search
    result_de = lmfit.minimize(
        _residual, params, method="differential_evolution", nan_policy="omit"
    )
    # Stage 2: local polish
    result = lmfit.minimize(
        _residual, result_de.params, method="leastsq", nan_policy="omit"
    )

    if not result.success and not result_de.success:
        raise ConvergenceError(
            f"Monomodal Langmuir fit failed to converge for step '{step_name}'"
        )

    best = result.params if result.success else result_de.params
    r_sq = _r_squared(theta, _model(best, t))

    cov: NDArray[np.float64] | None = None
    if result.covar is not None:
        cov = np.array(result.covar)

    return LangmuirResult(
        step_name=step_name,
        model="mono",
        r_squared=r_sq,
        k=float(best["k"].value),
        theta_max=float(best["theta_max"].value),
        covariance=cov,
    )


def _fit_langmuir_bi(
    step_name: str,
    t: NDArray[np.float64],
    theta: NDArray[np.float64],
    k_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
    force: bool = False,
) -> LangmuirResult:
    """Two-stage lmfit bimodal Langmuir fit with BIC degeneracy check."""
    try:
        import lmfit
    except ImportError as exc:
        raise ImportError(
            "lmfit is required for kinetics fitting. "
            "Install it with: pip install lmfit"
        ) from exc

    def _model_bi(
        params: lmfit.Parameters, t: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        k1 = params["k1"].value
        k2 = params["k2"].value
        th1 = params["theta1"].value
        th2 = params["theta2"].value
        return th1 * (1.0 - np.exp(-k1 * t)) + th2 * (1.0 - np.exp(-k2 * t))

    def _residual(params: lmfit.Parameters) -> NDArray[np.float64]:
        return _model_bi(params, t) - theta

    params = lmfit.Parameters()
    params.add("k1", value=0.1, min=k_bounds[0], max=k_bounds[1])
    params.add("k2", value=1.0, min=k_bounds[0], max=k_bounds[1])
    half_max = float(np.max(theta)) * 0.5
    params.add("theta1", value=half_max, min=theta_bounds[0], max=theta_bounds[1])
    params.add("theta2", value=half_max, min=theta_bounds[0], max=theta_bounds[1])

    result_de = lmfit.minimize(
        _residual, params, method="differential_evolution", nan_policy="omit"
    )
    result_bi = lmfit.minimize(
        _residual, result_de.params, method="leastsq", nan_policy="omit"
    )

    best_bi = result_bi.params if result_bi.success else result_de.params
    rss_bi = float(np.sum((_model_bi(best_bi, t) - theta) ** 2))
    bic_bi = _bic(rss_bi, len(t), 4)

    # Fit monomodal for comparison
    mono_result = _fit_langmuir_mono(step_name, t, theta, k_bounds, theta_bounds)
    rss_mono = float(np.sum(
        (mono_result.theta_max * (1.0 - np.exp(-mono_result.k * t)) - theta) ** 2  # type: ignore[operator]
    ))
    bic_mono = _bic(rss_mono, len(t), 2)

    # Collinearity check: always fall back when k1≈k2 (parameters non-identifiable).
    # BIC gate: when force=False, also fall back if the data don't statistically
    # prefer bimodal; when force=True, skip BIC but still warn.
    collinear = _is_bimodal_degenerate(result_bi)
    bic_prefers_mono = bic_mono <= bic_bi + 2.0
    degenerate = collinear or (not force and bic_prefers_mono)

    if degenerate:
        reason = (
            "collinear rate constants" if collinear
            else f"BIC_mono={bic_mono:.1f} ≤ BIC_bi={bic_bi:.1f}+2"
        )
        resolution = (
            "Returning monomodal fit." if not force or collinear
            else "force=True: returning bimodal anyway."
        )
        warnings.warn(
            f"Bimodal Langmuir fit for '{step_name}' is degenerate ({reason}). "
            + resolution,
            UserWarning,
            stacklevel=3,
        )
        if not force or collinear:
            return mono_result

    r_sq = _r_squared(theta, _model_bi(best_bi, t))
    cov: NDArray[np.float64] | None = None
    if result_bi.covar is not None:
        cov = np.array(result_bi.covar)

    return LangmuirResult(
        step_name=step_name,
        model="bi",
        r_squared=r_sq,
        k1=float(best_bi["k1"].value),
        k2=float(best_bi["k2"].value),
        theta1=float(best_bi["theta1"].value),
        theta2=float(best_bi["theta2"].value),
        covariance=cov,
    )


# ── Etch fitting ──────────────────────────────────────────────────────────────


def fit_etch(
    cycles: CycleCollection,
    step: str,
    sub_cycle: int | None = None,
    model: Literal["saturating", "linear"] = "saturating",
    k_bounds: tuple[float, float] = (0.001, 10.0),
    etch_bounds: tuple[float, float] = (0.0, 10.0),
) -> EtchResult:
    """Fit etch kinetics to a named ALE loss step.

    The fit is performed on the absolute value of mass change per occurrence.

    Parameters
    ----------
    cycles:
        Extracted cycle data.
    step:
        Name of the PulseStep with ``mass_effect='loss'`` to fit.
    sub_cycle:
        If given, restrict to steps from ``sub_cycle[sub_cycle]`` only.
    model:
        ``"saturating"`` for self-limiting ALE; ``"linear"`` for a constant
        etch rate (non-saturating regime).
    k_bounds:
        ``(min, max)`` bounds for rate constant k in s⁻¹ (saturating only).
    etch_bounds:
        ``(min, max)`` bounds for E_max in ng/cm² (saturating), or rate in
        ng/cm²/s (linear).

    Returns
    -------
    EtchResult
        Fit parameters. ``etch_max`` and ``rate`` are always positive.

    Raises
    ------
    ConvergenceError
        If the fit fails to converge.
    ValueError
        If no steps named ``step`` exist in ``cycles``.
    """
    step_results = cycles.steps(name=step, sub_cycle=sub_cycle)
    if not step_results:
        raise ValueError(
            f"No step named '{step}' found in CycleCollection"
            + (f" for sub_cycle={sub_cycle}" if sub_cycle is not None else "")
        )

    min_len = min(len(s.mass_corrected) for s in step_results)
    t_vals = step_results[0].time[:min_len].astype(np.float64)
    etch_vals = np.mean(
        np.vstack([np.abs(s.mass_corrected[:min_len]) for s in step_results]),
        axis=0,
    )

    if model == "linear":
        return _fit_etch_linear(step, t_vals, etch_vals, etch_bounds)
    return _fit_etch_saturating(step, t_vals, etch_vals, k_bounds, etch_bounds)


def _fit_etch_saturating(
    step_name: str,
    t: NDArray[np.float64],
    etch: NDArray[np.float64],
    k_bounds: tuple[float, float],
    etch_bounds: tuple[float, float],
) -> EtchResult:
    import lmfit

    def _model(params: lmfit.Parameters, t: NDArray[np.float64]) -> NDArray[np.float64]:
        k = params["k"].value
        e_max = params["etch_max"].value
        return e_max * (1.0 - np.exp(-k * t))

    def _residual(params: lmfit.Parameters) -> NDArray[np.float64]:
        return _model(params, t) - etch

    params = lmfit.Parameters()
    params.add("k", value=1.0, min=k_bounds[0], max=k_bounds[1])
    params.add(
        "etch_max", value=float(np.max(etch)), min=etch_bounds[0], max=etch_bounds[1]
    )

    result_de = lmfit.minimize(
        _residual, params, method="differential_evolution", nan_policy="omit"
    )
    result = lmfit.minimize(
        _residual, result_de.params, method="leastsq", nan_policy="omit"
    )

    if not result.success and not result_de.success:
        raise ConvergenceError(
            f"Saturating etch fit failed to converge for step '{step_name}'"
        )

    best = result.params if result.success else result_de.params
    r_sq = _r_squared(etch, _model(best, t))

    cov: NDArray[np.float64] | None = None
    if result.covar is not None:
        cov = np.array(result.covar)

    return EtchResult(
        step_name=step_name,
        model="saturating",
        r_squared=r_sq,
        k=float(best["k"].value),
        etch_max=float(best["etch_max"].value),
        covariance=cov,
    )


def _fit_etch_linear(
    step_name: str,
    t: NDArray[np.float64],
    etch: NDArray[np.float64],
    rate_bounds: tuple[float, float],
) -> EtchResult:
    import lmfit

    def _model(params: lmfit.Parameters, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return params["rate"].value * t

    def _residual(params: lmfit.Parameters) -> NDArray[np.float64]:
        return _model(params, t) - etch

    params = lmfit.Parameters()
    params.add("rate", value=float(np.mean(etch / np.where(t > 0, t, 1.0))),
               min=rate_bounds[0], max=rate_bounds[1])

    result_de = lmfit.minimize(
        _residual, params, method="differential_evolution", nan_policy="omit"
    )
    result = lmfit.minimize(
        _residual, result_de.params, method="leastsq", nan_policy="omit"
    )

    if not result.success and not result_de.success:
        raise ConvergenceError(
            f"Linear etch fit failed to converge for step '{step_name}'"
        )

    best = result.params if result.success else result_de.params
    r_sq = _r_squared(etch, _model(best, t))

    cov: NDArray[np.float64] | None = None
    if result.covar is not None:
        cov = np.array(result.covar)

    return EtchResult(
        step_name=step_name,
        model="linear",
        r_squared=r_sq,
        rate=float(best["rate"].value),
        covariance=cov,
    )


# ── Shared utilities ──────────────────────────────────────────────────────────


def _r_squared(y_obs: NDArray[np.float64], y_fit: NDArray[np.float64]) -> float:
    """Coefficient of determination R²."""
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _bic(rss: float, n: int, n_params: int) -> float:
    """Bayesian Information Criterion: BIC = n·ln(RSS/n) + k·ln(n)."""
    if rss <= 0 or n <= 0:
        return -np.inf
    return n * np.log(rss / n) + n_params * np.log(n)


def _is_bimodal_degenerate(result: object) -> bool:
    """Check if the bimodal fit is degenerate due to collinear rate constants.

    Degeneracy is declared when the Pearson correlation |ρ(k₁, k₂)| > 0.95
    in the parameter covariance matrix, indicating the two rate constants
    are not independently identifiable from the data.
    """
    try:
        covar = getattr(result, "covar", None)
        if covar is None:
            return False
        cov_mat = np.array(covar)
        # Parameters are ordered: k1, k2, theta1, theta2
        if cov_mat.shape[0] < 2:
            return False
        var_k1 = cov_mat[0, 0]
        var_k2 = cov_mat[1, 1]
        cov_k1k2 = cov_mat[0, 1]
        if var_k1 <= 0 or var_k2 <= 0:
            return True
        rho = abs(cov_k1k2 / np.sqrt(var_k1 * var_k2))
        return rho > 0.95
    except Exception:
        return False
