"""
Pulse detection: find the onset index of every step in the Recipe.

Two algorithms are provided:

``pelt_guided`` (default)
    Uses the PELT changepoint detection algorithm (``ruptures`` library, L2
    cost model) to locate all mean-shift changepoints in the mass signal
    globally. The set of candidates is then filtered and assigned to the
    expected Recipe events using the recipe timeline and a ±``recipe_tolerance``
    fractional window. Changepoint sign (positive Δm = gain, negative = loss)
    is used when the PulseStep has ``mass_effect='gain'`` or ``'loss'`` to
    discard false detections.

``hybrid``
    Recipe-guided Savitzky-Golay derivative search. For each expected pulse
    window (derived from the recipe timeline and ``recipe_tolerance``), a
    smoothed first-derivative peak is located within the window. Falls back
    to the maximum-derivative sample when no peak exceeds the adaptive
    threshold. This matches the algorithm used in QCMPy 0.5.

Both algorithms return a :class:`~qcm_pak._types.CycleIndex` with exactly
``recipe.total_events`` detected onsets, or raise
:class:`~qcm_pak.exceptions.DetectionError`.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from qcm_pak._types import CycleIndex, MassDataset
from qcm_pak.exceptions import DetectionError
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import Recipe

# ── Public entry point ─────────────────────────────────────────────────────────


def detect_pulses(
    data: MassDataset,
    params: ALDParameters,
    det_params: DetectionParameters | None = None,
) -> CycleIndex:
    """Detect all pulse-step onsets described by ``params.recipe``.

    Parameters
    ----------
    data:
        Sauerbrey-converted mass time series.
    params:
        Experiment parameters (carries the Recipe).
    det_params:
        Detection algorithm configuration. Defaults are constructed if
        ``None``.

    Returns
    -------
    CycleIndex
        Contains exactly ``params.recipe.total_events`` detected onset
        positions, each labelled with the step name.

    Raises
    ------
    DetectionError
        If detection cannot satisfy the recipe's expected event count.
    """
    if det_params is None:
        det_params = DetectionParameters()

    if det_params.method == "pelt_guided":
        return _detect_pelt_guided(data, params, det_params)
    return _detect_hybrid(data, params, det_params)


# ── Recipe timeline helper ─────────────────────────────────────────────────────


class _EventSpec(NamedTuple):
    """One expected recipe event."""

    step_name: str
    mass_effect: str        # 'gain', 'loss', 'any'
    expected_idx: int       # sample index in MassDataset arrays
    sub_cycle_index: int
    sub_cycle_run: int
    step_index: int
    outer_cycle: int
    step_dur_samples: int   # step duration converted to samples


def _build_timeline(recipe: Recipe, dt: float) -> list[_EventSpec]:
    """Build the ordered list of expected pulse-event sample indices.

    Parameters
    ----------
    recipe:
        The ALD/ALE recipe.
    dt:
        Median sampling interval of the MassDataset in seconds.

    Returns
    -------
    list[_EventSpec]
        One entry per expected pulse, in chronological order.
    """
    events: list[_EventSpec] = []
    t = recipe.start_time

    for outer in range(recipe.repeats):
        for sc_idx, sub_cycle in enumerate(recipe.sub_cycles):
            for run in range(sub_cycle.repeats):
                for s_idx, step in enumerate(sub_cycle.steps):
                    expected_idx = int(round(t / dt))
                    events.append(
                        _EventSpec(
                            step_name=step.name,
                            mass_effect=step.mass_effect,
                            expected_idx=expected_idx,
                            sub_cycle_index=sc_idx,
                            sub_cycle_run=run,
                            step_index=s_idx,
                            outer_cycle=outer,
                            step_dur_samples=max(1, int(step.duration / dt)),
                        )
                    )
                    t += step.duration
    return events


# ── PELT-guided algorithm ──────────────────────────────────────────────────────


def _detect_pelt_guided(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
) -> CycleIndex:
    """PELT changepoint detection with recipe-guided assignment."""
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise ImportError(
            "ruptures is required for pelt_guided detection. "
            "Install it with: pip install ruptures"
        ) from exc

    mass = data.mass
    n = len(mass)
    dt = data.dt

    # Estimate penalty
    if det.penalty == "auto":
        penalty = _bic_penalty(n, _estimate_noise(mass))
    else:
        penalty = float(det.penalty)

    # Run PELT with L2 cost (detects mean shifts — exactly what ALD pulses are)
    algo = rpt.Pelt(model="l2", min_size=det.min_pulse_spacing).fit(mass)
    breakpoints = algo.predict(pen=penalty)
    # ruptures returns the first index of each new segment (exclusive end of
    # previous segment). That IS the onset sample — do not subtract 1.
    candidates = [bp for bp in breakpoints if bp < n]

    # Build expected timeline
    timeline = _build_timeline(params.recipe, dt)
    candidates.sort()
    onsets = _assign_candidates(timeline, candidates, mass, n, dt, det.recipe_tolerance)

    _check_count(onsets, params.recipe)
    return CycleIndex(step_onsets=onsets, recipe=params.recipe)


def _assign_candidates(
    timeline: list[_EventSpec],
    candidates: list[int],
    mass: NDArray[np.float64],
    n: int,
    dt: float,
    tolerance: float,
) -> list[tuple[str, int]]:
    """Optimal order-preserving match of PELT candidates to recipe events.

    Recipe events are strictly chronological, so this is a monotonic
    assignment problem: each event gets at most one candidate (or falls
    back to its recipe-predicted position), and assigned candidates must
    appear in the same order as their events. Matching each event
    independently to its nearest-in-window candidate breaks down once a
    systematic recipe-timing bias (e.g. a mis-estimated ``start_time``)
    widens the effective overlap between adjacent windows enough that a
    neighboring event's true changepoint sits closer than the event's own
    — silently stealing the wrong candidate and desyncing every event after
    it. The DP below picks the whole sequence jointly, so it is immune to
    that local ambiguity (same technique as sequence alignment).
    """
    k = len(timeline)
    m = len(candidates)
    fallback_cost = float(n)  # always worse than any real in-window match

    windows: list[tuple[int, int]] = []
    for event in timeline:
        tol_samples = max(1, int(event.step_dur_samples * tolerance))
        lo = max(0, event.expected_idx - tol_samples)
        hi = min(n - 1, event.expected_idx + tol_samples)
        windows.append((lo, hi))

    # dp[i][j]: min cost assigning the first i events using only
    # candidates[:j]. choice[i][j]: 0 = leave candidate j-1 unused,
    # 1 = assign candidate j-1 to event i-1, 2 = event i-1 falls back.
    dp = [[0.0] * (m + 1) for _ in range(k + 1)]
    choice = [[0] * (m + 1) for _ in range(k + 1)]
    for i in range(1, k + 1):
        dp[i][0] = dp[i - 1][0] + fallback_cost
        choice[i][0] = 2

    for i in range(1, k + 1):
        event = timeline[i - 1]
        lo, hi = windows[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            best_cost, best_choice = row[j - 1], 0

            fallback = prev_row[j] + fallback_cost
            if fallback < best_cost:
                best_cost, best_choice = fallback, 2

            c = candidates[j - 1]
            if lo <= c <= hi and _sign_compatible(mass, c, event.mass_effect, n):
                assign = prev_row[j - 1] + abs(c - event.expected_idx)
                if assign < best_cost:
                    best_cost, best_choice = assign, 1

            row[j] = best_cost
            choice[i][j] = best_choice

    onsets: list[tuple[str, int] | None] = [None] * k
    i, j = k, m
    while i > 0:
        step = choice[i][j]
        event = timeline[i - 1]
        if step == 1:
            onsets[i - 1] = (event.step_name, candidates[j - 1])
            i, j = i - 1, j - 1
        elif step == 2:
            warnings.warn(
                f"No PELT candidate found near expected onset for "
                f"'{event.step_name}' at ~{event.expected_idx * dt:.1f}s "
                f"(outer cycle {event.outer_cycle}). Using recipe-estimated position.",
                stacklevel=4,
            )
            onsets[i - 1] = (event.step_name, min(event.expected_idx, n - 1))
            i -= 1
        else:
            j -= 1

    return [o for o in onsets if o is not None]


def _sign_compatible(
    mass: NDArray[np.float64],
    idx: int,
    mass_effect: str,
    n: int,
) -> bool:
    """Return True if the mean mass change across ``idx`` matches ``mass_effect``.

    Compares the mean of [idx-window, idx) with the mean of [idx, idx+window)
    so the sign estimate is noise-averaged rather than based on a single sample
    pair (which can flip sign within a flat noisy plateau).
    """
    if mass_effect == "any":
        return True
    window = 20
    pre = mass[max(0, idx - window):idx]
    post = mass[idx:min(n, idx + window)]
    if pre.size == 0 or post.size == 0:
        return True  # not enough context — don't reject a valid candidate
    delta = float(np.mean(post)) - float(np.mean(pre))
    if mass_effect == "gain":
        return delta >= 0
    return delta <= 0  # 'loss'


def _estimate_noise(mass: NDArray[np.float64]) -> float:
    """Estimate the noise σ from the first-difference of the mass signal."""
    diffs = np.diff(mass)
    return float(np.median(np.abs(diffs)) / 0.6745)   # MAD-based σ estimate


def _bic_penalty(n: int, noise: float) -> float:
    """BIC-motivated PELT penalty: σ² · log(n)."""
    return max(noise**2 * np.log(n), 1.0)


# ── Hybrid (derivative-based) algorithm ───────────────────────────────────────


def _detect_hybrid(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
) -> CycleIndex:
    """Recipe-guided derivative search (port of QCMPy 0.5 CycleDetector)."""
    mass = data.mass
    n = len(mass)
    dt = data.dt

    # Smooth the mass signal
    win = min(det.smoothing_window, n // 4 * 2 + 1)   # must be odd and < n
    if win % 2 == 0:
        win += 1
    poly = min(det.savgol_polyorder, win - 1)
    smooth = savgol_filter(mass, window_length=win, polyorder=poly)
    deriv = np.gradient(smooth, dt)

    timeline = _build_timeline(params.recipe, dt)
    baseline_pts = max(1, int(det.baseline_window / dt))

    onsets: list[tuple[str, int]] = []

    for event in timeline:
        # Additive tolerance: ±(tol × step_duration) in samples.
        tol_samples = max(1, int(event.step_dur_samples * det.recipe_tolerance))
        lo = max(0, event.expected_idx - tol_samples)
        hi = min(n - 1, event.expected_idx + tol_samples)

        # Baseline window: go back one full step duration before the search
        # window so we sample genuinely quiet pre-pulse signal rather than
        # the tail of the preceding event's derivative spike.
        baseline_end = max(0, lo - event.step_dur_samples)
        baseline_start = max(0, baseline_end - baseline_pts)
        baseline_deriv = deriv[baseline_start:baseline_end]
        if len(baseline_deriv) > 2:
            threshold = (
                np.mean(np.abs(baseline_deriv))
                + det.adaptive_threshold_sigma * np.std(baseline_deriv)
            )
        else:
            threshold = 0.0

        segment = deriv[lo : hi + 1]
        if len(segment) == 0:
            onsets.append((event.step_name, min(event.expected_idx, n - 1)))
            continue

        # Select derivative direction based on mass_effect
        if event.mass_effect == "loss":
            scored = -segment
        else:
            scored = segment   # gain or any → look for positive derivative spike

        # Peak of the signed derivative is the onset estimate.
        # First-crossing is avoided: SG pre-ringing causes the derivative to
        # rise 20-25 samples before the actual step, so "first above threshold"
        # is systematically early. The argmax is unbiased and also degrades
        # more gracefully under low-SNR conditions.
        best_local = int(np.argmax(scored))
        if scored[best_local] < threshold:
            warnings.warn(
                f"Low-confidence onset for '{event.step_name}' at "
                f"~{event.expected_idx * dt:.1f}s "
                f"(outer cycle {event.outer_cycle}). "
                "Peak derivative is below the adaptive threshold; detection "
                "may be inaccurate. Consider using pelt_guided instead.",
                stacklevel=3,
            )

        onsets.append((event.step_name, lo + best_local))

    _check_count(onsets, params.recipe)
    return CycleIndex(step_onsets=onsets, recipe=params.recipe)


# ── Shared validation ─────────────────────────────────────────────────────────


def _check_count(
    onsets: list[tuple[str, int]],
    recipe: Recipe,
) -> None:
    """Raise DetectionError if the detected count doesn't match the recipe."""
    expected = recipe.total_events
    got = len(onsets)
    if got != expected:
        raise DetectionError(
            f"Expected {expected} pulse events from recipe but detected {got}. "
            "Try adjusting DetectionParameters.recipe_tolerance or penalty."
        )
