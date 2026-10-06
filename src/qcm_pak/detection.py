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
    threshold. The peak marks the steepest part of the rise, so each onset is
    then refined against the raw mass, back to the last sample on the
    pre-pulse baseline (``refinement_window``). This matches the algorithm
    used in QCMPy 0.5, including its onset refinement.

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
        If detection cannot satisfy the recipe's expected event count, or if
        ``data.time`` isn't sorted in non-decreasing order (required for the
        time-based sample lookups detection relies on).
    """
    if det_params is None:
        det_params = DetectionParameters()

    if len(data.time) > 1 and np.any(np.diff(data.time) < 0):
        raise DetectionError(
            "MassDataset.time is not sorted in non-decreasing order. Pulse "
            "detection locates onsets by searching directly in the time "
            "array, which requires it to be sorted. Check that the correct "
            "time column was selected during import."
        )

    if det_params.method == "pelt_guided":
        return _detect_pelt_guided(data, params, det_params)
    return _detect_hybrid(data, params, det_params)


# ── Recipe timeline helper ─────────────────────────────────────────────────────


class _EventSpec(NamedTuple):
    """One expected recipe event."""

    step_name: str
    mass_effect: str        # 'gain', 'loss', 'any'
    expected_time: float    # recipe-predicted onset time, in seconds
    expected_idx: int       # nearest sample index for expected_time
    sub_cycle_index: int
    sub_cycle_run: int
    step_index: int
    outer_cycle: int
    step_duration: float    # step duration in seconds


def _build_timeline(recipe: Recipe, data: MassDataset) -> list[_EventSpec]:
    """Build the ordered list of expected pulse-event times/indices.

    Parameters
    ----------
    recipe:
        The ALD/ALE recipe.
    data:
        The dataset being searched — ``expected_idx`` is located directly in
        ``data.time`` via binary search rather than derived from a single
        global sample interval, so this works regardless of whether samples
        are evenly spaced (e.g. instrument bursts of near-duplicate
        timestamps).

    Returns
    -------
    list[_EventSpec]
        One entry per expected pulse, in chronological order.
    """
    events: list[_EventSpec] = []
    t = recipe.start_time
    time = data.time
    n = len(time)

    for outer in range(recipe.repeats):
        for sc_idx, sub_cycle in enumerate(recipe.sub_cycles):
            for run in range(sub_cycle.repeats):
                for s_idx, step in enumerate(sub_cycle.steps):
                    expected_idx = int(np.clip(np.searchsorted(time, t), 0, n - 1))
                    events.append(
                        _EventSpec(
                            step_name=step.name,
                            mass_effect=step.mass_effect,
                            expected_time=t,
                            expected_idx=expected_idx,
                            sub_cycle_index=sc_idx,
                            sub_cycle_run=run,
                            step_index=s_idx,
                            outer_cycle=outer,
                            step_duration=step.duration,
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
    time = data.time
    n = len(mass)

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
    timeline = _build_timeline(params.recipe, data)
    candidates.sort()
    onsets, confidence = _assign_candidates(
        timeline, candidates, mass, time, det.recipe_tolerance
    )

    _check_count(onsets, params.recipe)
    return CycleIndex(step_onsets=onsets, recipe=params.recipe, confidence=confidence)


def _assign_candidates(
    timeline: list[_EventSpec],
    candidates: list[int],
    mass: NDArray[np.float64],
    time: NDArray[np.float64],
    tolerance: float,
) -> tuple[list[tuple[str, int]], list[float]]:
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
    n = len(time)
    k = len(timeline)
    m = len(candidates)
    fallback_cost = float(n)  # always worse than any real in-window match

    windows: list[tuple[int, int]] = []
    tol_samples_list: list[int] = []
    for event in timeline:
        tol_seconds = event.step_duration * tolerance
        t_lo = event.expected_time - tol_seconds
        t_hi = event.expected_time + tol_seconds
        lo = int(np.clip(np.searchsorted(time, t_lo), 0, n - 1))
        hi = int(np.clip(np.searchsorted(time, t_hi), 0, n - 1))
        hi = max(hi, lo)
        windows.append((lo, hi))
        tol_samples_list.append(max(1, hi - lo))

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
    confidence: list[float] = [0.0] * k
    i, j = k, m
    while i > 0:
        step = choice[i][j]
        event = timeline[i - 1]
        if step == 1:
            c = candidates[j - 1]
            onsets[i - 1] = (event.step_name, c)
            tol = tol_samples_list[i - 1]
            dist = abs(c - event.expected_idx)
            # Full confidence for an exact match, decaying linearly to 0.3
            # at the edge of the tolerance window (still a real candidate,
            # just a less certain match).
            confidence[i - 1] = max(0.3, 1.0 - dist / tol) if tol > 0 else 1.0
            i, j = i - 1, j - 1
        elif step == 2:
            warnings.warn(
                f"No PELT candidate found near expected onset for "
                f"'{event.step_name}' at ~{event.expected_time:.1f}s "
                f"(outer cycle {event.outer_cycle}). Using recipe-estimated position.",
                stacklevel=4,
            )
            onsets[i - 1] = (event.step_name, min(event.expected_idx, n - 1))
            # No real changepoint was found — this onset is a pure recipe-timing
            # guess and should be reviewed first.
            confidence[i - 1] = 0.15
            i -= 1
        else:
            j -= 1

    result_onsets = [o for o in onsets if o is not None]
    result_confidence = [c for o, c in zip(onsets, confidence) if o is not None]
    return result_onsets, result_confidence


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
    time = data.time
    n = len(mass)

    # Smooth the mass signal
    win = min(det.smoothing_window, n // 4 * 2 + 1)   # must be odd and < n
    if win % 2 == 0:
        win += 1
    poly = min(det.savgol_polyorder, win - 1)
    smooth = savgol_filter(mass, window_length=win, polyorder=poly)
    # np.gradient accepts the actual sample coordinates, so the derivative is
    # correct at each point's real local spacing rather than assuming a
    # single global sample interval (which breaks down for non-uniformly
    # sampled data, e.g. instrument bursts of near-duplicate timestamps).
    deriv = np.gradient(smooth, time)

    timeline = _build_timeline(params.recipe, data)

    onsets: list[tuple[str, int]] = []
    confidence: list[float] = []

    for event in timeline:
        # Additive tolerance: ±(tol × step_duration), located directly in
        # the time array rather than converted through a global dt.
        tol_seconds = event.step_duration * det.recipe_tolerance
        t_lo = event.expected_time - tol_seconds
        t_hi = event.expected_time + tol_seconds
        lo = int(np.clip(np.searchsorted(time, t_lo), 0, n - 1))
        hi = int(np.clip(np.searchsorted(time, t_hi), 0, n - 1))
        hi = max(hi, lo)

        # Baseline window: go back one full step duration before the search
        # window so we sample genuinely quiet pre-pulse signal rather than
        # the tail of the preceding event's derivative spike.
        baseline_end_time = time[lo] - event.step_duration
        baseline_start_time = baseline_end_time - det.baseline_window
        baseline_end = int(np.clip(np.searchsorted(time, baseline_end_time), 0, n))
        baseline_start = int(np.clip(np.searchsorted(time, baseline_start_time), 0, n))
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
            onsets.append((event.step_name, event.expected_idx))
            confidence.append(0.15)
            continue

        # Select derivative direction based on mass_effect
        if event.mass_effect == "loss":
            scored = -segment
        else:
            scored = segment   # gain or any → look for positive derivative spike

        # Peak of the signed derivative locates the pulse coarsely.
        # First-crossing is avoided: SG pre-ringing causes the derivative to
        # rise 20-25 samples before the actual step, so "first above threshold"
        # is systematically early. The argmax is robust under low SNR, but it
        # marks the steepest part of the rise, which comes after the onset —
        # late by up to a second for slow or smoothed rises — so it is refined
        # below against the raw mass.
        best_local = int(np.argmax(scored))
        peak = float(scored[best_local])
        if peak < threshold:
            warnings.warn(
                f"Low-confidence onset for '{event.step_name}' at "
                f"~{event.expected_time:.1f}s "
                f"(outer cycle {event.outer_cycle}). "
                "Peak derivative is below the adaptive threshold; detection "
                "may be inaccurate. Consider using pelt_guided instead.",
                stacklevel=3,
            )

        lower = onsets[-1][1] + 1 if onsets else 0
        sign = -1.0 if event.mass_effect == "loss" else 1.0
        onset = _refine_onset(time, mass, lo + best_local, sign, lower, det)
        onsets.append((event.step_name, onset))
        # How far the derivative peak clears the adaptive threshold — a peak
        # well above threshold is a confident detection, one below it (the
        # warning case above) is not.
        confidence.append(
            float(np.clip(peak / threshold, 0.15, 1.0)) if threshold > 0 else 0.5
        )

    _check_count(onsets, params.recipe)
    return CycleIndex(step_onsets=onsets, recipe=params.recipe, confidence=confidence)


def _refine_onset(
    time: NDArray[np.float64],
    mass: NDArray[np.float64],
    peak_idx: int,
    sign: float,
    lower: int,
    det: DetectionParameters,
) -> int:
    """Move a coarse derivative-peak onset back to where the rise begins.

    The raw mass is compared with a straight-line baseline fitted to the
    ``det.baseline_window`` seconds before the ±``det.refinement_window``
    window around ``peak_idx`` (the line absorbs slow drift). The rise is
    first located as the first of two consecutive samples in that window more
    than ``det.adaptive_threshold_sigma`` noise standard deviations off the
    baseline in the direction ``sign`` (+1 gain, -1 loss). The onset is then
    the corner of a hinge — flat on the baseline, then a straight rise — fitted
    to the samples up to that first risen one, over candidate corners up to
    ``refinement_window`` seconds earlier. Ending the fit at the first risen
    sample keeps a fast, sharply curving rise from biasing the corner early;
    fitting it at all keeps a slow rise, whose first samples barely clear the
    noise, from biasing it late. The onset sample sits on the pre-pulse
    baseline — the convention extraction relies on (``mass_corrected[0] ≈ 0``).

    Returns ``peak_idx`` unchanged when refinement is disabled
    (``refinement_window == 0``), when there is too little quiet signal
    before the window to fit a baseline, or when no departure is found.
    Never returns an index below ``lower`` (keeps onsets strictly increasing).
    """
    if det.refinement_window <= 0:
        return peak_idx
    n = len(time)
    t_peak = time[peak_idx]
    start = max(int(np.searchsorted(time, t_peak - det.refinement_window)), lower)
    stop = min(int(np.searchsorted(time, t_peak + det.refinement_window)), n - 2)
    t_base_start = time[start] - det.baseline_window
    base_start = max(int(np.searchsorted(time, t_base_start)), lower)
    if start - base_start < 5 or stop <= start:
        return peak_idx

    t_base = time[base_start:start] - time[base_start]
    m_base = mass[base_start:start]
    if np.ptp(t_base) > 0:
        slope, intercept = np.polyfit(t_base, m_base, 1)
    else:
        slope, intercept = 0.0, float(np.mean(m_base))
    sigma = float(np.std(m_base - (intercept + slope * t_base)))
    if sigma <= 0:
        return peak_idx

    # Offset from the baseline (positive = in the expected direction), from the
    # start of the baseline stretch so the corner can sit before ``start``.
    t_all = time[base_start : stop + 2] - time[base_start]
    off = sign * (mass[base_start : stop + 2] - (intercept + slope * t_all))
    w0 = start - base_start
    k = det.adaptive_threshold_sigma
    departed = (off[w0:-1] > k * sigma) & (off[w0 + 1 :] > k * sigma)
    if not departed.any():
        return peak_idx
    first = w0 + int(np.argmax(departed))    # first clearly-risen sample

    j0 = int(np.searchsorted(t_all, t_all[first] - det.refinement_window))
    best_sse, corner = np.inf, first - 1
    for j in range(min(j0, first - 1), first):
        dt_rise = t_all[j + 1 : first + 1] - t_all[j]
        rise = off[j + 1 : first + 1]
        denom = float(dt_rise @ dt_rise)
        rate = float(dt_rise @ rise) / denom if denom > 0 else 0.0
        flat = off[j0 : j + 1]
        sse = float(flat @ flat) + float(np.sum((rise - rate * dt_rise) ** 2))
        if sse < best_sse:
            best_sse, corner = sse, j
    return max(base_start + corner, lower)


# ── Shared validation ─────────────────────────────────────────────────────────


def _check_count(
    onsets: list[tuple[str, int]],
    recipe: Recipe,
) -> None:
    """Raise DetectionError if the detected count is wrong, or onsets aren't
    strictly increasing.

    extraction.py assumes each step's window runs from its onset to the
    *next* onset — a tie or reversal produces a zero- or negative-length
    slice there, which surfaces as a confusing IndexError far from the real
    cause. ``hybrid`` is especially prone to this: unlike ``pelt_guided``
    (whose DP assignment explicitly preserves chronological order — see its
    docstring), it searches each event's window independently with no
    coordination between neighbors, so a long step's wide tolerance window
    can collide with a short neighboring step's window.
    """
    expected = recipe.total_events
    got = len(onsets)
    if got != expected:
        raise DetectionError(
            f"Expected {expected} pulse events from recipe but detected {got}. "
            "Try adjusting DetectionParameters.recipe_tolerance or penalty."
        )
    for i in range(1, got):
        prev_name, prev_idx = onsets[i - 1]
        name, idx = onsets[i]
        if idx <= prev_idx:
            raise DetectionError(
                f"Detected onset for '{name}' (event {i}, sample {idx}) is not "
                f"after '{prev_name}' (event {i - 1}, sample {prev_idx}) — onsets "
                "must be strictly increasing. This usually means two adjacent "
                "steps' search windows overlapped (a common failure mode for "
                "recipes with very uneven step durations under the 'hybrid' "
                "method). Try pelt_guided instead, or reduce recipe_tolerance."
            )
