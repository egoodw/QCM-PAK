"""
Pulse detection: find the onset index of every step in the Recipe.

Three algorithms are provided:

``pelt_guided`` (default)
    Uses the PELT changepoint detection algorithm (L2 cost model) to locate all
    mean-shift changepoints in the mass signal globally. The set of candidates
    is then filtered and assigned to the expected Recipe events using the recipe timeline and a ±``recipe_tolerance``
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

``autocorr``
    Recipe-guided iterative re-alignment on the mass derivative. A comb over
    every expected event finds one shared offset (the recipe start error);
    each class of pulses then builds a median template and every pulse is
    re-estimated against the template of the others, with a robust line
    (offset plus drift) through the confident ones. Pulses that disagree with
    the line, sit on a search bound or score weakly get
    ``FLOOR_CONFIDENCE`` in ``CycleIndex.confidence`` and are placed on the
    line.

All algorithms return a :class:`~qcm_pak._types.CycleIndex` with exactly
``recipe.total_events`` detected onsets, or raise
:class:`~qcm_pak.exceptions.DetectionError`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, find_peaks, savgol_filter
from scipy.stats import theilslopes

from qcm_pak._pelt import pelt_l2
from qcm_pak._types import CycleIndex, MassDataset
from qcm_pak.exceptions import DetectionError
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import Recipe
from qcm_pak.sampling import (
    SamplingInfo,
    analyze_sampling,
    collapse_repeats,
    median_interval,
)

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

    if det_params.collapse_repeats:
        sampling = analyze_sampling(data.time, data.mass)
        if sampling.repeats_detected:
            return _detect_on_collapsed(data, params, det_params, sampling)
    return _run_method(data, params, det_params)


def _run_method(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
) -> CycleIndex:
    """Run the algorithm named by ``det.method``."""
    if det.method == "pelt_guided":
        return _detect_pelt_guided(data, params, det)
    if det.method == "hybrid":
        return _detect_hybrid(data, params, det)
    if det.method == "autocorr":
        return _detect_autocorr(data, params, det)
    raise ValueError(f"Unknown detection method {det.method!r}.")


def _detect_on_collapsed(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
    sampling: SamplingInfo,
) -> CycleIndex:
    """Detect on one row per measurement, then map the onsets to the full trace.

    An instrument that stores each reading several times makes every
    sample-count setting (smoothing window, minimum spacing) a multiple too
    large and flattens the rise into a staircase. Dropping the repeats first
    puts detection back on the real measurements. A long flat stretch or time
    gap keeps both of its ends, so it keeps its true length.
    """
    keep, last_baseline = collapse_repeats(data.mass, sampling.repeat_length)
    collapsed = MassDataset(
        time=data.time[keep],
        frequency=data.frequency[keep],
        mass=data.mass[keep],
        temperature=None if data.temperature is None else data.temperature[keep],
    )
    found = _run_method(collapsed, params, det)
    onsets = [(name, int(last_baseline[i])) for name, i in found.step_onsets]
    _check_count(onsets, params.recipe)
    return CycleIndex(
        step_onsets=onsets, recipe=found.recipe, confidence=found.confidence
    )


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
    mass = data.mass
    time = data.time
    n = len(mass)

    # Estimate penalty
    if det.penalty == "auto":
        penalty = _bic_penalty(n, _estimate_noise(mass))
    else:
        penalty = float(det.penalty)

    # Run PELT with L2 cost (detects mean shifts — exactly what ALD pulses are)
    breakpoints = pelt_l2(mass, penalty, min_size=det.min_pulse_spacing)
    # Each breakpoint is the first index of a new segment (exclusive end of the
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


# A smoothing_window given in samples is read at this rate (the rate the default
# of 50 samples was tuned for), which turns it into a duration.
_REFERENCE_INTERVAL_S = 0.1
_MIN_SMOOTHING_SAMPLES = 5


def _smoothing_samples(det: DetectionParameters, dt: float) -> int:
    """Length of the smoothing window in samples at sampling interval ``dt``."""
    if dt <= 0:
        return det.smoothing_window
    seconds = (
        det.smoothing_window_s
        if det.smoothing_window_s is not None
        else det.smoothing_window * _REFERENCE_INTERVAL_S
    )
    return max(int(round(seconds / dt)), _MIN_SMOOTHING_SAMPLES)

def _detect_hybrid(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
) -> CycleIndex:
    """Recipe-guided derivative search (port of QCMPy 0.5 CycleDetector)."""
    mass = data.mass
    time = data.time
    n = len(mass)

    # The window and the refinement search are set in seconds, so a trace at
    # another sampling rate is smoothed over the same stretch of time.
    dt = median_interval(time)
    if det.refinement_window > 0 and dt > 0:
        det = replace(det, refinement_window=max(det.refinement_window, dt))

    # Smooth the mass signal
    win = min(_smoothing_samples(det, dt), n // 4 * 2 + 1)   # odd and < n
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
    # Per-sample noise of the raw mass, for onset refinement: the MAD of first
    # differences estimates sqrt(2) times it and ignores the few large steps.
    noise_floor = _estimate_noise(mass) / np.sqrt(2)

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
        onset = _refine_onset(
            time, mass, lo + best_local, sign, lower, det, noise_floor
        )
        onsets.append((event.step_name, onset))
        # How far the derivative peak clears the adaptive threshold — a peak
        # well above threshold is a confident detection, one below it (the
        # warning case above) is not.
        confidence.append(
            float(np.clip(peak / threshold, 0.15, 1.0)) if threshold > 0 else 0.5
        )

    _check_count(onsets, params.recipe)
    return CycleIndex(step_onsets=onsets, recipe=params.recipe, confidence=confidence)


# Fewest samples onset refinement fits its baseline line to. Fewer give a slope
# and noise estimate too rough to tell the start of a rise from a noisy sample.
_MIN_BASELINE_SAMPLES = 15


def _refine_onset(
    time: NDArray[np.float64],
    mass: NDArray[np.float64],
    peak_idx: int,
    sign: float,
    lower: int,
    det: DetectionParameters,
    noise_floor: float = 0.0,
) -> int:
    """Move a coarse derivative-peak onset back to where the rise begins.

    The raw mass is compared with a straight-line baseline fitted to the
    ``det.baseline_window`` seconds before the ±``det.refinement_window``
    window around ``peak_idx`` (the line absorbs slow drift), widened to at
    least ``_MIN_BASELINE_SAMPLES`` samples at low sample rates. The rise is
    first located as the first of two consecutive samples in that window more
    than ``det.adaptive_threshold_sigma`` noise standard deviations off the
    baseline in the direction ``sign`` (+1 gain, -1 loss). The noise is the
    scatter about the line, but never less than ``noise_floor`` (the
    trace-wide estimate): a short baseline that happens to be quiet would
    otherwise let a noisy pre-pulse sample pass for the rise. The onset is then
    the corner of a hinge — flat on the baseline, then a straight rise — fitted
    to the samples up to that first risen one, over candidate corners up to
    ``refinement_window`` seconds earlier. Ending the fit at the first risen
    sample keeps a fast, sharply curving rise from biasing the corner early;
    fitting it at all keeps a slow rise, whose first samples barely clear the
    noise, from biasing it late. The onset sample sits on the pre-pulse
    baseline — the convention extraction relies on (``mass_corrected[0] ≈ 0``).

    Returns ``peak_idx`` unchanged when refinement is disabled
    (``refinement_window == 0``), when there are fewer than
    ``_MIN_BASELINE_SAMPLES`` quiet samples before the window to fit a
    baseline, or when no departure is found.
    Never returns an index below ``lower`` (keeps onsets strictly increasing).
    """
    if det.refinement_window <= 0:
        return peak_idx
    n = len(time)
    t_peak = time[peak_idx]
    start = max(int(np.searchsorted(time, t_peak - det.refinement_window)), lower)
    stop = min(int(np.searchsorted(time, t_peak + det.refinement_window)), n - 2)
    t_base_start = time[start] - det.baseline_window
    base_start = int(np.searchsorted(time, t_base_start))
    # At low sample rates the baseline window holds only a few samples, which
    # underestimates the noise and lets a pre-pulse sample pass for the start
    # of the rise; widen it to a minimum number of samples.
    base_start = max(min(base_start, start - _MIN_BASELINE_SAMPLES), lower)
    if start - base_start < _MIN_BASELINE_SAMPLES or stop <= start:
        return peak_idx

    t_base = time[base_start:start] - time[base_start]
    m_base = mass[base_start:start]
    if np.ptp(t_base) > 0:
        slope, intercept = np.polyfit(t_base, m_base, 1)
    else:
        slope, intercept = 0.0, float(np.mean(m_base))
    # ddof=2: the line's slope and intercept were fitted to these samples.
    sigma = float(np.std(m_base - (intercept + slope * t_base), ddof=2))
    sigma = max(sigma, noise_floor)
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


# ── Autocorrelation (iterative re-alignment) algorithm ────────────────────────

# Confidence given to a pulse the method is unsure about. Reviewers should look
# at anything at or below it first.
FLOOR_CONFIDENCE = 0.15

_AC_SMOOTH_S = 0.3          # Gaussian sigma of the derivative filter (>= 0.8 samples)
_AC_FINE_DT_S = 0.025       # working grid; the derivative is spline-upsampled to it
_AC_MAX_UPSAMPLE = 8
_AC_MIN_ITER = 3            # always run this many passes ...
_AC_CONVERGED_S = 0.02      # ... and stop once no pulse moves more than this
_AC_PRE_S = 1.0             # template window before the onset
_AC_POST_EXTRA_S = 1.0      # template window after the pulse closes
_AC_EDGE_FRAC = 0.8         # onset = where the template rise crosses this fraction
_AC_MIN_CLASS = 3           # smaller classes are pooled with other small ones
_AC_NULL_PERCENTILE = 90.0
_AC_DEV_K = 3.0             # flag when further from the schedule than K class spreads
_AC_DEV_FLOOR_S = 0.5       # ... but never within this (half-sample noise)
_AC_DEV_CAP_S = 0.8         # ... and always beyond this (noisy classes spread wide)
_AC_WEAK_FRAC = 0.3         # weak score: under the quiet null and this x class median
_AC_FLOOR_TOL_FRAC = 0.4    # onset = floor(estimate + this x sample interval)
_AC_CONFIDENT_FRAC = 0.5    # pulses scoring this x class median feed the line fit
_AC_DRIFT_MIN_EVENTS = 8    # fewest confident pulses to fit a drift as well


class _Grid(NamedTuple):
    """The fine, uniform grid the derivative is worked on."""

    t0: float       # time of the first fine sample
    dtf: float      # fine sample interval, in seconds
    upsample: int   # fine samples per native sample
    x: NDArray[np.float64]


class _AcEvents(NamedTuple):
    """What the recipe says about every event, on the fine grid."""

    timeline: list[_EventSpec]
    tn: NDArray[np.float64]         # nominal onset times
    nominal: NDArray[np.float64]    # nominal onsets, in fine samples
    bound_n: NDArray[np.int64]      # per-pulse search bound, in fine samples
    glob_n: int                     # shared-offset bound, in fine samples
    pulses: list[float]


class _AcFit(NamedTuple):
    """Where the re-alignment settled."""

    est: NDArray[np.float64]
    q: NDArray[np.float64]              # the schedule line, in fine samples
    score: NDArray[np.float64]
    ratio: NDArray[np.float64]
    at_bound: NDArray[np.bool_]
    templates: dict[Any, NDArray[np.float64]]
    offset: float                       # in fine samples
    drift: float                        # fine samples per fine sample


def _pulse_lengths(recipe: Recipe) -> list[float]:
    """Pulse length of every event, in the order ``_build_timeline`` uses."""
    out: list[float] = []
    for _ in range(recipe.repeats):
        for sub_cycle in recipe.sub_cycles:
            for _ in range(sub_cycle.repeats):
                out.extend(step.pulse for step in sub_cycle.steps)
    return out


def _fine_derivative(data: MassDataset) -> _Grid:
    """Derivative of the mass on a fine uniform grid."""
    t, first = np.unique(np.asarray(data.time, dtype=float), return_index=True)
    m = np.asarray(data.mass, dtype=float)[first]
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if dt <= 0:
        raise DetectionError(
            "Cannot work out the sampling interval: the time column has fewer "
            "than two distinct values."
        )
    n = int(round((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    mu = np.interp(tu, t, m)
    sigma = max(_AC_SMOOTH_S, 0.8 * dt) / dt
    x = gaussian_filter1d(mu, sigma, order=1, mode="nearest") / dt
    up = int(min(_AC_MAX_UPSAMPLE, max(1, math.ceil(dt / _AC_FINE_DT_S - 1e-9))))
    dtf = dt / up
    if up > 1:
        xf = CubicSpline(tu, x)(tu[0] + dtf * np.arange((n - 1) * up + 1))
    else:
        xf = x
    return _Grid(float(tu[0]), dtf, up, np.asarray(xf, dtype=float))


def _comb_offset(
    xabs: NDArray[np.float64],
    nominal: NDArray[np.float64],
    keys: list[Any],
    bound_n: int,
) -> int:
    """Shared offset (fine samples) that puts the most |signal| under the events.

    Each class of pulses is scored on its own and z-scored before the classes
    are added, so a class with big pulses does not outvote the rest.
    """
    xp = np.pad(xabs, bound_n)
    lags = np.arange(-bound_n, bound_n + 1)
    total = np.zeros(lags.size)
    for key in dict.fromkeys(keys):
        members = [i for i, k in enumerate(keys) if k == key]
        at = np.round(nominal[members])[:, None].astype(int) + bound_n + lags[None, :]
        s = xp[np.clip(at, 0, xp.size - 1)].sum(axis=0)
        med = np.median(s)
        mad = 1.4826 * np.median(np.abs(s - med)) + 1e-12
        total += (s - med) / mad
    return int(lags[int(np.argmax(total))])


def _range_guard(
    keys: list[Any],
    sign: NDArray[np.float64],
    declared: list[str],
    offset_n: int,
    bound_n: int,
) -> str:
    """Reason the shared offset cannot be trusted (``""`` when it can).

    The |signal| comb cannot tell a pulse from its neighbour one step away, so a
    start error beyond the capture range lands every nominal event on the next
    step's pulse and nothing in |signal| shows it. Two tell-tales: the offset
    sits on its search bound, or the data signs of *every* declared class
    contradict the declared mass effects (one wrong declaration is not enough).
    """
    if abs(offset_n) >= bound_n - 1:
        return "offset-at-bound"
    verdicts = []
    for key in dict.fromkeys(keys):
        i = keys.index(key)
        if declared[i] in ("gain", "loss"):
            verdicts.append(sign[i] == (1.0 if declared[i] == "gain" else -1.0))
    if len(verdicts) >= 2 and not any(verdicts):
        return "signs-contradict-recipe"
    return ""


def _class_signs(
    xf: NDArray[np.float64],
    centers: NDArray[np.float64],
    declared: list[str],
    keys: list[Any],
    half_n: int,
) -> NDArray[np.float64]:
    """Sign of every (step, pulse) class, read from the data near its centres."""
    signs = np.ones(len(keys))
    for key in dict.fromkeys(keys):
        members = [i for i, k in enumerate(keys) if k == key]
        peaks = []
        for i in members:
            c = int(round(centers[i]))
            seg = xf[max(0, c - half_n) : c + half_n + 1]
            if seg.size:
                peaks.append(seg[int(np.argmax(np.abs(seg)))])
        v = float(np.median(peaks)) if peaks else 0.0
        if v != 0:
            sign = 1.0 if v > 0 else -1.0
        else:
            sign = -1.0 if declared[members[0]] == "loss" else 1.0
        for i in members:
            signs[i] = sign
    return signs


def _parabola(y: NDArray[np.float64], i: int) -> float:
    """Sub-sample offset of the peak at ``y[i]``, in [-1, 1]."""
    if 0 < i < y.size - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d < 0:
            return float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / d, -1.0, 1.0))
    return 0.0


def _template_edge(template: NDArray[np.float64], pre_n: int) -> float:
    """Template index where the rise before its maximum crosses the edge level."""
    jm = int(np.argmax(template))
    base = float(np.median(template[: max(1, pre_n // 2)]))
    level = base + _AC_EDGE_FRAC * (float(template[jm]) - base)
    j = jm
    while j > 0 and template[j - 1] >= level:
        j -= 1
    if j == 0:
        return 0.0
    lo, hi = float(template[j - 1]), float(template[j])
    return (j - 1) + (level - lo) / (hi - lo) if hi != lo else float(j)


def _unit_rows(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows with their mean removed and scaled to length one."""
    rows = rows - rows.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.divide(rows, norm, out=np.zeros_like(rows), where=norm > 0)


def _second_peak_ratio(corr: NDArray[np.float64], k: int) -> float:
    """Height of the best rival peak over the best peak (rivals stand clear)."""
    if corr[k] <= 0:
        return 1.0
    peaks, _ = find_peaks(corr, prominence=0.3 * corr[k])
    rivals = [corr[j] for j in peaks if j != k]
    return float(max(rivals) / corr[k]) if rivals else 0.0


def _null_threshold(
    xp: NDArray[np.float64],
    pad: int,
    template: NDArray[np.float64],
    pre_n: int,
    zones: list[tuple[int, int]],
    window_n: int,
) -> float:
    """Best matched-filter score the template reaches on quiet stretches.

    ``nan`` when the trace has too few quiet stretches to say.
    """
    length = template.size
    corr = fftconvolve(xp, _unit_rows(template)[::-1], mode="valid")
    quiet = np.ones(corr.size, dtype=bool)
    for lo, hi in zones:
        quiet[max(0, lo + pad - pre_n - length) : max(0, hi + pad - pre_n + 1)] = False
    quiet[: pad // 2] = False
    quiet[-(pad // 2) :] = False
    w = max(window_n, 1)
    maxima = [
        float(corr[s : s + w].max())
        for s in range(0, corr.size - w, w)
        if quiet[s : s + w].all()
    ]
    if len(maxima) >= 5:
        return float(np.percentile(maxima, _AC_NULL_PERCENTILE))
    return float("nan")


def _realign_class(
    xp: NDArray[np.float64],
    pad: int,
    pos: NDArray[np.int64],
    centre: NDArray[np.float64],
    bound_n: NDArray[np.int64],
    pre_n: int,
    post_n: int,
) -> dict[str, Any]:
    """One pass for one class: re-estimate each pulse against the others."""
    length = pre_n + post_n + 1
    view = sliding_window_view(xp, length)
    segs = _unit_rows(view[pos + pad - pre_n])
    m = len(pos)
    full = np.median(segs, axis=0)
    est = np.zeros(m)
    score = np.zeros(m)
    ratio = np.ones(m)
    at_bound = np.zeros(m, dtype=bool)
    for i in range(m):
        tmpl = np.median(np.delete(segs, i, axis=0), axis=0) if m > 2 else full
        edge = _template_edge(tmpl, pre_n)
        t_hat = _unit_rows(tmpl)
        c0 = int(round(centre[i]))
        lo, hi = c0 - int(bound_n[i]), c0 + int(bound_n[i])
        corr = view[lo + pad - pre_n : hi + pad - pre_n + 1] @ t_hat
        k = int(np.argmax(corr))
        est[i] = lo + k + _parabola(corr, k) - pre_n + edge
        score[i] = corr[k]
        at_bound[i] = k <= 1 or k >= corr.size - 2
        ratio[i] = _second_peak_ratio(corr, k)
    return {
        "est": est,
        "score": score,
        "ratio": ratio,
        "at_bound": at_bound,
        "template": full,
    }


def _ac_events(
    data: MassDataset, params: ALDParameters, det: DetectionParameters, dtf: float
) -> _AcEvents:
    """Expected events on the fine grid, with the search bounds."""
    timeline = _build_timeline(params.recipe, data)
    tn = np.array([e.expected_time for e in timeline])
    dur = np.array([e.step_duration for e in timeline])
    return _AcEvents(
        timeline=timeline,
        tn=tn,
        nominal=tn,    # shifted to the grid origin by the caller
        bound_n=np.maximum(
            2, np.round(det.autocorr_search_frac * dur / dtf)
        ).astype(np.int64),
        glob_n=max(1, int(round(det.autocorr_global_frac * float(dur.min()) / dtf))),
        pulses=_pulse_lengths(params.recipe),
    )


def _ac_classes(
    keys: list[Any], sign: NDArray[np.float64]
) -> tuple[list[Any], dict[Any, list[int]]]:
    """Group events into (step, pulse, sign) classes; pool the small ones."""
    full = [(k[0], k[1], int(s)) for k, s in zip(keys, sign, strict=True)]
    counts: dict[Any, int] = {}
    for k in full:
        counts[k] = counts.get(k, 0) + 1
    full = [k if counts[k] >= _AC_MIN_CLASS else ("pool", 0.0, k[2]) for k in full]
    classes: dict[Any, list[int]] = {}
    for i, k in enumerate(full):
        classes.setdefault(k, []).append(i)
    return full, classes


def _ac_align(
    grid: _Grid,
    ev: _AcEvents,
    sign: NDArray[np.float64],
    classes: dict[Any, list[int]],
    a_glob: int,
    n_iter: int,
) -> tuple[_AcFit, dict[Any, dict[str, Any]], int]:
    """Iterate: per-class templates, per-pulse shifts, a robust schedule line."""
    dtf = grid.dtf
    n_events = len(ev.timeline)
    pre_n = max(2, int(round(_AC_PRE_S / dtf)))
    dur = np.array([e.step_duration for e in ev.timeline])
    cls: dict[Any, dict[str, Any]] = {}
    for key, members in classes.items():
        post_s = min(ev.pulses[i] + _AC_POST_EXTRA_S for i in members)
        post_s = min(post_s, 0.6 * min(dur[i] for i in members))
        post_n = max(4, int(round(post_s / dtf)))
        pad = int(ev.bound_n[members].max() + pre_n + post_n + ev.glob_n + 8)
        cls[key] = {
            "members": np.array(members),
            "post_n": post_n,
            "pad": pad,
            "xp": np.pad(sign[members[0]] * grid.x, pad),
        }

    a_line, b_line = float(a_glob), 0.0
    t_ref = float(ev.tn[0])

    def line() -> NDArray[np.float64]:
        return ev.nominal + a_line + b_line * (ev.tn - t_ref) / dtf

    q = line()
    pos = np.round(q).astype(np.int64)
    est = q.copy()
    score = np.zeros(n_events)
    ratio = np.ones(n_events)
    at_bound = np.zeros(n_events, dtype=bool)
    templates: dict[Any, NDArray[np.float64]] = {}
    iterations = 0
    for it in range(n_iter):
        iterations = it + 1
        prev = est.copy()
        for key, c in cls.items():
            mem = c["members"]
            out = _realign_class(
                c["xp"], c["pad"], pos[mem], q[mem], ev.bound_n[mem], pre_n,
                c["post_n"],
            )
            est[mem], score[mem] = out["est"], out["score"]
            ratio[mem], at_bound[mem] = out["ratio"], out["at_bound"]
            templates[key] = out["template"]
        # Confident pulses: score at least a fraction of their class median.
        conf = np.zeros(n_events, dtype=bool)
        for c in cls.values():
            mem = c["members"]
            med = float(np.median(score[mem]))
            conf[mem] = (
                (score[mem] >= _AC_CONFIDENT_FRAC * med)
                & (score[mem] > 0)
                & ~at_bound[mem]
            )
        resid = est - ev.nominal
        if conf.sum() >= _AC_DRIFT_MIN_EVENTS:
            slope = theilslopes(resid[conf], ev.tn[conf])[0]
            b_line = float(slope) * dtf
            a_line = float(np.median(resid[conf] - slope * (ev.tn[conf] - t_ref)))
        elif conf.sum() >= 1:
            a_line, b_line = float(np.median(resid[conf])), 0.0
        q = line()
        # Next pass aligns on the trustworthy estimates, the line elsewhere.
        pos = np.round(np.where(conf, est, q)).astype(np.int64)
        moved = float(np.max(np.abs(est - prev))) * dtf
        if it + 1 >= _AC_MIN_ITER and moved < _AC_CONVERGED_S:
            break
    fit = _AcFit(est, q, score, ratio, at_bound, templates, a_line, b_line)
    return fit, cls, iterations


def _ac_confidence(
    ev: _AcEvents,
    fit: _AcFit,
    cls: dict[Any, dict[str, Any]],
    pre_n: int,
    dtf: float,
    guard: str,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Flag pulses that sit off the schedule, at a bound or have a weak signal."""
    n_events = len(ev.timeline)
    dev_s = (fit.est - fit.q) * dtf
    zones = [
        (int(round(fit.q[i])) - int(ev.bound_n[i]),
         int(round(fit.q[i])) + int(ev.bound_n[i]))
        for i in range(n_events)
    ]
    flagged = np.zeros(n_events, dtype=bool)
    confidence = np.zeros(n_events)
    for key, c in cls.items():
        mem = c["members"]
        w = int(2 * ev.bound_n[mem].max() + 1)
        padded_zones = [(lo - pre_n - c["post_n"], hi + pre_n + c["post_n"])
                        for lo, hi in zones]
        thr = _null_threshold(
            c["xp"], c["pad"], fit.templates[key], pre_n, padded_zones, w
        )
        med = float(np.median(fit.score[mem]))
        if not np.isfinite(thr):
            thr = 0.3 * med
        d = dev_s[mem]
        spread = 1.4826 * float(np.median(np.abs(d - np.median(d))))
        limit = float(np.clip(_AC_DEV_K * spread, _AC_DEV_FLOOR_S, _AC_DEV_CAP_S))
        for i in mem:
            bad = (
                abs(dev_s[i]) > limit
                or fit.at_bound[i]
                or (fit.score[i] <= thr and fit.score[i] < _AC_WEAK_FRAC * med)
            )
            flagged[i] = bad
            if bad:
                confidence[i] = FLOOR_CONFIDENCE
            else:
                span = max(med - thr, 1e-12)
                confidence[i] = float(
                    np.clip(0.3 + 0.7 * (fit.score[i] - thr) / span, 0.3, 1.0)
                )
    if guard:    # the alignment itself is in doubt: every pulse needs a look
        flagged[:] = True
        confidence[:] = FLOOR_CONFIDENCE
    return flagged, confidence


def _detect_autocorr(
    data: MassDataset,
    params: ALDParameters,
    det: DetectionParameters,
    snap_flagged: bool = True,
) -> CycleIndex:
    """Recipe-guided iterative re-alignment on the mass derivative.

    The recipe gives the nominal time of every event. A comb over all of them
    finds one shared offset (the recipe start error); each class of pulses then
    builds a median template, and every pulse is re-estimated against the
    template of the others inside a small window around the shared schedule.
    Pulses whose estimate disagrees with the schedule, sits on its search bound
    or scores weakly are flagged with :data:`FLOOR_CONFIDENCE` and, when
    ``snap_flagged`` is true, put on the schedule line.

    The onset is the last sample at or before the estimated rise, the same
    convention as ``hybrid``.
    """
    grid = _fine_derivative(data)
    ev = _ac_events(data, params, det, grid.dtf)
    ev = ev._replace(nominal=(ev.tn - grid.t0) / grid.dtf)
    declared = [e.mass_effect for e in ev.timeline]

    # Shared offset, sign-free: the declared mass effects are not trusted here.
    keys = [(e.step_name, round(p, 3)) for e, p in zip(ev.timeline, ev.pulses,
                                                        strict=True)]
    a_glob = _comb_offset(np.abs(grid.x), ev.nominal, keys, ev.glob_n)
    sign = _class_signs(
        grid.x, ev.nominal + a_glob, declared, keys, int(round(1.0 / grid.dtf))
    )
    guard = _range_guard(keys, sign, declared, a_glob, ev.glob_n)
    _, classes = _ac_classes(keys, sign)

    fit, cls, _ = _ac_align(
        grid, ev, sign, classes, a_glob, det.autocorr_iterations
    )
    pre_n = max(2, int(round(_AC_PRE_S / grid.dtf)))
    flagged, confidence = _ac_confidence(ev, fit, cls, pre_n, grid.dtf, guard)

    final = np.where(flagged & snap_flagged, fit.q, fit.est)
    t_on = grid.t0 + final * grid.dtf
    time = data.time
    native_dt = grid.dtf * grid.upsample
    idx = np.clip(
        np.searchsorted(time, t_on + _AC_FLOOR_TOL_FRAC * native_dt + 1e-9, "right")
        - 1,
        0,
        len(time) - 1,
    )
    for i in range(1, len(idx)):    # strictly increasing, as _check_count needs
        idx[i] = max(idx[i], idx[i - 1] + 1)
    onsets = [(ev.timeline[i].step_name, int(idx[i])) for i in range(len(idx))]
    _check_count(onsets, params.recipe)
    return CycleIndex(
        step_onsets=onsets,
        recipe=params.recipe,
        confidence=[float(c) for c in confidence],
    )


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
