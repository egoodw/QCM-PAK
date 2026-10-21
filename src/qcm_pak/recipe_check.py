"""Check that a recipe fits the data it is applied to.

A recipe with the wrong cycle count, a purge that is too long or too short, or
a start time far from the first pulse still runs through detection, which then
returns onsets that sit on the wrong pulses. The checks here look at the trace
itself, without the recipe: they find the pulses, measure how often they arrive
and compare that with what the recipe says. Each problem comes back as a
:class:`~qcm_pak._types.RecipeWarning` with a suggested recipe change.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from qcm_pak._types import MassDataset, RecipeWarning
from qcm_pak.recipe import Recipe

_GRID_DT_S = 0.25           # pulse search grid; rises are slower than this
_PEAK_K = 8.0               # a pulse stands out by this many noise spreads
_COUNT_SLACK = 0.02         # tolerated share of pulses missed or added
_DRIFT_FRACTION = 0.5       # of the search window (tolerance x shortest step)


def expected_onsets(recipe: Recipe) -> NDArray[np.float64]:
    """Onset time in seconds of every pulse the recipe describes."""
    durations = np.array(
        [
            step.duration
            for sc in recipe.sub_cycles
            for _ in range(sc.repeats)
            for step in sc.steps
        ]
    )
    within = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    cycles = np.arange(recipe.repeats)[:, None] * durations.sum()
    return recipe.start_time + (cycles + within[None, :]).ravel()


def pulse_times(data: MassDataset, min_gap_s: float) -> NDArray[np.float64]:
    """Times of the mass steps in the trace, found without the recipe.

    The mass is averaged onto a coarse grid, so the cost does not depend on the
    sampling rate or on repeated or bursty readings. A step is a peak of the
    absolute slope that stands out from the typical slope by ``_PEAK_K``
    robust spreads and is at least ``min_gap_s`` from its neighbours.
    """
    t = np.asarray(data.time, dtype=float)
    m = np.asarray(data.mass, dtype=float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m = t[ok], m[ok]
    if t.size < 8 or t[-1] <= t[0]:
        return np.empty(0)
    span = float(t[-1] - t[0])
    dt = max(_GRID_DT_S, span / (t.size - 1))
    n_bins = int(span / dt) + 1
    bins = ((t - t[0]) / dt).astype(np.intp)
    count = np.bincount(bins, minlength=n_bins)
    total = np.bincount(bins, weights=m, minlength=n_bins)
    have = count > 0
    if have.sum() < 8:
        return np.empty(0)
    grid = np.interp(np.arange(n_bins), np.flatnonzero(have), total[have] / count[have])
    slope = np.abs(np.gradient(gaussian_filter1d(grid, 1.0)))
    centre = float(np.median(slope))
    spread = 1.4826 * float(np.median(np.abs(slope - centre)))
    if spread <= 0:
        return np.empty(0)
    height = centre + _PEAK_K * spread
    peaks, _ = find_peaks(
        slope, height=height, prominence=height, distance=max(1, int(min_gap_s / dt))
    )
    return t[0] + peaks * dt


def check_recipe(
    data: MassDataset,
    recipe: Recipe,
    tolerance: float = 0.3,
) -> list[RecipeWarning]:
    """Compare ``recipe`` with the pulses in ``data``.

    Parameters
    ----------
    data:
        The mass trace.
    recipe:
        The recipe to check.
    tolerance:
        ``DetectionParameters.recipe_tolerance``: the search window for a pulse
        is this share of its step duration either side of the expected time.

    Returns
    -------
    list[RecipeWarning]
        Empty when the recipe fits. An ``"error"`` means the recipe runs past
        the end of the data, so detection cannot succeed.
    """
    out: list[RecipeWarning] = []
    expected = expected_onsets(recipe)
    k = recipe.steps_per_cycle
    min_duration = min(s.duration for sc in recipe.sub_cycles for s in sc.steps)
    t_end = float(data.time[-1])
    peaks = pulse_times(data, 0.5 * min_duration)

    mean_step = recipe.cycle_duration / k
    seen_step = None
    if peaks.size >= 2 * k and peaks.size >= 3:
        # Peaks sit on a grid, so the spacings are quantised and a median or
        # trimmed mean of them is biased. Average the ones near the median
        # instead, which still leaves out the spacings a missed or extra peak spoils.
        gaps = peaks[k:] - peaks[:-k]
        middle = float(np.median(gaps))
        seen_step = float(np.mean(gaps[np.abs(gaps - middle) <= 0.25 * middle])) / k
    seen_cycles = max(1, round(peaks.size / k)) if peaks.size else None
    window = _DRIFT_FRACTION * tolerance * min_duration

    drift = None
    if seen_step is not None:
        gap = seen_step - mean_step
        if abs(gap) * (expected.size - 1) > window:
            drift = round(gap, 2)

    if expected[-1] >= t_end:
        fit = int((t_end - recipe.start_time) // recipe.cycle_duration)
        repeats = seen_cycles if seen_cycles is not None else max(1, fit)
        suggestion: dict[str, float | int] = {"repeats": repeats}
        if drift is not None:
            suggestion["purge_delta_s"] = drift
        found = (
            f" The data holds about {peaks.size} pulses ({repeats} cycles)."
            if seen_cycles is not None
            else ""
        )
        out.append(
            RecipeWarning(
                "data_ends_early",
                "error",
                f"The recipe's last pulse is expected at {expected[-1]:.0f} s but "
                f"the data ends at {t_end:.0f} s.{found} Check the number of "
                "cycles and the pulse and purge times.",
                suggestion,
            )
        )

    if drift is not None:
        out.append(
            RecipeWarning(
                "timeline_drift",
                "warning",
                f"Pulses arrive every {seen_step:.2f} s on average but the recipe "
                f"expects every {mean_step:.2f} s, so it falls about "
                f"{abs(drift) * (expected.size - 1):.0f} s out of step by the end. "
                "Check the purge times.",
                {"purge_delta_s": drift},
            )
        )

    if seen_step is not None and abs(peaks.size - expected.size) > max(
        1.0, _COUNT_SLACK * expected.size
    ):
        fix = (
            {"repeats": seen_cycles}
            if seen_cycles is not None and seen_cycles != recipe.repeats
            else None
        )
        out.append(
            RecipeWarning(
                "pulse_count_mismatch",
                "warning",
                f"The recipe describes {expected.size} pulses but about "
                f"{peaks.size} can be seen in the data. Check the number of cycles.",
                fix,
            )
        )

    if peaks.size:
        first = recipe.sub_cycles[0].steps[0].duration
        if abs(peaks[0] - recipe.start_time) > tolerance * first:
            out.append(
                RecipeWarning(
                    "start_offset",
                    "warning",
                    f"The first pulse is at about {peaks[0]:.1f} s but the recipe "
                    f"starts at {recipe.start_time:.1f} s.",
                    {"start_time": round(float(peaks[0]), 1)},
                )
            )
    return out


def check_onsets(
    recipe: Recipe,
    onset_times: NDArray[np.float64],
    tolerance: float = 0.3,
) -> list[RecipeWarning]:
    """Warn when detected onsets drift away from the recipe's timeline.

    Fits a line through (detected - expected) against the expected time. A slope
    means the recipe's step length is wrong; the line's span over the run is the
    drift. Complements :func:`check_recipe` when the pulses could not be counted
    on their own.
    """
    expected = expected_onsets(recipe)[: onset_times.size]
    if expected.size < 4:
        return []
    min_duration = min(s.duration for sc in recipe.sub_cycles for s in sc.steps)
    slope = float(np.polyfit(expected, onset_times - expected, 1)[0])
    drift = slope * float(expected[-1] - expected[0])
    if abs(drift) <= _DRIFT_FRACTION * tolerance * min_duration:
        return []
    delta = round(slope * recipe.cycle_duration / recipe.steps_per_cycle, 2)
    return [
        RecipeWarning(
            "timeline_drift",
            "warning",
            f"Detected pulses drift {abs(drift):.0f} s away from the recipe's "
            "timeline over the run. Check the purge times.",
            {"purge_delta_s": delta},
        )
    ]
