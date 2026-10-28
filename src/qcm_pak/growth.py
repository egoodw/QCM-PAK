"""Growth-switch detection: find where a run starts to gain mass.

ALD and ALE runs sometimes show no growth, or even mass loss, for the first
cycles and then switch to steady mass gain (nucleation delay, or incubation).
:func:`find_growth_switch` locates that switch from the per-cycle mass change
and :func:`detect_growth_switch` does the same for an
:class:`~qcm_pak._types.AnalysisResult`. Both only *detect and report*; neither
fits a model to either regime, and neither changes any existing result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qcm_pak._types import AnalysisResult
from qcm_pak.exceptions import QCMPakError

# Scale factor turning a median absolute deviation into a standard deviation
# for normally distributed data.
_MAD_TO_SIGMA = 1.4826


@dataclass
class GrowthSwitchResult:
    """Outcome of :func:`find_growth_switch`.

    Parameters
    ----------
    found:
        ``True`` if a growth switch was found. When ``False``,
        ``switch_cycle``, ``switch_time`` and ``incubation_time`` are ``None``.
    switch_cycle:
        0-based index of the first cycle of the first run of ``n_consecutive``
        cycles whose mass change is above the noise threshold. ``0`` means
        the run grows from the start (no incubation).
    switch_time:
        Start time of ``switch_cycle`` in the units of the ``cycle_times``
        passed in (seconds for :func:`detect_growth_switch`). ``None`` if no
        times were given.
    incubation_time:
        ``switch_time`` minus the start time of the first cycle. ``None`` if no
        times were given.
    growth_before, growth_after:
        Mean mass change per cycle (ng/cm² per cycle) before ``switch_cycle``
        and from ``switch_cycle`` to the end of the run. When no switch is
        found, ``growth_before`` is the mean over the whole run and
        ``growth_after`` is ``None``. ``growth_before`` is ``None`` when the run
        grows from cycle 0.
    noise_sigma:
        Robust noise estimate of the per-cycle mass change (ng/cm²).
    threshold:
        ``noise_factor * noise_sigma``; a cycle counts as growing when its
        mass change is above this.
    n_consecutive, noise_factor:
        The parameters used.
    notes:
        Plain-language remarks for the user (e.g. the run never leaves
        loss/flat).
    """

    found: bool
    switch_cycle: int | None
    switch_time: float | None
    incubation_time: float | None
    growth_before: float | None
    growth_after: float | None
    noise_sigma: float
    threshold: float
    n_consecutive: int
    noise_factor: float
    notes: list[str] = field(default_factory=list)


def estimate_cycle_noise(per_cycle_mass_change: Sequence[float]) -> float:
    """Robust noise level (1 sigma) of a per-cycle mass-change series.

    Uses the median absolute deviation of the cycle-to-cycle differences,
    divided by sqrt(2) because a difference of two independent noisy values
    has twice the variance. Differencing removes the slowly varying level
    (loss, flat or growth regime), the single step at a regime switch lands in
    one difference that the median ignores, and isolated outlier cycles touch
    at most two differences. NaN cycles are skipped.

    Parameters
    ----------
    per_cycle_mass_change:
        Mass change of each cycle in ng/cm². Needs at least 3 finite values.

    Returns
    -------
    float
        Noise sigma in ng/cm², or ``nan`` if there are too few finite values.
    """
    y = np.asarray(per_cycle_mass_change, dtype=np.float64)
    d = np.diff(y)  # NaN where either neighbour is NaN
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(d - np.median(d))))
    return _MAD_TO_SIGMA * mad / np.sqrt(2.0)


def find_growth_switch(
    per_cycle_mass_change: Sequence[float],
    cycle_times: Sequence[float] | None = None,
    *,
    n_consecutive: int = 5,
    noise_factor: float = 2.0,
) -> GrowthSwitchResult:
    """Find the first cycle after which a run keeps gaining mass.

    The switch is the first cycle that starts a run of ``n_consecutive``
    cycles whose mass change is all above ``noise_factor`` times the robust
    noise level (see :func:`estimate_cycle_noise`). "Above" is measured from
    zero: flat cycles scatter around zero and loss cycles are negative, so
    neither passes. A single outlier cycle cannot trigger a switch as long as
    ``n_consecutive`` is at least 2.

    Parameters
    ----------
    per_cycle_mass_change:
        Net mass change of each cycle in ng/cm², in cycle order. NaN marks a
        cycle with no usable value; it breaks a run.
    cycle_times:
        Start time of each cycle, same length. Optional; without it
        ``switch_time`` and ``incubation_time`` are ``None``.
    n_consecutive:
        Number of consecutive growing cycles required (default 5).
    noise_factor:
        Threshold in noise sigmas (default 2.0). Raise it for noisier runs.

    Returns
    -------
    GrowthSwitchResult

    Raises
    ------
    QCMPakError
        If ``n_consecutive`` is below 1, ``noise_factor`` is negative, or
        ``cycle_times`` does not match the series length.
    """
    if n_consecutive < 1:
        raise QCMPakError(f"n_consecutive must be at least 1, got {n_consecutive}.")
    if noise_factor < 0:
        raise QCMPakError(f"noise_factor must not be negative, got {noise_factor}.")
    y = np.asarray(per_cycle_mass_change, dtype=np.float64)
    times: NDArray[np.float64] | None = None
    if cycle_times is not None:
        times = np.asarray(cycle_times, dtype=np.float64)
        if times.shape != y.shape:
            raise QCMPakError(
                f"cycle_times has {times.size} values but the mass series has "
                f"{y.size}; pass one start time per cycle."
            )

    sigma = estimate_cycle_noise(y)
    notes: list[str] = []
    result = GrowthSwitchResult(
        found=False,
        switch_cycle=None,
        switch_time=None,
        incubation_time=None,
        growth_before=None,
        growth_after=None,
        noise_sigma=sigma,
        threshold=float("nan"),
        n_consecutive=n_consecutive,
        noise_factor=noise_factor,
        notes=notes,
    )
    if y.size < n_consecutive or not np.isfinite(sigma):
        notes.append(
            f"Only {y.size} cycles: too few to look for {n_consecutive} growing "
            "cycles in a row and estimate the noise."
        )
        return result

    threshold = noise_factor * sigma
    result.threshold = threshold
    growing = y > threshold  # NaN compares False, so it breaks a run
    # run_ok[i] is True when cycles i .. i+n-1 are all growing
    window = np.convolve(growing.astype(int), np.ones(n_consecutive, int), "valid")
    starts = np.flatnonzero(window == n_consecutive)

    if starts.size == 0:
        result.growth_before = float(np.nanmean(y)) if np.isfinite(y).any() else None
        if result.growth_before is not None and result.growth_before <= threshold:
            notes.append(
                "No growth switch: the run stays flat or loses mass "
                f"(mean {result.growth_before:.3g} ng/cm² per cycle, noise "
                f"threshold {threshold:.3g})."
            )
        else:
            notes.append(
                f"No growth switch: no {n_consecutive} consecutive cycles are "
                f"above the noise threshold ({threshold:.3g} ng/cm²), although "
                "the mean is positive. Growth may be too slow or too noisy to "
                "tell from noise."
            )
        return result

    k = int(starts[0])
    result.found = True
    result.switch_cycle = k
    if times is not None:
        result.switch_time = float(times[k])
        result.incubation_time = float(times[k] - times[0])
    if k > 0:
        before = y[:k]
        result.growth_before = (
            float(np.nanmean(before)) if np.isfinite(before).any() else None
        )
        if result.growth_before is not None:
            if result.growth_before < -threshold:
                notes.append("Mass was lost before the switch.")
            else:
                notes.append("No net growth before the switch (flat within noise).")
    else:
        notes.append("Growth from the first cycle: no incubation.")
    after = y[k:]
    result.growth_after = float(np.nanmean(after)) if np.isfinite(after).any() else None
    return result


def detect_growth_switch(
    result: AnalysisResult,
    *,
    n_consecutive: int = 5,
    noise_factor: float = 2.0,
) -> GrowthSwitchResult:
    """Find the growth switch of an analysed run.

    Uses :attr:`~qcm_pak._types.Cycle.net_mass_change` for each outer cycle
    (the pulse windows already detected and extracted, whichever pulse
    detection algorithm was used). A cycle with any pulse excluded during
    review has no reliable total and is treated as missing. Cycle start times
    are the time of each cycle's first detected pulse, in seconds.

    Parameters
    ----------
    result:
        Output of :func:`~qcm_pak.pipeline.run_analysis`.
    n_consecutive, noise_factor:
        See :func:`find_growth_switch`.

    Returns
    -------
    GrowthSwitchResult
    """
    mass_change = [
        float("nan")
        if any(s.excluded for scr in c.sub_cycle_runs for s in scr.steps)
        else c.net_mass_change
        for c in result.cycles
    ]
    onsets = result.cycle_index.step_onsets
    n_cycles = len(mass_change)
    times: list[float] | None = None
    if n_cycles and len(onsets) % n_cycles == 0:
        per_cycle = len(onsets) // n_cycles
        times = [
            float(result.mass_data.time[onsets[k * per_cycle][1]])
            for k in range(n_cycles)
        ]
    return find_growth_switch(
        mass_change, times, n_consecutive=n_consecutive, noise_factor=noise_factor
    )
