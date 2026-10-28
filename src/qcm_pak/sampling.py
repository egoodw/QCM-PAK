"""
Sampling-rate analysis: how fast does the signal really update?

Some instruments write a new row at a high nominal rate but only measure a
new value every few rows (for example a 10 Hz reading stored at 100 Hz, so
each value appears about ten times in a row). Everything downstream that
counts samples (smoothing windows, minimum spacings, the search for a rise)
then works on a rate ten times too high. :func:`analyze_sampling` spots this
from the data, and :func:`repeat_runs` / :func:`collapse_repeats` let
detection work on the real measurements.

Detection is deliberately cautious: a run of equal values only counts as an
instrument repeat when the runs are evenly sized and almost every run holds a
different value. A flat noise-free stretch, a time gap or a quantised signal
does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qcm_pak.serialization import dataclass_to_dict

# Fewest runs of equal values needed before the pattern counts as evidence.
_MIN_RUNS = 10
# A run counts as "typical" when within this fraction of the median run length.
_RUN_TOLERANCE = 0.3
# Share of runs that must be typical, and of runs that must hold a new value.
_MIN_TYPICAL_SHARE = 0.6
_MIN_DISTINCT_SHARE = 0.8
# A run longer than this many typical runs is kept at both ends when collapsing,
# so a genuinely constant stretch or a time gap keeps its true extent.
_LONG_RUN_FACTOR = 2.0
# A gap is an interval this many times the median interval.
_GAP_FACTOR = 5.0


@dataclass(frozen=True)
class SamplingInfo:
    """How a trace is sampled, and whether its values repeat.

    Parameters
    ----------
    n_points:
        Number of samples.
    nominal_interval_s:
        Median positive time step between rows, in seconds (0 if unknown).
    nominal_rate_hz:
        ``1 / nominal_interval_s`` (0 if unknown).
    effective_interval_s:
        Median time between value changes when the values repeat in even
        runs; equal to the nominal interval otherwise.
    effective_rate_hz:
        ``1 / effective_interval_s`` (0 if unknown).
    repeats_detected:
        ``True`` when the file holds even runs of repeated values, i.e. the
        instrument measures slower than it writes rows.
    repeat_length:
        Typical number of rows per value (1 when nothing repeats).
    repeated_fraction:
        Share of rows whose value equals the previous row's.
    has_gaps:
        ``True`` when some time step is at least five times the median.
    """

    n_points: int
    nominal_interval_s: float
    nominal_rate_hz: float
    effective_interval_s: float
    effective_rate_hz: float
    repeats_detected: bool
    repeat_length: int
    repeated_fraction: float
    has_gaps: bool

    @property
    def message(self) -> str | None:
        """A sentence for the user when the stored rate is not the real one.

        ``None`` when the values do not repeat.
        """
        if not self.repeats_detected:
            return None
        return (
            f"Looks like {_rate_text(self.effective_rate_hz)} data stored at "
            f"{_rate_text(self.nominal_rate_hz)}: each value appears about "
            f"{self.repeat_length} times in a row. Detection works on the "
            f"{_rate_text(self.effective_rate_hz)} measurements."
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict, with ``message`` included."""
        out = dataclass_to_dict(self)
        out["message"] = self.message
        return out  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SamplingInfo:
        """Build a SamplingInfo from a dict such as :meth:`to_dict` returns."""
        known = {k: v for k, v in data.items() if k != "message"}
        return cls(**known)


def _rate_text(rate_hz: float) -> str:
    if rate_hz >= 1.0 or rate_hz == 0.0:
        return f"{rate_hz:.3g} Hz"
    return f"{1.0 / rate_hz:.3g} s per sample"


def median_interval(time: NDArray[np.float64]) -> float:
    """Median positive time step in seconds (0.0 when there is none)."""
    if len(time) < 2:
        return 0.0
    steps = np.diff(time)
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else 0.0


def repeat_runs(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Start index and length of every run of exactly equal consecutive values."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    starts = np.flatnonzero(np.concatenate(([True], np.diff(values) != 0)))
    lengths = np.diff(np.append(starts, n))
    return starts.astype(np.intp), lengths.astype(np.intp)


def _typical_run_length(
    values: NDArray[np.float64], starts: NDArray[np.intp], lengths: NDArray[np.intp]
) -> int:
    """Rows per value if the runs look like an instrument repeating readings, else 1."""
    if len(starts) < _MIN_RUNS:
        return 1
    typical = int(round(float(np.median(lengths))))
    if typical < 2:
        return 1
    near = np.abs(lengths - typical) <= max(1.0, _RUN_TOLERANCE * typical)
    if float(near.mean()) < _MIN_TYPICAL_SHARE:
        return 1
    run_values = values[starts]
    if np.unique(run_values).size < _MIN_DISTINCT_SHARE * len(starts):
        return 1    # a few levels revisited: piecewise-constant, not repeats
    return typical


def analyze_sampling(
    time: NDArray[np.float64], values: NDArray[np.float64]
) -> SamplingInfo:
    """Work out the nominal and effective sampling rate of a trace.

    Parameters
    ----------
    time:
        Sample times in seconds, non-decreasing.
    values:
        The measured signal (frequency or mass). Rows whose value equals the
        previous row's are treated as repeats when they come in even runs.

    Returns
    -------
    SamplingInfo
    """
    n = len(time)
    nominal = median_interval(time)
    nominal_rate = 1.0 / nominal if nominal > 0 else 0.0
    same = np.diff(values) == 0 if n > 1 else np.zeros(0, dtype=bool)
    repeated_fraction = float(same.mean()) if same.size else 0.0
    steps = np.diff(time) if n > 1 else np.zeros(0)
    has_gaps = bool(nominal > 0 and np.any(steps > _GAP_FACTOR * nominal))

    starts, lengths = repeat_runs(np.asarray(values, dtype=np.float64))
    typical = _typical_run_length(np.asarray(values, dtype=np.float64), starts, lengths)
    effective = nominal
    if typical > 1:
        run_steps = np.diff(time[starts])
        run_steps = run_steps[run_steps > 0]
        if run_steps.size:
            effective = float(np.median(run_steps))
        if effective < 1.5 * nominal:    # time stamps say the values do change
            typical, effective = 1, nominal
    return SamplingInfo(
        n_points=n,
        nominal_interval_s=nominal,
        nominal_rate_hz=nominal_rate,
        effective_interval_s=effective,
        effective_rate_hz=1.0 / effective if effective > 0 else 0.0,
        repeats_detected=typical > 1,
        repeat_length=typical,
        repeated_fraction=repeated_fraction,
        has_gaps=has_gaps,
    )


def collapse_repeats(
    values: NDArray[np.float64], repeat_length: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Choose the rows that carry the real measurements.

    The first row of every run of equal values is kept. A run much longer than
    usual (a flat stretch, or a time gap) keeps its last row too, so the
    collapsed trace does not smear a change over the whole stretch.

    Parameters
    ----------
    values:
        The signal the runs are found in.
    repeat_length:
        Typical rows per value, from :func:`analyze_sampling`.

    Returns
    -------
    keep, last_baseline
        ``keep`` are the original indices of the kept rows. ``last_baseline[j]``
        is the original index to report for an onset that detection places at
        kept row ``j``: the last row still showing the pre-onset value, which
        is the convention the onsets use (``mass_corrected[0] ≈ 0``).
    """
    starts, lengths = repeat_runs(values)
    ends = starts + lengths - 1
    long_run = lengths > _LONG_RUN_FACTOR * repeat_length
    keep: list[int] = []
    last: list[int] = []
    for s, e, is_long in zip(starts, ends, long_run, strict=True):
        if is_long:
            keep.extend((int(s), int(e)))
            last.extend((int(s), int(e)))
        else:
            keep.append(int(s))
            last.append(int(e))
    return np.asarray(keep, dtype=np.intp), np.asarray(last, dtype=np.intp)
