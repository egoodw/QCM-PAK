"""
User-guided correction of detected pulse onsets.

Automatic detection (:mod:`qcm_pak.detection`) can misplace a pulse's onset —
its (0,0) baseline reference point — on noisy or drifting traces. This module
lets a caller (typically a UI that lets a person scan through detected pulses)
apply per-pulse overrides after the fact, without re-running detection.

Corrections only move onsets or flag pulses as excluded; they never remove
entries from :class:`~qcm_pak._types.CycleIndex.step_onsets`. Removing an
entry would shift every later pulse's position in the recipe hierarchy
(:meth:`qcm_pak.recipe.Recipe.event_position` assumes a flat index that
matches the recipe's step order 1:1). Call :func:`~qcm_pak.extraction.extract_cycles`
again on the corrected :class:`~qcm_pak._types.CycleIndex` to regenerate results.
"""

from __future__ import annotations

from qcm_pak._types import CycleIndex, PulseCorrection


def apply_corrections(
    index: CycleIndex, corrections: list[PulseCorrection]
) -> CycleIndex:
    """Apply user-reviewed onset overrides and exclusions to a CycleIndex.

    Parameters
    ----------
    index:
        The detected CycleIndex to correct.
    corrections:
        One :class:`~qcm_pak._types.PulseCorrection` per pulse to adjust.
        ``pulse_id`` indexes into ``index.step_onsets``.

    Returns
    -------
    CycleIndex
        A new CycleIndex with onsets moved and/or exclusion flags set.
        Confidence scores are left as detected — an onset a person just fixed
        is, by definition, no longer uncertain, but this function doesn't
        assume that; callers that want to reflect the review can bump
        confidence themselves.

    Raises
    ------
    ValueError
        If a ``pulse_id`` is out of range, or if the corrected onsets are no
        longer in chronological order (extraction relies on each step's
        window ending at the next onset).
    """
    n = len(index.step_onsets)
    onsets = list(index.step_onsets)
    excluded = list(index.excluded) if index.excluded else [False] * n

    for correction in corrections:
        if not (0 <= correction.pulse_id < n):
            raise ValueError(
                f"pulse_id {correction.pulse_id} out of range for CycleIndex "
                f"with {n} pulses"
            )
        name, onset_idx = onsets[correction.pulse_id]
        if correction.new_onset_idx is not None:
            onset_idx = correction.new_onset_idx
        onsets[correction.pulse_id] = (name, onset_idx)
        excluded[correction.pulse_id] = correction.excluded

    for i in range(1, n):
        if onsets[i][1] <= onsets[i - 1][1]:
            raise ValueError(
                f"Corrected onset at pulse_id={i} ({onsets[i][1]}) is not "
                f"after pulse_id={i - 1} ({onsets[i - 1][1]}); corrections "
                "must preserve chronological order."
            )

    return CycleIndex(
        step_onsets=onsets,
        recipe=index.recipe,
        confidence=list(index.confidence),
        excluded=excluded,
    )
