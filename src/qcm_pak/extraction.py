"""
Cycle extraction: slice the mass time series into per-step windows and apply
baseline correction, then assemble the three-level result hierarchy.

The extraction window for each step runs from its detected onset to the onset
of the *next* detected event (or end of data for the last event). A short
pre-pulse baseline window is used to anchor the corrected mass to zero at each
step onset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qcm_pak._types import (
    Cycle,
    CycleCollection,
    CycleIndex,
    MassDataset,
    StepResult,
    SubCycleRun,
)
from qcm_pak.parameters import ALDParameters
from qcm_pak.recipe import Recipe

# How many seconds before each step onset to average for the baseline
_BASELINE_PRE_PULSE_S = 2.0


def extract_cycles(
    data: MassDataset,
    index: CycleIndex,
    params: ALDParameters,
) -> CycleCollection:
    """Slice the mass time series into per-step windows with baseline correction.

    Parameters
    ----------
    data:
        Sauerbrey-converted mass dataset.
    index:
        Detected pulse onset positions from :func:`~qcm_pak.detection.detect_pulses`.
    params:
        Experiment parameters (carries the Recipe).

    Returns
    -------
    CycleCollection
        Full three-level hierarchy: Cycle → SubCycleRun → StepResult.
    """
    recipe = params.recipe
    onsets = index.step_onsets  # list of (step_name, array_idx)
    n = len(data.mass)
    dt = data.dt

    # Map (step_name → mass_effect) for quick lookup
    effect_map: dict[str, str] = {
        step.name: step.mass_effect
        for sc in recipe.sub_cycles
        for step in sc.steps
    }

    # Build StepResult objects from onset list
    # Each onset's window ends at the next onset (or end of array)
    step_results: list[StepResult] = []
    for i, (step_name, onset_idx) in enumerate(onsets):
        end_idx = onsets[i + 1][1] if i + 1 < len(onsets) else n
        step_results.append(
            _extract_step(
                data=data,
                step_name=step_name,
                mass_effect=effect_map.get(step_name, "any"),
                onset_idx=onset_idx,
                end_idx=end_idx,
                dt=dt,
                event_position=i,
                recipe=recipe,
            )
        )

    # Group StepResults into SubCycleRuns → Cycles
    cycles: list[Cycle] = []
    steps_per_outer = recipe.steps_per_cycle  # total events per outer repeat
    for outer in range(recipe.repeats):
        start = outer * steps_per_outer
        outer_steps = step_results[start : start + steps_per_outer]
        sub_cycle_runs = _group_sub_cycle_runs(outer_steps, outer, recipe)
        cycles.append(Cycle(cycle_number=outer, sub_cycle_runs=sub_cycle_runs))

    return CycleCollection(cycles=cycles, recipe=recipe)


def _extract_step(
    data: MassDataset,
    step_name: str,
    mass_effect: str,
    onset_idx: int,
    end_idx: int,
    dt: float,
    event_position: int,
    recipe: Recipe,) -> StepResult:
    """Extract and baseline-correct one step window."""

    mass_raw = data.mass[onset_idx:end_idx]
    time_abs = data.time[onset_idx:end_idx]
    time = time_abs - time_abs[0]   # relative to onset

    mass_corrected = _baseline_correct(
        mass=data.mass,
        onset_idx=onset_idx,
        end_idx=end_idx,
        dt=dt,
    )

    mass_change = float(mass_corrected[-1]) if len(mass_corrected) > 0 else 0.0

    # Determine position within recipe hierarchy from event_position
    sc_index, sc_run, s_index, outer = _position_in_recipe(
        event_position, recipe
    )

    return StepResult(
        step_name=step_name,
        mass_effect=mass_effect,  # type: ignore[arg-type]
        step_index=s_index,
        sub_cycle_index=sc_index,
        sub_cycle_run=sc_run,
        outer_cycle=outer,
        time=time.astype(np.float64),
        mass_raw=mass_raw.astype(np.float64),
        mass_corrected=mass_corrected.astype(np.float64),
        mass_change=mass_change,
    )


def _baseline_correct(
    mass: NDArray[np.float64],
    onset_idx: int,
    end_idx: int,
    dt: float,
) -> NDArray[np.float64]:
    """Subtract a pre-pulse baseline from the step window.

    The baseline is the mean of the samples immediately before the onset,
    within a short window (up to ``_BASELINE_PRE_PULSE_S`` seconds). This
    anchors ``mass_corrected[0] ≈ 0``.
    """
    n_baseline = max(1, int(_BASELINE_PRE_PULSE_S / dt))
    baseline_start = max(0, onset_idx - n_baseline)
    baseline = float(np.mean(mass[baseline_start:onset_idx])) if onset_idx > 0 else 0.0
    segment = mass[onset_idx:end_idx]
    return segment - baseline


def _position_in_recipe(
    event_position: int,
    recipe: Recipe,  # noqa: F821
) -> tuple[int, int, int, int]:
    """Map a flat event index to its position in the recipe hierarchy.

    Returns ``(sub_cycle_index, sub_cycle_run, step_index, outer_cycle)``.


    Parameters
    ----------
    event_position:
        0-based absolute position in the flat CycleIndex.step_onsets list.
    recipe:
        The Recipe used during detection.

    Returns
    -------
    tuple
        ``(sub_cycle_index, sub_cycle_run, step_index, outer_cycle)``
    """

    steps_per_outer = recipe.steps_per_cycle
    outer = event_position // steps_per_outer
    pos_in_outer = event_position % steps_per_outer

    for sc_idx, sub_cycle in enumerate(recipe.sub_cycles):
        for run in range(sub_cycle.repeats):
            for s_idx in range(len(sub_cycle.steps)):
                if pos_in_outer == 0:
                    return sc_idx, run, s_idx, outer
                pos_in_outer -= 1

    # Should never reach here if CycleIndex is valid
    raise RuntimeError(  # pragma: no cover
        f"event_position {event_position} exceeds recipe structure"
    )


def _group_sub_cycle_runs(
    outer_steps: list[StepResult],
    outer: int,
    recipe: Recipe,  # noqa: F821
) -> list[SubCycleRun]:
    """Group the StepResults for one outer cycle into SubCycleRun objects."""

    runs: list[SubCycleRun] = []
    pos = 0
    for sc_idx, sub_cycle in enumerate(recipe.sub_cycles):
        for run in range(sub_cycle.repeats):
            n_steps = len(sub_cycle.steps)
            run_steps = outer_steps[pos : pos + n_steps]
            runs.append(
                SubCycleRun(
                    sub_cycle_index=sc_idx,
                    run_number=run,
                    outer_cycle=outer,
                    steps=run_steps,
                )
            )
            pos += n_steps
    return runs
