"""Tests for corrections.py — apply_corrections and excluded-step filtering."""

import numpy as np
import pytest

from qcm_pak._types import MassDataset, PulseCorrection
from qcm_pak.corrections import apply_corrections
from qcm_pak.detection import detect_pulses
from qcm_pak.extraction import extract_cycles
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def _synthetic_ald(
    n_cycles: int = 4,
    dt: float = 0.5,
    start_time: float = 10.0,
    step_dur: float = 30.0,
    mass_per_step: float = 1.5,
) -> tuple[MassDataset, Recipe]:
    rng = np.random.default_rng(0)
    total_s = start_time + 2 * step_dur * n_cycles + 5.0
    time = np.arange(0, total_s, dt)
    mass = np.zeros(len(time))

    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", pulse=dt, purge=step_dur - dt),
            PulseStep("B", pulse=dt, purge=step_dur - dt),
        ])],
        repeats=n_cycles,
        start_time=start_time,
    )

    cumulative = 0.0
    t_cur = start_time
    for _ in range(n_cycles):
        for _ in ("A", "B"):
            onset_idx = int(round(t_cur / dt))
            end_idx = int(round((t_cur + step_dur) / dt))
            cumulative += mass_per_step
            mass[onset_idx:end_idx] = cumulative
            t_cur += step_dur

    mass += rng.normal(0, 0.01, size=len(mass))
    return MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass), recipe


def _index_and_params(tmp_path, n_cycles=4):
    data, recipe = _synthetic_ald(n_cycles=n_cycles)
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    params = ALDParameters(input_file=dummy, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    return data, params, index


def test_confidence_populated(tmp_path) -> None:
    _, _, index = _index_and_params(tmp_path)
    assert len(index.confidence) == index.n_detected
    assert all(0.0 <= c <= 1.0 for c in index.confidence)


def test_apply_corrections_moves_onset(tmp_path) -> None:
    _, _, index = _index_and_params(tmp_path)
    original_idx = index.step_onsets[0][1]
    corrected = apply_corrections(
        index, [PulseCorrection(pulse_id=0, new_onset_idx=original_idx + 2)]
    )
    assert corrected.step_onsets[0][1] == original_idx + 2
    assert corrected.step_onsets[0][0] == index.step_onsets[0][0]
    # Untouched pulses stay the same
    assert corrected.step_onsets[1] == index.step_onsets[1]


def test_apply_corrections_out_of_range(tmp_path) -> None:
    _, _, index = _index_and_params(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        apply_corrections(index, [PulseCorrection(pulse_id=index.n_detected + 5)])


def test_apply_corrections_rejects_reordering(tmp_path) -> None:
    _, _, index = _index_and_params(tmp_path)
    # Push pulse 0 past pulse 1's onset — should be rejected
    bad_idx = index.step_onsets[1][1] + 10
    with pytest.raises(ValueError, match="chronological order"):
        apply_corrections(index, [PulseCorrection(pulse_id=0, new_onset_idx=bad_idx)])


def test_excluded_pulse_dropped_from_steps(tmp_path) -> None:
    data, params, index = _index_and_params(tmp_path)
    corrected = apply_corrections(index, [PulseCorrection(pulse_id=0, excluded=True)])
    cycles = extract_cycles(data, corrected, params)

    all_steps = cycles.steps(include_excluded=True)
    default_steps = cycles.steps()
    assert len(default_steps) == len(all_steps) - 1
    assert all_steps[0].excluded is True
    assert not any(s.excluded for s in default_steps)


def test_excluded_pulse_removed_from_kinetics_dataset(tmp_path) -> None:
    data, params, index = _index_and_params(tmp_path)
    # Exclude the first "A" step (pulse_id 0)
    corrected = apply_corrections(index, [PulseCorrection(pulse_id=0, excluded=True)])
    cycles = extract_cycles(data, corrected, params)

    a_steps_default = cycles.steps(name="A")
    a_steps_all = cycles.steps(name="A", include_excluded=True)
    assert len(a_steps_default) == len(a_steps_all) - 1
