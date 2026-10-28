"""Tests for sampling.py and the repeated-value handling in detection."""

import pathlib

import numpy as np
import pytest

from qcm_pak._types import MassDataset
from qcm_pak.detection import detect_pulses
from qcm_pak.io import ColumnSpec, load_data
from qcm_pak.parameters import ALDParameters, DetectionParameters
from qcm_pak.recipe import PulseStep, Recipe, SubCycle
from qcm_pak.sampling import SamplingInfo, analyze_sampling, collapse_repeats
from qcm_pak.sauerbrey import frequency_to_mass


def _noisy(n: int, dt: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return np.arange(n) * dt, np.cumsum(rng.normal(0, 0.1, n))


def test_plain_trace_is_not_flagged() -> None:
    time, values = _noisy(2000, 0.1)
    info = analyze_sampling(time, values)
    assert not info.repeats_detected
    assert info.repeat_length == 1
    assert info.message is None
    assert info.nominal_rate_hz == pytest.approx(10.0)
    assert info.effective_rate_hz == pytest.approx(10.0)


def test_ten_hz_stored_at_one_hundred_hz_is_found() -> None:
    _, slow = _noisy(500, 0.1)
    values = np.repeat(slow, 10)
    time = np.arange(len(values)) * 0.01
    info = analyze_sampling(time, values)
    assert info.repeats_detected
    assert info.repeat_length == 10
    assert info.nominal_rate_hz == pytest.approx(100.0)
    assert info.effective_rate_hz == pytest.approx(10.0)
    assert info.repeated_fraction == pytest.approx(0.9, abs=0.01)
    assert info.message is not None
    assert "10 Hz" in info.message and "100 Hz" in info.message


def test_uneven_run_lengths_still_count() -> None:
    rng = np.random.default_rng(1)
    _, slow = _noisy(600, 0.1)
    values = np.repeat(slow, rng.integers(9, 12, size=600))
    time = np.arange(len(values)) * 0.01
    info = analyze_sampling(time, values)
    assert info.repeats_detected
    assert info.effective_rate_hz == pytest.approx(10.0, rel=0.1)


def test_a_long_flat_stretch_is_not_repeats() -> None:
    time, values = _noisy(3000, 0.1)
    values[1000:2000] = values[1000]
    assert not analyze_sampling(time, values).repeats_detected


def test_quantised_signal_is_not_repeats() -> None:
    time = np.arange(3000) * 0.1
    values = np.round(np.sin(time / 30) * 3)    # a few levels, revisited
    assert not analyze_sampling(time, values).repeats_detected


def test_two_rows_per_value_is_half_the_stored_rate() -> None:
    rng = np.random.default_rng(2)
    values = np.repeat(np.cumsum(rng.normal(0, 1, 400)), 2)
    time = np.arange(len(values)) * 0.1
    info = analyze_sampling(time, values)
    assert info.repeats_detected
    assert info.effective_rate_hz == pytest.approx(5.0, rel=0.05)

def test_gaps_are_reported() -> None:
    time, values = _noisy(1000, 0.1)
    time[500:] += 60.0
    assert analyze_sampling(time, values).has_gaps


def test_empty_and_tiny_inputs_do_not_raise() -> None:
    for n in (0, 1, 2):
        info = analyze_sampling(np.arange(n, dtype=float), np.zeros(n))
        assert not info.repeats_detected


def test_sampling_info_round_trips() -> None:
    _, slow = _noisy(500, 0.1)
    values = np.repeat(slow, 10)
    info = analyze_sampling(np.arange(len(values)) * 0.01, values)
    data = info.to_dict()
    assert data["message"] == info.message
    assert SamplingInfo.from_dict(data) == info


def test_collapse_keeps_first_row_of_each_run_and_maps_to_last() -> None:
    values = np.repeat([1.0, 2.0, 3.0, 4.0], 5)
    keep, last = collapse_repeats(values, repeat_length=5)
    assert keep.tolist() == [0, 5, 10, 15]
    assert last.tolist() == [4, 9, 14, 19]


def test_collapse_keeps_both_ends_of_a_long_run() -> None:
    values = np.concatenate([np.repeat([1.0, 2.0], 5), np.full(100, 3.0), [4.0]])
    keep, last = collapse_repeats(values, repeat_length=5)
    assert keep.tolist() == [0, 5, 10, 109, 110]
    assert last.tolist() == [4, 9, 10, 109, 110]


# ── detection on repeated data ────────────────────────────────────────────────


def _stair_ald(
    repeat: int, n_cycles: int = 6
) -> tuple[MassDataset, Recipe, list[float], np.ndarray]:
    """A 10 Hz A/B trace written as ``repeat`` rows per measurement."""
    rng = np.random.default_rng(7)
    dt = 0.1
    cycle = 40.0
    start = 10.0
    time = np.arange(0, start + cycle * n_cycles + 10, dt)
    mass = np.zeros(len(time))
    onsets_t = []
    level = 0.0
    t_cur = start
    for _ in range(n_cycles):
        for _name in ("A", "B"):
            onsets_t.append(t_cur)
            level += 1.5
            mass[int(round(t_cur / dt)):] = level
            t_cur += cycle / 2
        # the surface etches back a little between cycles
    mass += rng.normal(0, 0.02, len(mass))
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 19.9), PulseStep("B", 0.1, 19.9)])
        ],
        repeats=n_cycles,
        start_time=start,
    )
    slow_time = time
    time = np.arange(len(mass) * repeat) * (dt / repeat)
    mass = np.repeat(mass, repeat)
    data = MassDataset(time=time, frequency=np.full(len(time), 5e6), mass=mass)
    return data, recipe, onsets_t, slow_time


@pytest.mark.parametrize("method", ["pelt_guided", "hybrid"])
def test_stair_stepped_data_gives_the_same_onsets_as_plain_data(
    method: str, tmp_path: pathlib.Path
) -> None:
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    plain, recipe, onsets_t, _ = _stair_ald(1)
    stairs, _, _, _ = _stair_ald(10)
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method=method, recipe_tolerance=0.4)
    a = detect_pulses(plain, params, det)
    b = detect_pulses(stairs, params, det)
    t_plain = [plain.time[i] for _, i in a.step_onsets]
    t_stair = [stairs.time[i] for _, i in b.step_onsets]
    assert np.max(np.abs(np.array(t_plain) - np.array(t_stair))) <= 0.35
    assert [i for _, i in b.step_onsets] == sorted({i for _, i in b.step_onsets})


def test_collapse_can_be_switched_off(tmp_path: pathlib.Path) -> None:
    dummy = tmp_path / "d.csv"
    dummy.write_text("t,f\n")
    stairs, recipe, _, _ = _stair_ald(10)
    params = ALDParameters(input_file=dummy, recipe=recipe)
    det = DetectionParameters(method="pelt_guided", recipe_tolerance=0.4,
                              collapse_repeats=False)
    assert detect_pulses(stairs, params, det).n_detected == recipe.total_events


def test_load_data_reports_the_effective_rate(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(3)
    slow = 5e6 - np.cumsum(rng.normal(0, 0.5, 400))
    freq = np.repeat(slow, 10)
    path = tmp_path / "stair.csv"
    lines = ["t,f"] + [f"{i * 0.01:.2f},{float(f)!r}" for i, f in enumerate(freq)]
    path.write_text("\n".join(lines) + "\n")
    spec = ColumnSpec(freq_col="f", time_col="t", header_rows=1)
    ds = load_data(path, spec)
    assert ds.sampling is not None
    assert ds.sampling.repeats_detected
    assert ds.sampling.effective_rate_hz == pytest.approx(10.0)
    mass = frequency_to_mass(ds)
    assert mass.sampling == ds.sampling
    assert "sampling" in ds.to_dict()


def test_detection_parameters_validate_rate_settings() -> None:
    with pytest.raises(ValueError):
        DetectionParameters(smoothing_window_s=0.0)
    with pytest.raises(ValueError):
        DetectionParameters(method="nonsense")    # type: ignore[arg-type]
