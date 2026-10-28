"""Tests for AnalysisResult.save() and AnalysisResult.to_bytes()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from qcm_pak._types import AnalysisResult, MassDataset  # noqa: E402
from qcm_pak.detection import detect_pulses  # noqa: E402
from qcm_pak.export import save_full_export  # noqa: E402
from qcm_pak.extraction import extract_cycles  # noqa: E402
from qcm_pak.parameters import ALDParameters, DetectionParameters  # noqa: E402
from qcm_pak.recipe import PulseStep, Recipe, SubCycle  # noqa: E402


def _analysis(tmp_path: Path, n_cycles: int = 4) -> AnalysisResult:
    dt, step_dur, start = 0.5, 20.0, 10.0
    time = np.arange(0, start + 2 * step_dur * n_cycles + 5.0, dt)
    mass = np.zeros(len(time))
    t_cur, cumulative = start, 0.0
    for _ in range(2 * n_cycles):
        cumulative += 1.5
        mass[int(round(t_cur / dt)):int(round((t_cur + step_dur) / dt))] = cumulative
        t_cur += step_dur
    mass += np.random.default_rng(0).normal(0, 0.01, len(mass))
    data = MassDataset(time=time, frequency=np.full(len(time), 6e6), mass=mass)
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("TMA", pulse=dt, purge=step_dur - dt),
            PulseStep("H2O", pulse=dt, purge=step_dur - dt),
        ])],
        repeats=n_cycles,
        start_time=start,
    )
    trace = tmp_path / "trace.csv"
    trace.write_text("t,f\n")
    params = ALDParameters(input_file=trace, recipe=recipe)
    index = detect_pulses(data, params, DetectionParameters(recipe_tolerance=0.4))
    return AnalysisResult(extract_cycles(data, index, params), data, index, params)


def _read_tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_to_bytes_returns_the_same_files_as_save(tmp_path: Path) -> None:
    result = _analysis(tmp_path)
    result.save(tmp_path / "out")
    saved = _read_tree(tmp_path / "out")

    in_memory = result.to_bytes()

    assert sorted(in_memory) == sorted(saved)
    for name, content in saved.items():
        assert in_memory[name] == content, name


def test_to_bytes_names_files_like_save(tmp_path: Path) -> None:
    files = _analysis(tmp_path).to_bytes()
    assert sorted(files) == [
        "data/cycle_data_H2O.tsv",
        "data/cycle_data_TMA.tsv",
        "figures/derivative_analysis.png",
        "figures/full_trace.png",
    ]
    assert files["figures/full_trace.png"].startswith(b"\x89PNG\r\n\x1a\n")
    header = files["data/cycle_data_TMA.tsv"].decode("utf-8").splitlines()[0]
    assert header.split("\t")[0] == "time_s"


def test_to_bytes_writes_nothing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _analysis(tmp_path)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    result.to_bytes()
    assert list(workdir.iterdir()) == []


def test_full_export_tsvs_match_to_bytes(tmp_path: Path) -> None:
    # save_full_export writes the same per-step TSVs through its own path;
    # the two must not drift apart.
    r = _analysis(tmp_path)
    save_full_export(tmp_path / "full", r.mass_data, r.cycle_index, r.cycles,
                     r.params, cycle_batch_size=2)
    files = r.to_bytes()
    for step in ("TMA", "H2O"):
        name = f"data/cycle_data_{step}.tsv"
        assert (tmp_path / "full" / name).read_bytes() == files[name]
