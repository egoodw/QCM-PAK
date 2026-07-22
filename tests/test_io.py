"""Tests for io.py — ColumnSpec and load_data."""

import pathlib

import pytest

from qcm_pak.exceptions import DataLoadError
from qcm_pak.io import ColumnSpec, load_data

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_load_seconds_by_name() -> None:
    spec = ColumnSpec(freq_col="freq_hz", time_col="time_s", time_unit="seconds", header_rows=1)
    ds = load_data(FIXTURES / "sample_seconds.csv", spec)
    assert ds.time[0] == pytest.approx(0.0)
    assert ds.frequency[0] == pytest.approx(5_000_000.0)
    assert len(ds.time) == 10


def test_load_seconds_by_index() -> None:
    spec = ColumnSpec(freq_col=1, time_col=0, time_unit="seconds", header_rows=1)
    ds = load_data(FIXTURES / "sample_seconds.csv", spec)
    assert ds.time[1] == pytest.approx(0.5)


def test_load_minutes_conversion() -> None:
    spec = ColumnSpec(freq_col="freq_hz", time_col="time_min", time_unit="minutes", header_rows=1)
    ds = load_data(FIXTURES / "sample_minutes.csv", spec)
    # 0.5 min → 30 s, elapsed from 0 min
    assert ds.time[1] == pytest.approx(30.0)


def test_load_clock() -> None:
    spec = ColumnSpec(freq_col="freq_hz", time_col="time", time_unit="clock", header_rows=1)
    ds = load_data(FIXTURES / "sample_clock.csv", spec)
    assert ds.time[0] == pytest.approx(0.0)
    assert ds.time[1] == pytest.approx(1.0)
    assert ds.time[4] == pytest.approx(4.0)


def test_load_dt() -> None:
    spec = ColumnSpec(freq_col="freq_hz", dt=0.5, header_rows=1)
    ds = load_data(FIXTURES / "sample_dt.csv", spec)
    assert ds.time[0] == pytest.approx(0.0)
    assert ds.time[1] == pytest.approx(0.5)
    assert ds.time[4] == pytest.approx(2.0)


def test_time_elapsed_from_zero() -> None:
    spec = ColumnSpec(freq_col="freq_hz", time_col="time_s", time_unit="seconds", header_rows=1)
    ds = load_data(FIXTURES / "sample_seconds.csv", spec)
    assert ds.time[0] == pytest.approx(0.0)


def test_missing_file_raises() -> None:
    spec = ColumnSpec(freq_col="freq", time_col="time", time_unit="seconds")
    with pytest.raises(DataLoadError, match="not found"):
        load_data("nonexistent.csv", spec)


def test_missing_column_raises() -> None:
    spec = ColumnSpec(freq_col="WRONG_COL", time_col="time_s", time_unit="seconds", header_rows=1)
    with pytest.raises(DataLoadError, match="not found"):
        load_data(FIXTURES / "sample_seconds.csv", spec)


def test_column_spec_both_time_and_dt_raises() -> None:
    with pytest.raises(ValueError, match="time_col.*OR.*dt"):
        ColumnSpec(freq_col="f", time_col="t", dt=0.5)


def test_column_spec_neither_time_nor_dt_raises() -> None:
    with pytest.raises(ValueError, match="time_col.*or.*dt"):
        ColumnSpec(freq_col="f")


def test_source_path_stored() -> None:
    spec = ColumnSpec(freq_col="freq_hz", time_col="time_s", time_unit="seconds", header_rows=1)
    path = FIXTURES / "sample_seconds.csv"
    ds = load_data(path, spec)
    assert ds.source_path == path
