"""
CSV/TSV data loading for QCM data files.

The only public surface is :class:`ColumnSpec` and :func:`load_data`. All
column resolution and time-unit conversion is handled internally.

Time mode summary
-----------------
``time_unit='seconds'|'milliseconds'|'minutes'|'hours'``
    Numeric column; multiplied by the appropriate factor to get seconds.
``time_unit='datetime'``
    ISO 8601 strings (``"2024-01-15T14:30:00"``); elapsed seconds from the
    first row.
``time_unit='clock'``
    Instrument wall-clock strings (``"1:20:39 PM"`` or ``"14:20:39"``);
    elapsed seconds from the first row. Handles midnight rollovers.
``dt=<float>`` (no ``time_col``)
    No time column; synthesise ``time = i * dt`` in seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qcm_pak._types import QCMDataset
from qcm_pak.exceptions import DataLoadError
from qcm_pak.serialization import known_fields, to_jsonable

_FREQ_FACTORS: dict[str, float] = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
}

_TIME_FACTORS: dict[str, float] = {
    "seconds": 1.0,
    "milliseconds": 1.0e-3,
    "microseconds": 1.0e-6,
    "minutes": 60.0,
    "hours": 3600.0,
}


@dataclass
class ColumnSpec:
    """Declares how to read columns from a QCM CSV/TSV file.

    Columns may be specified by integer index (0-based) or by string name
    (matched against the header row identified by ``header_rows``).

    Parameters
    ----------
    freq_col:
        Column containing crystal resonance frequency.
    time_col:
        Column containing the time/timestamp values. Mutually exclusive with
        ``dt``.
    dt:
        Fixed sampling interval in seconds. Use when the file has no time
        column. Mutually exclusive with ``time_col``.
    header_rows:
        Number of header rows to skip before the data begins. ``0`` means the
        first row is data. ``1`` means the first row is a header and column
        names are read from it.
    time_unit:
        Interpretation of the ``time_col`` values. Ignored when ``dt`` is set.
        See module docstring for details.
    freq_unit:
        Unit of the frequency column values. Converted to Hz on load.
    temp_col:
        Optional column containing temperature measurements (°C).
    delimiter:
        Column separator character. Default is comma.
    encoding:
        File encoding. Default is UTF-8. Use ``"latin-1"`` for files exported
        by some QCM instruments (e.g., EON).
    skiprows:
        Number of rows to skip at the top of the file before the header
        (or before data when ``header_rows=0``). Use this for files with
        instrument metadata headers (e.g., QSoft files with 8 metadata lines
        before the column header row).
    drop_duplicate_times:
        If ``True`` (default), rows with the same timestamp as the previous
        row are removed after loading. Many QCM instruments write multiple
        identical rows per unique measurement interval; keeping duplicates
        inflates the dataset and breaks the sampling-interval estimator.
    """

    freq_col: str | int
    time_col: str | int | None = None
    dt: float | None = None
    header_rows: int = 0
    time_unit: Literal[
        "seconds", "milliseconds", "microseconds",
        "minutes", "hours", "datetime", "clock",
    ] = "seconds"
    freq_unit: Literal["Hz", "kHz", "MHz"] = "Hz"
    temp_col: str | int | None = None
    delimiter: str = ","
    encoding: str = "utf-8"
    skiprows: int = 0
    drop_duplicate_times: bool = True

    def __post_init__(self) -> None:
        if self.time_col is None and self.dt is None:
            raise ValueError(
                "ColumnSpec requires either 'time_col' or 'dt' — provide one."
            )
        if self.time_col is not None and self.dt is not None:
            raise ValueError(
                "ColumnSpec: provide 'time_col' OR 'dt', not both."
            )
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.header_rows < 0:
            raise ValueError(f"header_rows must be >= 0, got {self.header_rows}")
        if self.skiprows < 0:
            raise ValueError(f"skiprows must be >= 0, got {self.skiprows}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return cast(dict[str, Any], to_jsonable(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColumnSpec:
        """Reconstruct a ColumnSpec from a dict produced by :meth:`to_dict`."""
        return cls(**known_fields(cls, data))


def load_data(path: str | Path, spec: ColumnSpec) -> QCMDataset:
    """Read a QCM data file and return a :class:`~qcm_pak._types.QCMDataset`.

    Parameters
    ----------
    path:
        Path to the CSV/TSV file.
    spec:
        Column layout and unit declaration.

    Returns
    -------
    QCMDataset
        Time in seconds (elapsed from first point), frequency in Hz.

    Raises
    ------
    DataLoadError
        If the file cannot be opened, a named column is not found, or the
        time values cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Data file not found: {path}")

    try:
        df = _read_raw(path, spec)
    except Exception as exc:
        raise DataLoadError(f"Failed to read {path}: {exc}") from exc

    # If time_col is provided, parse time first so we can deduplicate before
    # extracting other columns (duplicates in time → identical rows in all cols).
    if spec.dt is None:
        time_raw = _extract_column(df, spec.time_col, "time_col", path)
        time = _parse_time(time_raw, spec.time_unit, path)
        if spec.drop_duplicate_times:
            mask = np.concatenate(([True], np.diff(time) > 0))
            df = df.iloc[mask].reset_index(drop=True)
            time = time[mask]
    else:
        # dt mode: synthesise after extracting frequency (for length)
        pass

    freq_raw = _extract_column(df, spec.freq_col, "freq_col", path)
    frequency = freq_raw.to_numpy(dtype=np.float64) * _FREQ_FACTORS[spec.freq_unit]

    if spec.dt is not None:
        time = np.arange(len(frequency), dtype=np.float64) * spec.dt

    temperature: NDArray[np.float64] | None = None
    if spec.temp_col is not None:
        temp_raw = _extract_column(df, spec.temp_col, "temp_col", path)
        temperature = temp_raw.to_numpy(dtype=np.float64)

    try:
        return QCMDataset(
            time=time,
            frequency=frequency,
            temperature=temperature,
            source_path=path,
        )
    except ValueError as exc:
        raise DataLoadError(str(exc)) from exc


# ── Private helpers ────────────────────────────────────────────────────────────


def _read_raw(path: Path, spec: ColumnSpec) -> pd.DataFrame:
    """Read the CSV into a DataFrame using the correct skiprows/header logic."""
    if spec.header_rows == 0:
        return pd.read_csv(
            path,
            sep=spec.delimiter,
            header=None,
            skiprows=spec.skiprows,
            encoding=spec.encoding,
            engine="python",
        )
    return pd.read_csv(
        path,
        sep=spec.delimiter,
        header=list(range(spec.header_rows)),
        skiprows=spec.skiprows,
        encoding=spec.encoding,
        engine="python",
    )


def _extract_column(
    df: pd.DataFrame,
    col: str | int | None,
    field_name: str,
    path: Path,
) -> pd.Series:  # type: ignore[type-arg]
    """Resolve a column by name or 0-based integer index."""
    if col is None:
        raise DataLoadError(f"{field_name} is None (internal error)")
    if isinstance(col, int):
        if col >= len(df.columns):
            raise DataLoadError(
                f"{field_name}={col} is out of range; "
                f"file has {len(df.columns)} columns: {path}"
            )
        return df.iloc[:, col]
    # string column name
    if isinstance(df.columns, pd.MultiIndex):
        flat = [" ".join(str(c) for c in col_tuple).strip() for col_tuple in df.columns]
        if col not in flat:
            raise DataLoadError(
                f"{field_name}='{col}' not found in header. "
                f"Available columns: {flat}  [{path}]"
            )
        pos = flat.index(col)
        return df.iloc[:, pos]
    if col not in df.columns:
        raise DataLoadError(
            f"{field_name}='{col}' not found in header. "
            f"Available columns: {list(df.columns)}  [{path}]"
        )
    return df[col]


def _parse_time(
    series: pd.Series,  # type: ignore[type-arg]
    time_unit: str,
    path: Path,
) -> NDArray[np.float64]:
    """Convert a raw time column to elapsed seconds from the first point."""
    try:
        if time_unit in _TIME_FACTORS:
            raw = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
            elapsed = (raw - raw[0]) * _TIME_FACTORS[time_unit]
            return elapsed

        if time_unit == "datetime":
            parsed = pd.to_datetime(series)
            elapsed = (
                (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
            )
            return elapsed

        if time_unit == "clock":
            return _parse_clock(series)

    except Exception as exc:
        raise DataLoadError(
            f"Cannot parse time column with time_unit='{time_unit}': {exc}  [{path}]"
        ) from exc

    raise DataLoadError(f"Unknown time_unit='{time_unit}'")


def _parse_clock(series: pd.Series) -> NDArray[np.float64]:  # type: ignore[type-arg]
    """Parse instrument wall-clock strings into elapsed seconds.

    Handles 12-hour AM/PM (``"1:20:39 PM"``) and 24-hour (``"14:20:39"``).
    Correctly wraps midnight rollovers by detecting backward jumps.
    """
    import re

    _12H = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)", re.IGNORECASE)
    _24H = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})$")

    def _to_seconds(s: str) -> float:
        s = s.strip()
        m12 = _12H.match(s)
        if m12:
            h, mi, sec, period = m12.groups()
            h = int(h)
            if period.upper() == "PM" and h != 12:
                h += 12
            elif period.upper() == "AM" and h == 12:
                h = 0
            return h * 3600 + int(mi) * 60 + float(sec)
        m24 = _24H.match(s)
        if m24:
            h, mi, sec = m24.groups()
            return int(h) * 3600 + int(mi) * 60 + float(sec)
        raise DataLoadError(f"Cannot parse clock string: '{s}'")

    raw = np.array([_to_seconds(str(v)) for v in series], dtype=np.float64)

    # Handle midnight rollover: wherever time jumps backward by more than 6 hours
    rollover_s = 86400.0  # seconds in a day
    offsets = np.zeros(len(raw))
    accumulated = 0.0
    for i in range(1, len(raw)):
        if raw[i] < raw[i - 1] - 6 * 3600:
            accumulated += rollover_s
        offsets[i] = accumulated

    absolute = raw + offsets
    return (absolute - absolute[0]).astype(np.float64)
