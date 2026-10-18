"""
Conversion between qcm_pak objects and plain, JSON-safe Python values.

Every public data class has a ``to_dict()`` method, and the configuration
classes (``Recipe``, ``SubCycle``, ``PulseStep``, ``ColumnSpec``,
``SauerbreyConstants``, ``DetectionParameters``, ``ALDParameters``) and
``PulseCorrection`` also have ``from_dict()``. Both are built on the two
helpers here, so every class follows the same rules:

- Field names become keys, nested objects become nested dicts.
- NumPy arrays become (nested) lists, NumPy scalars become ``int``, ``float``
  or ``bool``, ``Path`` becomes ``str`` and tuples become lists.
- NaN and infinite values become ``None``, so the result passes
  ``json.dumps(..., allow_nan=False)`` and any strict JSON parser.

Public API:

- :func:`to_jsonable` — convert any qcm_pak object, or a dict or list of
  them (e.g. ``{step: {model_id: LangmuirResult}}``), to JSON-safe values.
- :func:`dataclass_to_dict` — convert one data class field by field; what
  most ``to_dict()`` methods call.
- :func:`dataclass_kwargs` — check a dict against a class's fields before
  building it; used by every ``from_dict()``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:  # noqa: ANN401 (any JSON value)
    """Convert ``obj`` to plain values that ``json.dumps`` accepts.

    Parameters
    ----------
    obj:
        A qcm_pak object, NumPy array or scalar, ``Path``, or a dict, list or
        tuple containing any of these. A data class with its own
        ``to_dict()`` is converted with it.

    Returns
    -------
    Any
        ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` or ``None``.
        Dict keys become strings; NaN and infinity become ``None``.

    Raises
    ------
    TypeError
        If ``obj`` holds something JSON has no form for, such as a function.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biu" or (
            obj.dtype.kind == "f" and bool(np.isfinite(obj).all())
        ):
            return obj.tolist()   # fast path: already plain numbers
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        to_dict = getattr(obj, "to_dict", None)
        return to_dict() if callable(to_dict) else dataclass_to_dict(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    raise TypeError(
        f"Cannot convert a {type(obj).__name__} to JSON: {obj!r}. Convert it to "
        "a number, string, list or dict first."
    )


def dataclass_to_dict(obj: Any) -> dict[str, Any]:  # noqa: ANN401 (any data class)
    """Convert each field of data class instance ``obj`` with :func:`to_jsonable`."""
    return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}


def dataclass_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Check ``data`` against the fields of data class ``cls``.

    Missing optional fields are left out, so ``cls`` uses its defaults.

    Parameters
    ----------
    cls:
        The data class about to be built.
    data:
        Field values, as produced by ``cls.to_dict()``.

    Returns
    -------
    dict[str, Any]
        ``data`` as a new dict, ready to be passed as ``cls(**kwargs)``.

    Raises
    ------
    TypeError
        If ``data`` is not a mapping.
    ValueError
        If ``data`` has a key that is not a field of ``cls``, or lacks a
        required one. The message lists the valid field names.
    """
    name = cls.__name__
    if not isinstance(data, Mapping):
        raise TypeError(
            f"{name}.from_dict() needs a dict, got {type(data).__name__}."
        )
    cls_fields = fields(cls)
    names = [f.name for f in cls_fields]
    unknown = [k for k in data if k not in names]
    if unknown:
        raise ValueError(
            f"{name}.from_dict(): unknown field(s) {', '.join(map(repr, unknown))}. "
            f"Valid fields: {', '.join(names)}."
        )
    missing = [
        f.name for f in cls_fields
        if f.default is MISSING and f.default_factory is MISSING
        and f.name not in data
    ]
    if missing:
        raise ValueError(
            f"{name}.from_dict(): missing required field(s) "
            f"{', '.join(map(repr, missing))}."
        )
    return dict(data)
