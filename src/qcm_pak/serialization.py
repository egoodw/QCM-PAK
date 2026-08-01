"""
JSON-safe serialization helpers shared across qcm_pak's dataclasses.

:func:`to_jsonable` recursively converts a dataclass graph (including nested
dataclasses, numpy arrays, and ``Path`` values) into plain dicts/lists/
primitives suitable for ``json.dumps``. Every public dataclass in qcm_pak
exposes a ``to_dict()`` method that is a thin wrapper around this function,
so the schema stays in one place regardless of which object is serialized.

:func:`known_fields` is a small helper for ``from_dict`` classmethods on flat
(non-nested) config dataclasses: it filters an incoming dict down to the
class's own field names so missing keys fall back to the dataclass's own
defaults instead of raising a ``TypeError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert ``obj`` into JSON-safe plain Python types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def known_fields(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Filter ``data`` down to keys matching ``cls``'s dataclass field names."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}
