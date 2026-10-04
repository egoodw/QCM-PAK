"""
Registry of fittable kinetics models.

:func:`~qcm_pak.kinetics.fit_langmuir` and :func:`~qcm_pak.kinetics.fit_etch`
each implement two model variants directly. This registry gives callers (in
particular a UI presenting a selectable model list) a single place to
enumerate "what models exist" without hardcoding that list — each entry is
just a display name, a category, and the already-existing fit function bound
to the right variant.

Categories today are ``"growth"`` (ALD mass-gain kinetics) and ``"etch"``
(ALE mass-loss kinetics). ``"pressure"`` is reserved for a future model
family (e.g. reactor pressure-trace kinetics) — no models are registered
under it yet; it exists so the registry shape doesn't need to change when
that work starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

from qcm_pak._types import EtchResult, LangmuirResult
from qcm_pak.kinetics import (
    fit_etch,
    fit_etch_per_cycle,
    fit_langmuir,
    fit_langmuir_per_cycle,
)


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the model registry.

    Parameters
    ----------
    id:
        Stable identifier used by callers (e.g. a web API's
        ``selected_models`` list) to refer to this model.
    display_name:
        Human-readable name for UI presentation.
    category:
        ``"growth"``, ``"etch"``, or ``"pressure"`` (reserved, unpopulated).
    mass_effect:
        Which :class:`~qcm_pak.recipe.PulseStep` steps this model applies to
        — ``"gain"`` steps for growth models, ``"loss"`` steps for etch
        models.
    fit:
        Callable with the same signature as
        :func:`~qcm_pak.kinetics.fit_langmuir`/:func:`~qcm_pak.kinetics.fit_etch`
        minus ``model`` (already bound to this entry's variant): ``fit(cycles,
        step, sub_cycle=None, **kwargs) -> LangmuirResult | EtchResult``. This
        is the single ensemble fit averaged across every occurrence of the step.
    fit_per_cycle:
        Same signature and model binding, but dispatches to
        :func:`~qcm_pak.kinetics.fit_langmuir_per_cycle`/
        :func:`~qcm_pak.kinetics.fit_etch_per_cycle`: ``fit_per_cycle(cycles,
        step, sub_cycle=None, **kwargs) -> list[LangmuirResult | EtchResult]``,
        one result per occurrence — for drift/trend charts (k vs. cycle, etc.).
    """

    id: str
    display_name: str
    category: Literal["growth", "etch", "pressure"]
    mass_effect: Literal["gain", "loss"]
    fit: Callable[..., LangmuirResult | EtchResult]
    fit_per_cycle: Callable[..., list[LangmuirResult | EtchResult]]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "langmuir_mono": ModelSpec(
        id="langmuir_mono",
        display_name="Monomodal Langmuir",
        category="growth",
        mass_effect="gain",
        fit=partial(fit_langmuir, model="mono"),
        fit_per_cycle=partial(fit_langmuir_per_cycle, model="mono"),
    ),
    "langmuir_bi": ModelSpec(
        id="langmuir_bi",
        display_name="Bimodal Langmuir",
        category="growth",
        mass_effect="gain",
        fit=partial(fit_langmuir, model="bi"),
        fit_per_cycle=partial(fit_langmuir_per_cycle, model="bi"),
    ),
    "etch_saturating": ModelSpec(
        id="etch_saturating",
        display_name="Saturating Etch",
        category="etch",
        mass_effect="loss",
        fit=partial(fit_etch, model="saturating"),
        fit_per_cycle=partial(fit_etch_per_cycle, model="saturating"),
    ),
    "etch_linear": ModelSpec(
        id="etch_linear",
        display_name="Linear Etch",
        category="etch",
        mass_effect="loss",
        fit=partial(fit_etch, model="linear"),
        fit_per_cycle=partial(fit_etch_per_cycle, model="linear"),
    ),
}


def list_models(category: str | None = None) -> list[ModelSpec]:
    """List registered models, optionally filtered by category.

    Parameters
    ----------
    category:
        If given, return only models in this category (``"growth"``,
        ``"etch"``, or ``"pressure"``).

    Returns
    -------
    list[ModelSpec]
        Registered models, in registration order.
    """
    specs = list(MODEL_REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return specs
