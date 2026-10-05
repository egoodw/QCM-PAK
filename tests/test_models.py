"""Tests for models.py — the fittable-model registry."""

from qcm_pak.models import MODEL_REGISTRY, ModelSpec, list_models


def test_registry_has_expected_ids() -> None:
    assert set(MODEL_REGISTRY) == {
        "langmuir_mono",
        "langmuir_bi",
        "etch_saturating",
        "etch_linear",
    }


def test_list_models_all() -> None:
    specs = list_models()
    assert len(specs) == len(MODEL_REGISTRY)
    assert all(isinstance(s, ModelSpec) for s in specs)


def test_list_models_filtered_by_category() -> None:
    growth = list_models(category="growth")
    assert {s.id for s in growth} == {"langmuir_mono", "langmuir_bi"}

    etch = list_models(category="etch")
    assert {s.id for s in etch} == {"etch_saturating", "etch_linear"}

    pressure = list_models(category="pressure")
    assert pressure == []


def test_registered_fit_functions_are_callable_with_bound_model(tmp_path) -> None:
    import numpy as np

    from qcm_pak._types import Cycle, CycleCollection, StepResult, SubCycleRun
    from qcm_pak.recipe import PulseStep, Recipe, SubCycle

    t = np.linspace(0, 30, 50)
    mass_c = 5.0 * (1.0 - np.exp(-0.5 * t))
    step = StepResult(
        step_name="A",
        mass_effect="gain",
        step_index=0,
        sub_cycle_index=0,
        sub_cycle_run=0,
        outer_cycle=0,
        time=t,
        mass_raw=mass_c,
        mass_corrected=mass_c,
        mass_change=float(mass_c[-1]),
    )
    recipe = Recipe(sub_cycles=[SubCycle(steps=[PulseStep("A", 30.0, 0.1)])], repeats=1)
    cycles = CycleCollection(
        cycles=[Cycle(cycle_number=0, sub_cycle_runs=[
            SubCycleRun(sub_cycle_index=0, run_number=0, outer_cycle=0, steps=[step])
        ])],
        recipe=recipe,
    )

    result = MODEL_REGISTRY["langmuir_mono"].fit(cycles, "A")
    assert result.model == "mono"

    per_cycle = MODEL_REGISTRY["langmuir_mono"].fit_per_cycle(cycles, "A")
    assert len(per_cycle) == 1
    assert per_cycle[0].outer_cycle == 0
