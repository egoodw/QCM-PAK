"""Tests for recipe.py — PulseStep, SubCycle, Recipe."""

import pytest

from qcm_pak.recipe import PulseStep, Recipe, SubCycle


def test_pulse_step_duration() -> None:
    step = PulseStep("TMA", pulse=0.1, purge=30.0)
    assert step.duration == pytest.approx(30.1)


def test_pulse_step_invalid_pulse() -> None:
    with pytest.raises(ValueError, match="pulse must be"):
        PulseStep("TMA", pulse=0.0, purge=30.0)


def test_pulse_step_invalid_purge() -> None:
    with pytest.raises(ValueError, match="purge must be"):
        PulseStep("TMA", pulse=0.1, purge=-1.0)


def test_sub_cycle_empty_steps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SubCycle(steps=[])


def test_sub_cycle_invalid_repeats() -> None:
    with pytest.raises(ValueError, match="repeats must be"):
        SubCycle(steps=[PulseStep("A", 0.1, 10)], repeats=0)


def test_sub_cycle_duration() -> None:
    sc = SubCycle(
        steps=[PulseStep("A", 0.1, 10.0), PulseStep("B", 0.1, 10.0)],
        repeats=3,
    )
    assert sc.duration == pytest.approx(20.2)
    assert sc.total_duration == pytest.approx(60.6)


def test_recipe_steps_per_cycle_binary() -> None:
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10)])],
        repeats=5,
    )
    assert recipe.steps_per_cycle == 2
    assert recipe.total_events == 10


def test_recipe_steps_per_cycle_super() -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10)], repeats=3),
            SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("C", 0.2, 10)], repeats=2),
        ],
        repeats=4,
    )
    assert recipe.steps_per_cycle == 10   # 3*2 + 2*2
    assert recipe.total_events == 40


def test_recipe_step_names_unique() -> None:
    recipe = Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("TMA", 0.1, 10), PulseStep("H2O", 0.1, 10)]),
            SubCycle(steps=[PulseStep("TMA", 0.1, 10), PulseStep("O3", 0.2, 10)]),
        ],
        repeats=1,
    )
    # TMA appears in both subcycles but should only appear once in step_names
    assert recipe.step_names() == ["TMA", "H2O", "O3"]


def test_recipe_invalid_start_time() -> None:
    with pytest.raises(ValueError, match="start_time must be"):
        Recipe(
            sub_cycles=[SubCycle(steps=[PulseStep("A", 0.1, 10)])],
            start_time=-1.0,
        )


def test_recipe_cycle_duration() -> None:
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", 0.1, 10.0),
            PulseStep("B", 0.1, 10.0),
        ])],
        repeats=1,
    )
    assert recipe.cycle_duration == pytest.approx(20.2)


def _super_cycle_recipe() -> Recipe:
    return Recipe(
        sub_cycles=[
            SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10)], repeats=2),
            SubCycle(steps=[PulseStep("A", 0.1, 10), PulseStep("C", 0.2, 10)], repeats=3),
        ],
        repeats=2,
    )


def test_event_position_matches_recipe_traversal() -> None:
    recipe = _super_cycle_recipe()
    expected = [
        (outer, sc_idx, run, s_idx)
        for outer in range(recipe.repeats)
        for sc_idx, sc in enumerate(recipe.sub_cycles)
        for run in range(sc.repeats)
        for s_idx in range(len(sc.steps))
    ]
    assert [recipe.event_position(i) for i in range(recipe.total_events)] == expected


def test_event_position_three_step_cycle() -> None:
    recipe = Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("A", 0.1, 10), PulseStep("B", 0.1, 10), PulseStep("C", 0.1, 10),
        ])],
        repeats=100,
    )
    # Event 5 is the 3rd step (C) of the 2nd outer cycle
    assert recipe.event_position(5) == (1, 0, 0, 2)
    assert recipe.event_position(299) == (99, 0, 0, 2)


def test_event_position_out_of_range() -> None:
    recipe = _super_cycle_recipe()
    with pytest.raises(IndexError):
        recipe.event_position(recipe.total_events)
    with pytest.raises(IndexError):
        recipe.event_position(-1)
