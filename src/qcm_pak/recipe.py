"""
ALD/ALE experiment description.

Users build a Recipe from SubCycles and PulseSteps to describe their
deposition process. The Recipe drives pulse detection and extraction —
no analysis logic lives here.

Examples
--------
Simple A+B (100 cycles)::

    Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("TMA", pulse=0.1, purge=30.0),
            PulseStep("H2O", pulse=0.1, purge=30.0),
        ])],
        repeats=100,
        start_time=120.0,
    )

Super cycle [(A+B)*n + (A+C)*m] * l::

    Recipe(
        sub_cycles=[
            SubCycle(
                steps=[PulseStep("TMA", 0.1, 30), PulseStep("H2O", 0.1, 30)],
                repeats=n,
                label="AB growth",
            ),
            SubCycle(
                steps=[PulseStep("TMA", 0.1, 30), PulseStep("O3", 0.2, 30)],
                repeats=m,
                label="AC passivation",
            ),
        ],
        repeats=l,
        start_time=120.0,
    )

ALE with two loss steps::

    Recipe(
        sub_cycles=[SubCycle(steps=[
            PulseStep("HF",  pulse=0.1, purge=30.0, mass_effect="loss"),
            PulseStep("TMA", pulse=0.1, purge=30.0, mass_effect="loss"),
        ])],
        repeats=50,
        start_time=60.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PulseStep:
    """One precursor pulse followed by a purge.

    Parameters
    ----------
    name:
        Human-readable precursor name (e.g., ``"TMA"``, ``"H2O"``).
    pulse:
        Pulse duration in seconds.
    purge:
        Purge duration in seconds.
    mass_effect:
        Expected direction of mass change during this step.
        ``"gain"`` for ALD growth steps, ``"loss"`` for ALE etch steps,
        ``"any"`` when the sign is unknown or variable.
    """

    name: str
    pulse: float
    purge: float
    mass_effect: Literal["gain", "loss", "any"] = "gain"

    def __post_init__(self) -> None:
        if self.pulse <= 0:
            raise ValueError(
                f"PulseStep '{self.name}': pulse must be > 0, got {self.pulse}"
            )
        if self.purge < 0:
            raise ValueError(
                f"PulseStep '{self.name}': purge must be >= 0, got {self.purge}"
            )

    @property
    def duration(self) -> float:
        """Total step duration (pulse + purge) in seconds."""
        return self.pulse + self.purge


@dataclass
class SubCycle:
    """An ordered sequence of PulseSteps that repeats ``repeats`` times per outer cycle.

    Parameters
    ----------
    steps:
        Ordered list of pulse steps executed once per repeat.
    repeats:
        Number of times this sequence repeats within each outer Recipe cycle.
    label:
        Optional human-readable label (e.g., ``"AB growth segment"``).
    """

    steps: list[PulseStep]
    repeats: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("SubCycle must contain at least one PulseStep")
        if self.repeats < 1:
            raise ValueError(f"SubCycle repeats must be >= 1, got {self.repeats}")

    @property
    def duration(self) -> float:
        """Total duration of one SubCycle repeat in seconds."""
        return sum(s.duration for s in self.steps)

    @property
    def total_duration(self) -> float:
        """Total duration of all repeats of this SubCycle in seconds."""
        return self.duration * self.repeats


@dataclass
class Recipe:
    """Full description of an ALD or ALE experiment.

    Parameters
    ----------
    sub_cycles:
        Ordered list of SubCycles executed sequentially within each outer repeat.
    repeats:
        Number of outer cycles (what most ALD researchers call "number of cycles").
    start_time:
        Time in seconds at which the first pulse event begins. Used by the
        pulse detector to anchor the expected timeline.

    Notes
    -----
    For simple binary ALD (A+B), use one SubCycle with ``repeats=1`` and set
    ``Recipe.repeats`` to the desired cycle count.

    For super cycles ``[(A+B)*n + (A+C)*m] * l``, use two SubCycles with
    ``repeats=n`` and ``repeats=m`` respectively, and set ``Recipe.repeats=l``.
    """

    sub_cycles: list[SubCycle]
    repeats: int = 1
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.sub_cycles:
            raise ValueError("Recipe must contain at least one SubCycle")
        if self.repeats < 1:
            raise ValueError(f"Recipe repeats must be >= 1, got {self.repeats}")
        if self.start_time < 0:
            raise ValueError(f"Recipe start_time must be >= 0, got {self.start_time}")

    @property
    def cycle_duration(self) -> float:
        """Total duration of one outer repeat in seconds."""
        return sum(sc.total_duration for sc in self.sub_cycles)

    @property
    def total_duration(self) -> float:
        """Expected total experiment duration in seconds (from start_time)."""
        return self.start_time + self.cycle_duration * self.repeats

    @property
    def steps_per_cycle(self) -> int:
        """Total number of individual pulse events per outer repeat."""
        return sum(sc.repeats * len(sc.steps) for sc in self.sub_cycles)

    @property
    def total_events(self) -> int:
        """Total pulse events across the entire experiment."""
        return self.steps_per_cycle * self.repeats

    def event_position(self, event_index: int) -> tuple[int, int, int, int]:
        """Map a flat event index to its position in the recipe hierarchy.

        Parameters
        ----------
        event_index:
            0-based position in the chronological event sequence (the same
            order as ``CycleIndex.step_onsets``).

        Returns
        -------
        tuple
            ``(outer_cycle, sub_cycle_index, sub_cycle_run, step_index)``
        """
        if not 0 <= event_index < self.total_events:
            raise IndexError(
                f"event_index {event_index} out of range for a recipe with "
                f"{self.total_events} events"
            )
        outer, pos = divmod(event_index, self.steps_per_cycle)
        for sc_idx, sub_cycle in enumerate(self.sub_cycles):
            block = sub_cycle.repeats * len(sub_cycle.steps)
            if pos < block:
                run, s_idx = divmod(pos, len(sub_cycle.steps))
                return outer, sc_idx, run, s_idx
            pos -= block
        raise RuntimeError(  # pragma: no cover
            f"event_index {event_index} exceeds recipe structure"
        )

    def step_names(self) -> list[str]:
        """Unique step names across all SubCycles, in order of first appearance."""
        seen: list[str] = []
        for sc in self.sub_cycles:
            for s in sc.steps:
                if s.name not in seen:
                    seen.append(s.name)
        return seen
