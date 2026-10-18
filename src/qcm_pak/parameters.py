"""
Analysis configuration dataclasses.

Three dataclasses cover all user-facing configuration:

- ``ALDParameters``: links an input file to a Recipe.
- ``SauerbreyConstants``: physical properties of the QCM crystal.
- ``DetectionParameters``: tuning knobs for the pulse detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from qcm_pak.exceptions import DataLoadError
from qcm_pak.recipe import Recipe


@dataclass
class ALDParameters:
    """Links an input data file to a Recipe.

    This is intentionally thin — all experiment details live in the Recipe.

    Parameters
    ----------
    input_file:
        Path to the QCM CSV/TSV data file.
    recipe:
        The ALD/ALE recipe describing the expected pulse sequence.
    """

    input_file: str | Path
    recipe: Recipe

    def __post_init__(self) -> None:
        self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            raise DataLoadError(f"Input file not found: {self.input_file}")


@dataclass(frozen=True)
class SauerbreyConstants:
    """Physical properties of the QCM crystal used for frequency-to-mass conversion.

    Defaults correspond to a 6 MHz AT-cut quartz crystal at the fundamental
    frequency (C ≈ −12.27 ng/cm²/Hz). Override if you use a different crystal,
    e.g. ``SauerbreyConstants(fundamental_frequency=5.0e6)`` for 5 MHz
    (C ≈ −17.7 ng/cm²/Hz).

    Parameters
    ----------
    fundamental_frequency:
        Resonant frequency of the crystal in Hz.
    quartz_density:
        Density of quartz in g/cm³.
    shear_modulus:
        Shear modulus of quartz in g/(cm·s²).
    overtone:
        Harmonic number. 1 = fundamental, 3 = third harmonic, etc.

    Notes
    -----
    The Sauerbrey equation is: Δm = C · Δf
    where C = −√(ρ_q · μ_q) / (2 · f₀² · n)
    and the result is in ng/cm² per Hz (C is negative).
    """

    fundamental_frequency: float = 6.0e6   # Hz
    quartz_density: float = 2.648          # g/cm³
    shear_modulus: float = 2.947e11        # g/(cm·s²)
    overtone: int = 1

    @property
    def conversion_factor(self) -> float:
        """Sauerbrey constant C in ng/cm²/Hz (negative by convention)."""
        C = np.sqrt(self.quartz_density * self.shear_modulus) / (
            2.0 * self.fundamental_frequency**2 * self.overtone
        )
        return -C * 1e9   # g/cm² → ng/cm²


@dataclass
class DetectionParameters:
    """Algorithm tuning knobs for pulse detection.

    Default values suit typical ALD QCM datasets acquired at ~1–10 Hz on a
    5–6 MHz crystal. Adjust if your instrument has a significantly different
    sampling rate or noise floor.

    Parameters
    ----------
    method:
        ``"pelt_guided"`` (default): PELT changepoint detection filtered by
        recipe timing. Requires the ``ruptures`` library.
        ``"hybrid"``: Recipe-guided Savitzky-Golay derivative method ported
        from QCMPy 0.5. Use as a fallback if pelt_guided misfires.
    penalty:
        PELT penalty value controlling changepoint sensitivity. ``"auto"``
        estimates it from the noise floor using BIC.
    min_pulse_spacing:
        Minimum number of samples between consecutive detected pulses.
        Prevents double-detection of a single event.
    recipe_tolerance:
        Fractional tolerance (±) around the recipe-expected spacing when
        assigning PELT candidates to recipe steps. Default 0.3 = ±30%.
    smoothing_window:
        Savitzky-Golay smoothing window length in samples (hybrid only).
    savgol_polyorder:
        Savitzky-Golay polynomial order (hybrid only).
    derivative_threshold_a:
        Derivative threshold multiplier for precursor-A pulses (hybrid only).
    derivative_threshold_b:
        Derivative threshold multiplier for precursor-B pulses (hybrid only).
    adaptive_threshold_sigma:
        Number of standard deviations above baseline for adaptive threshold
        (hybrid only). Onset refinement uses the same multiple of the raw-mass
        noise to decide where the rise begins.
    refinement_window:
        Half-width in seconds of the window around each derivative peak in
        which the onset is refined back to the last sample on the pre-pulse
        baseline (hybrid only). ``0`` disables refinement and keeps the
        derivative peak, which lands late on fast or smoothed rises.
    baseline_window:
        Pre-pulse window in seconds used to estimate local baseline for
        adaptive threshold, and the quiet stretch that onset refinement fits
        its baseline line to (hybrid only).
    min_snr:
        Minimum signal-to-noise ratio for accepting a detected pulse
        (hybrid only).
    """

    method: Literal["pelt_guided", "hybrid"] = "pelt_guided"

    # pelt_guided options
    penalty: float | Literal["auto"] = "auto"
    min_pulse_spacing: int = 10
    recipe_tolerance: float = 0.3

    # hybrid options
    smoothing_window: int = 50
    savgol_polyorder: int = 3
    derivative_threshold_a: float = 3.0
    derivative_threshold_b: float = 1.0
    adaptive_threshold_sigma: float = 3.0
    refinement_window: float = 1.0
    baseline_window: float = 5.0
    min_snr: float = 2.0

    def __post_init__(self) -> None:
        if self.recipe_tolerance <= 0 or self.recipe_tolerance >= 1:
            raise ValueError(
                f"recipe_tolerance must be in (0, 1), got {self.recipe_tolerance}"
            )
        if self.min_pulse_spacing < 1:
            raise ValueError(
                f"min_pulse_spacing must be >= 1, got {self.min_pulse_spacing}"
            )
        if self.refinement_window < 0:
            raise ValueError(
                f"refinement_window must be >= 0 (0 disables refinement), "
                f"got {self.refinement_window}"
            )
