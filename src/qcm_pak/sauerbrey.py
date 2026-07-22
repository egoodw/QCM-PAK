"""
Sauerbrey frequency-to-mass conversion.

The Sauerbrey equation relates a frequency shift Δf (Hz) to an areal mass
change Δm (g/cm²):

    Δm = C · Δf,   where C = −√(ρ_q · μ_q) / (2 · f₀² · n)

For a standard 5 MHz AT-cut quartz crystal at the fundamental harmonic,
C ≈ −17.7 ng/cm²/Hz (a frequency decrease corresponds to a mass increase).
"""

from __future__ import annotations

from qcm_pak._types import MassDataset, QCMDataset
from qcm_pak.parameters import SauerbreyConstants


def frequency_to_mass(
    data: QCMDataset,
    constants: SauerbreyConstants | None = None,
) -> MassDataset:
    """Convert a frequency time series to areal mass density via the Sauerbrey equation.

    The baseline is the first frequency point: Δf = f(t) − f(t₀), so
    ``mass[0] == 0`` by definition. A frequency decrease produces a positive
    (mass gain) value; a frequency increase produces a negative (mass loss)
    value.

    Parameters
    ----------
    data:
        Raw QCM dataset from :func:`~qcm_pak.io.load_data`.
    constants:
        Crystal physical constants. If ``None``, standard 5 MHz AT-cut
        quartz defaults are used.

    Returns
    -------
    MassDataset
        Same time and frequency arrays as ``data``, plus the areal mass
        array in ng/cm².
    """
    if constants is None:
        constants = SauerbreyConstants()

    C = constants.conversion_factor  # ng/cm²/Hz, negative

    delta_f = data.frequency - data.frequency[0]
    mass = C * delta_f  # ng/cm²; positive when frequency drops (mass gain)

    return MassDataset(
        time=data.time,
        frequency=data.frequency,
        mass=mass,
        temperature=data.temperature,
    )
