"""
Full analysis pipeline: load → Sauerbrey → detect → extract.

The :func:`run_analysis` function is the main convenience entry point.
Kinetics fitting (Langmuir, etch) is intentionally left to the user:
call :func:`~qcm_pak.kinetics.fit_langmuir` or :func:`~qcm_pak.kinetics.fit_etch`
on the returned :class:`~qcm_pak._types.AnalysisResult` as needed.
"""

from __future__ import annotations

from qcm_pak._types import AnalysisResult
from qcm_pak.detection import detect_pulses
from qcm_pak.extraction import extract_cycles
from qcm_pak.io import ColumnSpec, load_data
from qcm_pak.parameters import ALDParameters, DetectionParameters, SauerbreyConstants
from qcm_pak.sauerbrey import frequency_to_mass


def run_analysis(
    params: ALDParameters,
    spec: ColumnSpec,
    crystal: SauerbreyConstants | None = None,
    det_params: DetectionParameters | None = None,
) -> AnalysisResult:
    """Run the full QCM analysis pipeline.

    Steps
    -----
    1. Load raw frequency data with :func:`~qcm_pak.io.load_data`.
    2. Convert to areal mass density with :func:`~qcm_pak.sauerbrey.frequency_to_mass`.
    3. Detect pulse onsets with :func:`~qcm_pak.detection.detect_pulses`.
    4. Slice into per-step windows with :func:`~qcm_pak.extraction.extract_cycles`.

    Parameters
    ----------
    params:
        Links the input file to a :class:`~qcm_pak.recipe.Recipe`.
    spec:
        Column layout and unit declaration for the CSV file.
    crystal:
        QCM crystal physical constants. If ``None``, standard 5 MHz AT-cut
        quartz defaults are used.
    det_params:
        Pulse detection configuration. If ``None``, ``pelt_guided`` defaults
        are used.

    Returns
    -------
    AnalysisResult
        All intermediate and final results. Call
        :meth:`~qcm_pak._types.AnalysisResult.save` to write outputs to disk,
        or access ``.cycles``, ``.mass_data``, and
        ``.cycle_index`` directly for downstream computation.

    Raises
    ------
    DataLoadError
        If the file cannot be read or columns cannot be resolved.
    DetectionError
        If pulse detection cannot satisfy the recipe's expected event count.
    """
    dataset = load_data(params.input_file, spec)
    mass = frequency_to_mass(dataset, crystal)
    index = detect_pulses(mass, params, det_params)
    cycles = extract_cycles(mass, index, params)
    return AnalysisResult(
        cycles=cycles,
        mass_data=mass,
        cycle_index=index,
        params=params,
    )
