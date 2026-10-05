"""
QCM-PAK — QCM data analysis for ALD and ALE experiments.

Typical usage::

    import qcm_pak as qcm

    recipe = qcm.Recipe(
        sub_cycles=[qcm.SubCycle(steps=[
            qcm.PulseStep("TMA", pulse=0.1, purge=30.0),
            qcm.PulseStep("H2O", pulse=0.1, purge=30.0),
        ])],
        repeats=100,
        start_time=120.0,
    )
    spec = qcm.ColumnSpec(
        freq_col="Frequency [Hz]",
        time_col="Time",
        time_unit="clock",
        delimiter="\\t",
        encoding="latin-1",
        header_rows=1,
    )
    params = qcm.ALDParameters(input_file="run_042.tsv", recipe=recipe)
    result = qcm.run_analysis(params, spec)
    result.save("output/")

    langmuir = qcm.fit_langmuir(result.cycles, step="TMA", model="bi")
    print(f"R²={langmuir.r_squared:.4f}")
"""

from qcm_pak import visualization
from qcm_pak._types import (
    AnalysisResult,
    Cycle,
    CycleCollection,
    CycleIndex,
    EtchResult,
    LangmuirResult,
    MassDataset,
    PulseCorrection,
    QCMDataset,
    StepResult,
    SubCycleRun,
)
from qcm_pak._version import __version__
from qcm_pak.corrections import apply_corrections
from qcm_pak.detection import detect_pulses
from qcm_pak.exceptions import (
    ConvergenceError,
    DataLoadError,
    DetectionError,
    QCMPakError,
)
from qcm_pak.export import build_analysis_report, save_full_export
from qcm_pak.extraction import extract_cycles
from qcm_pak.io import ColumnSpec, load_data
from qcm_pak.kinetics import (
    fit_etch,
    fit_etch_per_cycle,
    fit_langmuir,
    fit_langmuir_per_cycle,
)
from qcm_pak.models import MODEL_REGISTRY, ModelSpec, list_models
from qcm_pak.parameters import (
    ALDParameters,
    DetectionParameters,
    SauerbreyConstants,
)
from qcm_pak.pipeline import run_analysis
from qcm_pak.recipe import PulseStep, Recipe, SubCycle
from qcm_pak.sauerbrey import frequency_to_mass

__all__ = [
    "__version__",
    # Exceptions
    "QCMPakError",
    "DataLoadError",
    "DetectionError",
    "ConvergenceError",
    # Recipe
    "PulseStep",
    "SubCycle",
    "Recipe",
    # Parameters
    "ALDParameters",
    "SauerbreyConstants",
    "DetectionParameters",
    # Types — data
    "QCMDataset",
    "MassDataset",
    "CycleIndex",
    "PulseCorrection",
    # Types — results
    "StepResult",
    "SubCycleRun",
    "Cycle",
    "CycleCollection",
    # Types — kinetics
    "LangmuirResult",
    "EtchResult",
    # Types — top level
    "AnalysisResult",
    # Functions
    "load_data",
    "ColumnSpec",
    "frequency_to_mass",
    "detect_pulses",
    "extract_cycles",
    "apply_corrections",
    "fit_langmuir",
    "fit_langmuir_per_cycle",
    "fit_etch",
    "fit_etch_per_cycle",
    "run_analysis",
    "save_full_export",
    "build_analysis_report",
    # Models
    "MODEL_REGISTRY",
    "ModelSpec",
    "list_models",
    # Submodule
    "visualization",
]
