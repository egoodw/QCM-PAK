"""Typed exceptions for the QCM-PAK public API."""


class QCMPakError(Exception):
    """Base class for all QCM-PAK errors."""


class DataLoadError(QCMPakError):
    """Raised when a data file cannot be read or columns cannot be resolved."""


class DetectionError(QCMPakError):
    """Raised when pulse detection cannot satisfy the Recipe's expected event count."""


class RecipeMismatchError(DetectionError):
    """Raised when the recipe cannot fit the data, for example when it runs past
    the end of the trace.

    ``warnings`` holds the :class:`~qcm_pak._types.RecipeWarning` list, whose
    ``suggestion`` fields say what recipe would fit.
    """

    def __init__(self, message: str, warnings: list | None = None) -> None:
        super().__init__(message)
        self.warnings = list(warnings or [])


class ConvergenceError(QCMPakError):
    """Raised when a kinetics fit diverges and no usable result can be returned."""
