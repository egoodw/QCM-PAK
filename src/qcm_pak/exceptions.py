"""Typed exceptions for the QCM-PAK public API."""


class QCMPakError(Exception):
    """Base class for all QCM-PAK errors."""


class DataLoadError(QCMPakError):
    """Raised when a data file cannot be read or columns cannot be resolved."""


class DetectionError(QCMPakError):
    """Raised when pulse detection cannot satisfy the Recipe's expected event count."""


class ConvergenceError(QCMPakError):
    """Raised when a kinetics fit diverges and no usable result can be returned."""
