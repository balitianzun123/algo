"""Algorithm-core exception hierarchy."""

# 自定义异常

class AlgorithmCoreError(Exception):
    """Base exception for product-callable algorithm-core failures."""


class InvalidTrialShapeError(AlgorithmCoreError):
    """Raised when an EEG trial array does not have shape [channels, samples]."""


class MissingChannelError(AlgorithmCoreError):
    """Raised when required channel names are absent."""


class DuplicateChannelError(MissingChannelError):
    """Raised when channel names are duplicated."""


class InvalidSamplingRateError(AlgorithmCoreError):
    """Raised when a sampling rate is missing, non-finite, or incompatible."""


class NonFiniteDataError(AlgorithmCoreError):
    """Raised when input data, powers, baselines, scores, or thresholds are not finite."""


class UnsupportedDataUnitError(AlgorithmCoreError):
    """Raised when raw EEG units are absent or incompatible with the legacy pipeline."""


class InvalidEventSampleError(AlgorithmCoreError):
    """Raised when an event sample is non-integer or outside the input signal."""


class RawTrialTooShortError(AlgorithmCoreError):
    """Raised when a raw trial cannot provide the requested event-aligned epoch."""


class PreprocessingError(AlgorithmCoreError):
    """Raised when legacy-compatible preprocessing cannot produce a valid trial."""


class InvalidRawTrialError(PreprocessingError):
    """Raised when a raw trial fails validation before filtering."""


class InvalidCalibrationCountError(AlgorithmCoreError):
    """Raised when a calibration set does not contain the required number of trials."""


class InvalidRequiredRestTrialsError(InvalidCalibrationCountError):
    """Raised when a requested backend Rest calibration count is invalid."""


class InvalidCalibrationLabelError(AlgorithmCoreError):
    """Raised when calibration trial labels do not match the requested calibration mode."""


class InvalidProfileError(AlgorithmCoreError):
    """Raised when a profile is incomplete or internally inconsistent."""


class ProfileAlreadyExistsError(AlgorithmCoreError):
    """Raised when saving a profile would overwrite an existing file."""
