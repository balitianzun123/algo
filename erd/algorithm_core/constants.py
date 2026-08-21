"""Fixed constants for the rule-based hierarchical three-class algorithm."""

# 通道、频带、时间窗、标签等固定参数

REQUIRED_CHANNELS: tuple[str, ...] = ("FC3", "FC4", "C3", "C4", "CP3", "CPz", "CP4", "Cz")
ACTION_CHANNELS: tuple[str, ...] = ("FC3", "FC4", "C3", "C4", "CP3", "CP4")
LEFT_ACTION_WEIGHTS: dict[str, float] = {
    "C3": 1.0,
    "CP3": 0.9,
    "FC3": 0.8,
}
RIGHT_ACTION_WEIGHTS: dict[str, float] = {
    "C4": 1.0,
    "CP4": 0.9,
    "FC4": 0.8,
}
LATERALITY_CHANNELS: tuple[str, str] = ("C3", "C4")

ANALYSIS_BAND_HZ: tuple[float, float] = (8.0, 30.0)
WINDOW_START_S: float = 0.5
WINDOW_END_S: float = 2.5
FILTER_ORDER: int = 4
BANDPOWER_METHOD: str = "filter_mean_square"

PREPROCESSING_NOTCH_HZ: float = 50.0
PREPROCESSING_BANDPASS_HZ: tuple[float, float] = (8.0, 30.0)
PREPROCESSING_FILTER_METHOD: str = "iir"
PREPROCESSING_IIR_ORDER: int = 4
PREPROCESSING_EPOCH_WINDOW_S: tuple[float, float] = (0.0, 4.0)
PREPROCESSING_OUTPUT_DTYPE: str = "float32"
SUPPORTED_RAW_DATA_UNITS: tuple[str, ...] = ("unknown_raw_npy_units_preserved",)

REST10_LATERALITY_THRESHOLD: float = 0.0
DEFAULT_REQUIRED_REST_TRIALS: int = 10
MIN_REQUIRED_REST_TRIALS: int = 3

REST_LABEL: str = "rest"
LEFT_LABEL: str = "left"
RIGHT_LABEL: str = "right"
ACTION_LABEL: str = "action"

PREDICTION_REST: int = 0
PREDICTION_ACTION: int = 1
PREDICTION_INVALID_QUALITY: int = -1

REST_ID: int = 0
LEFT_ID: int = 1
RIGHT_ID: int = 2

EEG_TRACE_ENABLED: bool = True
EEG_TRACE_DETAIL: bool = False

PROFILE_SCHEMA_VERSION: str = "2.1.0"
