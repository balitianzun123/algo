"""Product-callable rule-based EEG motor-imagery algorithm core."""

from .calibration import calibrate_cal10, calibrate_rest, calibrate_rest10, rest_loo_action_scores
from .inference import predict_trial
from .models import PersonalProfile, PredictionResult, RawTrialInput, TrialFeatures, TrialInput
from .preprocessing import preprocess_raw_trial

__all__ = [
    "PersonalProfile", "PredictionResult", "RawTrialInput", "TrialFeatures", "TrialInput",
    "calibrate_cal10", "calibrate_rest", "calibrate_rest10", "preprocess_raw_trial",
    "predict_trial", "rest_loo_action_scores",
]
