"""Onset-aware guitar transcription utilities."""

from .decoder import DecodedNote, decode_predictions
from .features import DEFAULT_HOP_LENGTH, extract_log_bin_features
from .labels import TargetConfig, labels_to_targets
from .model import OnsetsAndFramesModel

__all__ = [
    "DEFAULT_HOP_LENGTH",
    "DecodedNote",
    "OnsetsAndFramesModel",
    "TargetConfig",
    "decode_predictions",
    "extract_log_bin_features",
    "labels_to_targets",
]
