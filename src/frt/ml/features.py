from __future__ import annotations

import numpy as np

from frt.audio.analysis import compute_log_bins
from frt.config import SAMPLE_RATE

DEFAULT_HOP_LENGTH = 512


def extract_log_bin_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Return normalized log-bin features shaped [1, T, F]."""
    db, _, _ = compute_log_bins(
        audio,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    features = db.T.astype(np.float32, copy=False)
    mean = float(features.mean())
    std = float(features.std())
    if std > 1e-6:
        features = (features - mean) / std
    else:
        features = features - mean
    return np.expand_dims(np.ascontiguousarray(features), axis=0)
