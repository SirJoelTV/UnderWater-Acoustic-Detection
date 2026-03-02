import numpy as np
import librosa

from anomaly_detection.config import TARGET_SR

def logmel_features(
    segment,
    n_mels=64,
    n_fft=1024,
    hop_length=512
):
    """
    Compute log-mel energy features for one audio segment.
    Returns mean + std over time.
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=TARGET_SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Aggregate over time
    mean = np.mean(log_mel, axis=1)
    std = np.std(log_mel, axis=1)

    return np.concatenate([mean, std])


def logmel_dataset(segments):
    """
    Apply log-mel extraction to all segments.
    """
    features = [logmel_features(seg) for seg in segments]
    return np.asarray(features, dtype=np.float32)
