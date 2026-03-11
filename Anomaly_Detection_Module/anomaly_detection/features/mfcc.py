import numpy as np
import librosa
from Anomaly_Detection_Module.anomaly_detection.AE_config import TARGET_SR, N_MFCC, N_FFT, HOP_LENGTH


def mfcc_from_segment(segment):
    """
    Extract MFCCs from a 1-second audio segment and
    aggregate to a fixed-length vector (mean + std).
    """
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def mfcc_dataset(segments):
    """
    Apply MFCC extraction to all segments.
    """
    features = [mfcc_from_segment(seg) for seg in segments]
    return np.vstack(features)
