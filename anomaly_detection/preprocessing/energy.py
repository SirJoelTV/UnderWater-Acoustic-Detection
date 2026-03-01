"""
Spectral Energy Analysis for Segment Classification

Uses spectrogram-based analysis instead of simple RMS energy
to better distinguish ambient noise from active sounds.
"""

import numpy as np
import librosa
from anomaly_detection.config import TARGET_SR


def spectral_energy(segments, sr=TARGET_SR, n_fft=1024, hop_length=512):
    """
    Compute spectral energy for each segment using spectrogram.

    This captures frequency-domain energy, which is better at
    distinguishing ambient noise from biological/mechanical sounds.

    Args:
        segments: np.ndarray of shape (num_segments, num_samples)
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length for STFT

    Returns:
        np.ndarray of shape (num_segments,) - spectral energy per segment
    """
    energies = []
    for seg in segments:
        # Compute spectrogram
        S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop_length))
        # Total spectral energy (sum of squared magnitudes)
        energy = np.mean(S ** 2)
        energies.append(energy)

    return np.array(energies)


def spectral_flatness_score(segments, sr=TARGET_SR, n_fft=1024, hop_length=512):
    """
    Compute spectral flatness for each segment.

    Spectral flatness measures how noise-like a sound is:
    - High flatness (~1.0) = noise-like (ambient)
    - Low flatness (~0.0) = tonal (whale calls, engine hum)

    Args:
        segments: np.ndarray of shape (num_segments, num_samples)

    Returns:
        np.ndarray of shape (num_segments,) - flatness score
    """
    flatness_scores = []
    for seg in segments:
        flatness = librosa.feature.spectral_flatness(
            y=seg, n_fft=n_fft, hop_length=hop_length
        )
        flatness_scores.append(np.mean(flatness))

    return np.array(flatness_scores)


def spectral_contrast_score(segments, sr=TARGET_SR, n_fft=1024, hop_length=512):
    """
    Compute spectral contrast for each segment.

    Measures the difference between peaks and valleys in the spectrum.
    - Low contrast = flat/ambient noise
    - High contrast = distinct sound sources (anomalies)

    Args:
        segments: np.ndarray of shape (num_segments, num_samples)

    Returns:
        np.ndarray of shape (num_segments,) - contrast score
    """
    contrast_scores = []
    for seg in segments:
        contrast = librosa.feature.spectral_contrast(
            y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        contrast_scores.append(np.mean(contrast))

    return np.array(contrast_scores)


def segment_activity_score(segments, sr=TARGET_SR):
    """
    Combined activity score using spectral energy + spectral contrast.

    Higher score = more likely to be an active sound (anomaly).
    Lower score = more likely to be ambient noise.

    Args:
        segments: np.ndarray of shape (num_segments, num_samples)

    Returns:
        np.ndarray of shape (num_segments,) - combined activity score
    """
    energy = spectral_energy(segments, sr)
    contrast = spectral_contrast_score(segments, sr)

    # Normalize each to [0, 1]
    if np.max(energy) > np.min(energy):
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    else:
        energy_norm = np.zeros_like(energy)

    if np.max(contrast) > np.min(contrast):
        contrast_norm = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast))
    else:
        contrast_norm = np.zeros_like(contrast)

    # Combined score: weighted average
    score = 0.5 * energy_norm + 0.5 * contrast_norm

    return score


# Keep backward compatibility
def segment_energy(segments):
    """
    Compute spectral energy for each segment (replaces old RMS energy).
    """
    return spectral_energy(segments)


def select_low_energy_segments(segments, percentile=30):
    """
    Select low-activity segments as ambient noise using spectral analysis.
    """
    scores = segment_activity_score(segments)
    threshold = np.percentile(scores, percentile)

    mask = scores <= threshold
    return segments[mask]
