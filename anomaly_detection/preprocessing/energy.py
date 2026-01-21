import numpy as np


def segment_energy(segments):
    """
    Compute RMS energy for each audio segment.
    segments: np.ndarray of shape (num_segments, num_samples)
    returns: np.ndarray of shape (num_segments,)
    """
    return np.mean(segments ** 2, axis=1)


def select_low_energy_segments(segments, percentile=30):
    """
    Select low-energy segments as ambient noise.
    percentile: lower percentile → stricter ambient selection
    """
    energies = segment_energy(segments)
    threshold = np.percentile(energies, percentile)

    mask = energies <= threshold
    return segments[mask]
