import numpy as np
from anomaly_detection.config import TARGET_SR, WINDOW_SECONDS, OVERLAP

WINDOW_SIZE = int(TARGET_SR * WINDOW_SECONDS)
HOP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))



def segment_audio(audio):
    """
    Split a 1D audio signal into overlapping fixed-length segments.
    """
    segments = []

    if len(audio) < WINDOW_SIZE:
        return segments  # drop too-short clips

    for start in range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE):
        segment = audio[start:start + WINDOW_SIZE]
        segments.append(segment)

    return np.asarray(segments)


def segment_dataset(audio_list):
    """
    Apply segmentation to a list of audio signals.
    """
    all_segments = []

    for audio in audio_list:
        segs = segment_audio(audio)
        if len(segs) > 0:
            all_segments.append(segs)

    if len(all_segments) == 0:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)

    return np.vstack(all_segments)
