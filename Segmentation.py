import numpy as np

WINDOW_SEC   = 1 

def segment_signal(data: np.ndarray, sfreq: float,
                   window_sec: float = WINDOW_SEC) -> tuple[np.ndarray, np.ndarray]:

    window_samples = int(sfreq * window_sec)
    n_channels, n_times = data.shape
    n_segments = n_times // window_samples  # discard trailing incomplete window
 
    segments = np.zeros((n_segments, n_channels, window_samples), dtype=np.float32)
    segment_times = np.zeros(n_segments, dtype=np.float64)
 
    for i in range(n_segments):
        start = i * window_samples
        end   = start + window_samples
        segments[i] = data[:, start:end]
        segment_times[i] = start / sfreq
 
    return segments, segment_times
 
 
