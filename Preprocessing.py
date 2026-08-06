import numpy as np
from scipy import signal as scipy_signal

BANDPASS_LOW = 0.5        # Hz
BANDPASS_HIGH = 40.0      # Hz

def bandpass_filter(data: np.ndarray, sfreq: float,
                    low: float = BANDPASS_LOW,
                    high: float = BANDPASS_HIGH) -> np.ndarray:
    nyq = sfreq / 2.0
    low_n  = low  / nyq
    high_n = high / nyq
    # Clip to valid range (avoids numerical issues with borderline Nyquist)
    high_n = min(high_n, 0.999)
    b, a = scipy_signal.butter(4, [low_n, high_n], btype="band")
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        try:
            filtered[ch] = scipy_signal.filtfilt(b, a, data[ch])
        except Exception:
            filtered[ch] = data[ch]   # fallback: keep raw if filter fails
    return filtered
 
 
def zscore_normalize(data: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[0]):
        ch_data = data[ch].astype(np.float64)
        mu  = np.mean(ch_data)
        std = np.std(ch_data) + 1e-8   # epsilon prevents /0
        ch_data = (ch_data - mu) / std
        ch_data = np.clip(ch_data, -10.0, 10.0)  # artifact suppression
        normalized[ch] = ch_data.astype(np.float32)
    return normalized
 
 
def preprocess(data: np.ndarray, sfreq: float,
               apply_filter: bool = True) -> np.ndarray:
    if apply_filter:
        data = bandpass_filter(data, sfreq)
    data = zscore_normalize(data)
    return data
 
 
