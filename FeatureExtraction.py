import warnings
import logging
 
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 

# Standard 10/20 montage channels we care about (WITHOUT reference suffixes).
# The loader strips "-REF", "-LE", "-AVG" before matching.
# If a recording has fewer, we pad with zeros. If it has more, we subset.
STANDARD_CHANNELS = [
    "EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ",
]
N_CHANNELS = len(STANDARD_CHANNELS)  # 19
 
# Channels that are never EEG and must be excluded before any processing.
# Matched as substrings (uppercase) after suffix stripping.
NON_EEG_SUBSTRINGS = [
    "IBI", "BURSTS", "SUPPR",   # quantitative EEG metrics (not raw signal)
    "EKG", "ECG",               # cardiac
    "LOC", "ROC",               # eye movement
    "A1", "A2",                 # ear reference electrodes
    "EMG",                      # muscle
]
 
# All TUH seizure-type label codes that mean "this segment is ictal".
# The original parser only checked for "seiz" which misses all of these.
TUH_SEIZURE_LABELS = {
    "cpsz",   # complex partial seizure
    "absz",   # absence seizure
    "tnsz",   # tonic seizure
    "mysz",   # myoclonic seizure
    "fnsz",   # focal non-specific seizure
    "gnsz",   # generalised non-specific seizure
    "spsz",   # simple partial seizure
    "tcsz",   # tonic-clonic seizure
    "seiz",   # generic label (older annotations)
}
 
# Frequency bands (Hz)
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}
 
WINDOW_SEC   = 1          # segment length in seconds
BANDPASS_LOW = 0.5        # Hz
BANDPASS_HIGH = 40.0      # Hz
MIN_SFREQ    = 128        # Hz — recordings below this are resampled up
DATASET_PATH = r"D:/Seizure_Dataset"  # <-- UPDATE THIS PATH

def band_power(channel_data: np.ndarray, sfreq: float,
               fmin: float, fmax: float) -> float:
    nperseg = min(len(channel_data), int(sfreq))   # 1-second sub-windows
    freqs, psd = scipy_signal.welch(channel_data.astype(np.float64),
                                    fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))   # area under PSD curve
 
 
def extract_frequency_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
    features = []
    for ch in range(segment.shape[0]):
        ch_data = segment[ch].astype(np.float64)
        total_power = band_power(ch_data, sfreq, 0.5, 40.0) + 1e-8
        for band, (fmin, fmax) in FREQ_BANDS.items():
            rel_power = band_power(ch_data, sfreq, fmin, fmax) / total_power
            features.append(rel_power)
    return np.array(features, dtype=np.float32)
 
 
def extract_statistical_features(segment: np.ndarray) -> np.ndarray:

    from scipy.stats import entropy as scipy_entropy
 
    features = []
    for ch in range(segment.shape[0]):
        ch_data = segment[ch].astype(np.float64)
        diff1   = np.diff(ch_data)
        diff2   = np.diff(diff1)
 
        # Basic stats
        ch_var  = np.var(ch_data) + 1e-8
        ch_mean = np.mean(ch_data)
        ch_kurt = float(np.nan_to_num(kurtosis(ch_data)))
        ch_skew = float(np.nan_to_num(skew(ch_data)))
 
        # Line length — seizure onset detector
        line_length = float(np.sum(np.abs(diff1)))
 
        # Hjorth parameters
        hjorth_activity   = float(ch_var)
        var_diff1         = np.var(diff1) + 1e-8
        hjorth_mobility   = float(np.sqrt(var_diff1 / ch_var))
        var_diff2         = np.var(diff2) + 1e-8
        mobility_diff     = float(np.sqrt(var_diff2 / var_diff1))
        hjorth_complexity = float(mobility_diff / (hjorth_mobility + 1e-8))
 
        # Spectral entropy
        nperseg = min(len(ch_data), 128)
        _, psd  = scipy_signal.welch(ch_data, nperseg=nperseg)
        psd_norm = psd / (psd.sum() + 1e-8)
        spec_entropy = float(scipy_entropy(psd_norm + 1e-12))
 
        # Zero crossing rate
        zcr = float(np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data))
 
        features.extend([
            ch_mean,
            ch_var,
            ch_kurt,
            ch_skew,
            line_length,
            hjorth_activity,
            hjorth_mobility,
            hjorth_complexity,
            spec_entropy,
            zcr,
        ])
 
    return np.array(np.nan_to_num(features), dtype=np.float32)
 
  
def extract_spatial_features(segment: np.ndarray) -> np.ndarray:

    # Correlation matrix: shape (N_CHANNELS, N_CHANNELS)
    # Use only non-zero channels (padded zero channels give spurious 0 correlation)
    active_mask = np.array([np.std(segment[ch]) > 1e-6 for ch in range(segment.shape[0])])
    active_data = segment[active_mask]
 
    if active_data.shape[0] < 2:
        return np.zeros(4, dtype=np.float32)
 
    corr_matrix = np.corrcoef(active_data)
    # Extract upper triangle (excluding diagonal) — avoids double-counting
    idx_upper = np.triu_indices_from(corr_matrix, k=1)
    corr_vals = corr_matrix[idx_upper]
    corr_vals = np.nan_to_num(corr_vals, nan=0.0)
 
    mean_corr   = float(np.mean(np.abs(corr_vals)))
    max_corr    = float(np.max(np.abs(corr_vals)))
    var_corr    = float(np.var(corr_vals))
    n_high_corr = int(np.sum(np.abs(corr_vals) > 0.7))
 
    return np.array([mean_corr, max_corr, var_corr, n_high_corr], dtype=np.float32)
 
 
 
def extract_artifact_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
 
    max_amp_ratios = []
    variances      = []
    hf_ratios      = []
 
    for ch in range(segment.shape[0]):
        ch_data = segment[ch].astype(np.float64)
        amp_abs    = np.abs(ch_data)
        median_amp = np.median(amp_abs) + 1e-8
        max_amp_ratios.append(np.max(amp_abs) / median_amp)
        variances.append(np.var(ch_data))
 
        # High frequency ratio — must run on RAW signal (see process_recording)
        total_pwr = band_power(ch_data, sfreq, 0.5, 40.0) + 1e-8
        hf_pwr    = band_power(ch_data, sfreq, 30.0, 40.0)
        hf_ratios.append(hf_pwr / total_pwr)
 
    # ZCR removed — now computed in extract_statistical_features to avoid
    # redundancy and to ensure it runs on preprocessed (not raw) data.
    features = [
        float(np.mean(max_amp_ratios)),   # mean amplitude spike (electrode pop)
        float(np.std(variances)),         # cross-channel variance spike (focal artifact)
        float(np.mean(hf_ratios)),        # high-frequency ratio (muscle artifact)
    ]
    return np.array(features, dtype=np.float32)
 
 
def extract_features_single(segment: np.ndarray, sfreq: float,
                            raw_segment: np.ndarray | None = None) -> np.ndarray:

    freq_feats    = extract_frequency_features(segment, sfreq)
    stat_feats    = extract_statistical_features(segment)
    spatial_feats = extract_spatial_features(segment)
 
    # Use raw signal for artifact features if available
    artifact_input = raw_segment if raw_segment is not None else segment
    artifact_feats = extract_artifact_features(artifact_input, sfreq)
 
    return np.concatenate([freq_feats, stat_feats, spatial_feats, artifact_feats])
 
 
def extract_features(segments: np.ndarray, sfreq: float,
                     raw_segments: np.ndarray | None = None) -> np.ndarray:

    all_features = []
    for i, seg in enumerate(segments):
        raw_seg = raw_segments[i] if raw_segments is not None else None
        all_features.append(extract_features_single(seg, sfreq, raw_seg))
    return np.array(all_features, dtype=np.float32)
 
 
 
def add_temporal_context(features: np.ndarray) -> np.ndarray:

    n, d = features.shape
    pad = np.zeros((1, d), dtype=np.float32)
 
    padded_next = np.concatenate([features, pad], axis=0)  # shape (n+1, d)
 
    context = np.concatenate([
        features,           # t     (current)
        padded_next[1:],    # t+1   (next)
    ], axis=1)
 
    log.info(f"Temporal context [curr|next]: {features.shape} -> {context.shape}")
    return context.astype(np.float32)
 
 
def get_feature_names(include_temporal: bool = True) -> list[str]:
    names = []
 
    # Frequency features — relative band power only
    for ch in range(N_CHANNELS):
        ch_name = STANDARD_CHANNELS[ch].replace("EEG ", "")
        for band in FREQ_BANDS:
            names.append(f"freq_{ch_name}_{band}_rel")
 
    # Statistical features — 10 per channel
    stat_names = ["mean", "var", "kurtosis", "skew",
                  "line_length", "hjorth_activity", "hjorth_mobility",
                  "hjorth_complexity", "spec_entropy", "zcr"]
    for ch in range(N_CHANNELS):
        ch_name = STANDARD_CHANNELS[ch].replace("EEG ", "")
        for s in stat_names:
            names.append(f"stat_{ch_name}_{s}")
 
    # Spatial features
    names += ["spatial_mean_corr", "spatial_max_corr",
              "spatial_var_corr", "spatial_n_high_corr"]
 
    # Artifact features — 3 (zcr removed)
    names += ["artifact_amp_ratio", "artifact_var_spike", "artifact_hf_ratio"]
 
    if include_temporal:
        base = names.copy()
        # curr | next only (prev dropped to reduce dimensionality)
        names = [f"curr_{n}" for n in base] + [f"next_{n}" for n in base]
 
    return names
 
 