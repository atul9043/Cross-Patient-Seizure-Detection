import os
import warnings
import logging
 
import numpy as np

from DataLoading import find_recordings, load_annotations, load_edf
from FeatureExtraction import add_temporal_context, extract_features
from Labeling import label_segments
from Preprocessing import preprocess
from Segmentation import segment_signal
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 
WINDOW_SEC   = 1  
 
def process_recording(edf_path: str, csv_path: str,
                      apply_filter: bool = True,
                      window_sec: float = WINDOW_SEC
                      ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    try:
        log.info(f"Processing: {os.path.basename(edf_path)}")
 
        # Load
        data, sfreq, ch_mask = load_edf(edf_path)
        annotations = load_annotations(csv_path)
 
        raw_segments, segment_times = segment_signal(data, sfreq, window_sec)
 
        if len(raw_segments) == 0:
            log.warning(f"No segments extracted from {edf_path}")
            return None, None
 
        # Preprocess for all non-artifact features
        data_proc = preprocess(data, sfreq, apply_filter=apply_filter)
        proc_segments, _ = segment_signal(data_proc, sfreq, window_sec)
 
        # Label using segment times from raw (identical to processed times)
        labels = label_segments(segment_times, annotations, sfreq, window_sec)
 
        # Extract features:
        #   freq / stat / spatial → preprocessed segments
        #   artifact              → raw segments (correct scale)
        # Quick check — add this temporarily inside process_recording
        # just before features = extract_features(...)
        print(f"raw_segments dtype : {raw_segments.dtype}")
        print(f"raw_segments range : {raw_segments.min():.1f} to {raw_segments.max():.1f}")
        print(f"proc_segments range: {proc_segments.min():.1f} to {proc_segments.max():.1f}")
        # raw should show values in microvolts (e.g. -200 to 200)
        # proc should show z-scored values (-10 to 10)
        features = extract_features(proc_segments, sfreq,
                                    raw_segments=raw_segments)
 
        # Add temporal context
        features = add_temporal_context(features)
 
        n_seizure = np.sum(labels)
        log.info(f"  -> {len(labels)} segments | {n_seizure} seizure "
                 f"({100*n_seizure/len(labels):.1f}%)")
 
        return features, labels
 
    except Exception as e:
        log.error(f"Failed processing {edf_path}: {e}")
        return None, None
 
 
def count_seizure_seconds(pairs: list[dict]) -> float:

    total = 0.0
    for p in pairs:
        ann = load_annotations(p["csv"])
        if not ann.empty:
            for _, row in ann.iterrows():
                total += row["stop_time"] - row["start_time"]
    return total
 
 
def undersample_majority(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         target_seizure_ratio: float = 0.10,
                         random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.RandomState(random_state)
 
    seizure_idx     = np.where(y == 1)[0]
    non_seizure_idx = np.where(y == 0)[0]
 
    n_seizure = len(seizure_idx)
    if n_seizure == 0:
        raise RuntimeError("No seizure segments to undersample around.")
 
    # How many non-seizure segments do we need for the target ratio?
    # target = n_seizure / (n_seizure + n_non)  →  n_non = n_seizure * (1-target) / target
    n_non_target = int(n_seizure * (1.0 - target_seizure_ratio) / target_seizure_ratio)
    n_non_target = min(n_non_target, len(non_seizure_idx))  # can't exceed what we have
 
    # Sample non-seizure indices WITHOUT replacement
    sampled_non = rng.choice(non_seizure_idx, size=n_non_target, replace=False)
 
    # Combine and shuffle
    keep = np.concatenate([seizure_idx, sampled_non])
    rng.shuffle(keep)
 
    actual_ratio = n_seizure / len(keep)
    log.info(f"Undersampling: {len(non_seizure_idx)} → {n_non_target} non-seizure segments")
    log.info(f"After undersampling: {len(keep)} segments, "
             f"seizure rate = {100*actual_ratio:.1f}%  "
             f"(target was {100*target_seizure_ratio:.0f}%)")
 
    return X[keep], y[keep], groups[keep]
 
 
def build_dataset(dataset_root: str,
                  apply_filter: bool = True,
                  max_recordings: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterates over all recordings and builds the full (X, y, groups) dataset.
 
    groups : recording index for each segment.
    This is CRITICAL to prevent data leakage in cross-validation:
    segments from the same recording must not appear in both train and test.
 
    Returns
    -------
    X      : np.ndarray, shape (total_segments, n_features)
    y      : np.ndarray, shape (total_segments,)
    groups : np.ndarray, shape (total_segments,) — recording index per segment
    """
    pairs = find_recordings(dataset_root)
 
    # Shuffle before slicing so max_recordings gives a representative sample
    # rather than always the first N files (which may all be from one session).
    import random
    random.Random(42).shuffle(pairs)
 
    if max_recordings:
        pairs = pairs[:max_recordings]
 
    # Build a stable patient_id → integer mapping for GroupShuffleSplit.
    # String IDs work with GroupShuffleSplit too, but integer groups are faster
    # and safer across all sklearn/imblearn versions.
    all_patient_ids = sorted({p["patient_id"] for p in pairs})
    patient_to_int  = {pid: idx for idx, pid in enumerate(all_patient_ids)}
    log.info(f"Patients in this run: {all_patient_ids}")
 
    X_all, y_all, groups_all = [], [], []
 
    for pair in pairs:
        features, labels = process_recording(
            pair["edf"], pair["csv"], apply_filter=apply_filter
        )
        if features is None:
            continue
        patient_int = patient_to_int[pair["patient_id"]]
        X_all.append(features)
        y_all.append(labels)
        # Every segment from the same patient gets the same group integer.
        # This guarantees GroupShuffleSplit never splits a patient across
        # train and test — the core requirement for no data leakage.
        groups_all.append(np.full(len(labels), patient_int, dtype=np.int32))
 
    if not X_all:
        raise RuntimeError("No valid recordings found. Check dataset_root path.")
 
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(groups_all)
 
    # Replace NaN/Inf arising from degenerate segments
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
 
    n_patients   = len(np.unique(groups))
    n_recordings = len(X_all)
    log.info(f"\nDataset: {X.shape[0]} segments, {X.shape[1]} features")
    log.info(f"  Patients   : {n_patients}")
    log.info(f"  Recordings : {n_recordings}")
    log.info(f"  Seizure    : {np.sum(y)} ({100*np.mean(y):.2f}%)")
    log.info(f"  Non-seizure: {np.sum(y==0)}")
    log.info(f"  Class imbalance ratio: {np.sum(y==0) / (np.sum(y==1) + 1e-8):.1f}:1 (non-seizure:seizure)")
 
    return X, y, groups
 