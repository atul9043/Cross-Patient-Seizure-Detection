import os
import glob
import warnings
import logging
 
import numpy as np
import pandas as pd
import mne
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

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

MIN_SFREQ    = 128  
 


def get_patient_id(edf_path: str, dataset_root: str) -> str:
    # Resolve both paths to eliminate symlinks / relative components
    abs_edf  = os.path.realpath(edf_path)
    abs_root = os.path.realpath(dataset_root)
 
    # Get the relative path from root, e.g. "patient_001/session_1/rec.edf"
    rel = os.path.relpath(abs_edf, abs_root)
 
    # First component of the relative path is always the patient folder
    patient_id = rel.split(os.sep)[0]
    return patient_id
 
 
def find_recordings(dataset_root: str) -> list[dict]:
    pairs = []
    for edf_path in glob.glob(os.path.join(dataset_root, "**", "*.edf"), recursive=True):
        csv_path = edf_path.replace(".edf", ".csv")
        if os.path.exists(csv_path):
            patient_id = get_patient_id(edf_path, dataset_root)
            pairs.append({"edf": edf_path, "csv": csv_path, "patient_id": patient_id})
        else:
            log.warning(f"No CSV found for {edf_path} — skipping.")
 
    unique_patients = len({p["patient_id"] for p in pairs})
    log.info(f"Found {len(pairs)} valid EDF+CSV pairs across {unique_patients} patient(s).")
    return pairs
 
 
def _normalise_ch_name(raw_name: str) -> str:
    name = raw_name.strip().upper()
 
    # Step 2: strip reference suffixes
    for suffix in ("-REF", "-LE", "-AVG", "-A1", "-A2"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
 
    # Step 3/4: ensure "EEG " prefix for scalp electrodes only
    if not name.startswith("EEG "):
        # Non-EEG channels (IBI, BURSTS, EKG …) keep their raw name so that
        # _is_non_eeg() can match and reject them correctly.
        is_non_eeg = any(s in name for s in NON_EEG_SUBSTRINGS)
        if not is_non_eeg:
            # bare name like "FP1" or "C3" — add the standard prefix
            bare = name.replace("EEG", "").strip()
            name = "EEG " + bare if bare else name
 
    return name
 
 
def _is_non_eeg(normalised_name: str) -> bool:
    """Returns True if this channel should be excluded from EEG processing."""
    upper = normalised_name.upper()
    return any(s in upper for s in NON_EEG_SUBSTRINGS)
 
 
def load_edf(edf_path: str) -> tuple[np.ndarray, float, list[bool]]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
 
    # Resample if sfreq is very low (ensures consistent window_samples)
    if sfreq < MIN_SFREQ:
        raw.resample(MIN_SFREQ, verbose=False)
        sfreq = float(MIN_SFREQ)
 
    # Build normalised-name → original-index map, skipping non-EEG channels
    norm_to_idx: dict[str, int] = {}
    for orig_idx, orig_name in enumerate(raw.ch_names):
        norm = _normalise_ch_name(orig_name)
        if _is_non_eeg(norm):
            continue                  # skip IBI, BURSTS, EKG …
        norm_to_idx[norm] = orig_idx
 
    n_times = raw.n_times
    aligned = np.zeros((N_CHANNELS, n_times), dtype=np.float32)
    ch_mask = []
 
    for slot_idx, std_ch in enumerate(STANDARD_CHANNELS):
        if std_ch in norm_to_idx:
            src_idx = norm_to_idx[std_ch]
            aligned[slot_idx] = raw.get_data(picks=[src_idx])[0].astype(np.float32)
            ch_mask.append(True)
        else:
            # Channel absent — zero-padded; spatial features will ignore it
            ch_mask.append(False)
 
    present = sum(ch_mask)
    log.info(f"  Channels: {present}/{N_CHANNELS} standard EEG channels found "
             f"(excluded {len(raw.ch_names) - len(norm_to_idx)} non-EEG channels)")
 
    return aligned, float(sfreq), ch_mask
 
 
def load_annotations(csv_path: str) -> pd.DataFrame:
    try:
        header_line_idx = None
        with open(csv_path, "r", errors="replace") as fh:
            for line_idx, line in enumerate(fh):
                lower = line.lower()
                has_time  = any(w in lower for w in ("start", "stop", "time"))
                has_label = any(w in lower for w in ("label", "type", "channel"))
                if has_time and has_label:
                    header_line_idx = line_idx
                    break
 
        if header_line_idx is None:
            log.warning(f"Could not find header row in {csv_path} "
                        "— treating as no-seizure recording.")
            return pd.DataFrame(columns=["start_time", "stop_time"])
 
        df = pd.read_csv(csv_path, skiprows=header_line_idx)
        df.columns = [c.strip().lower() for c in df.columns]
 
        rename = {}
        for col in df.columns:
            if "start" in col and "start_time" not in df.columns:
                rename[col] = "start_time"
            elif ("stop" in col or "end" in col) and "stop_time" not in df.columns:
                rename[col] = "stop_time"
            elif ("label" in col or "type" in col) and "label" not in df.columns:
                rename[col] = "label"
        df = df.rename(columns=rename)
 
        required = {"start_time", "stop_time", "label"}
        if not required.issubset(df.columns):
            log.warning(f"CSV {csv_path} missing columns {required - set(df.columns)} "
                        "— treating as no-seizure recording.")
            return pd.DataFrame(columns=["start_time", "stop_time"])
 
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["label"].isin(TUH_SEIZURE_LABELS)].copy()
 
        if df.empty:
            log.info(f"  No seizure annotations in {csv_path}")
            return pd.DataFrame(columns=["start_time", "stop_time"])
 
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["stop_time"]  = pd.to_numeric(df["stop_time"],  errors="coerce")
        df = df.dropna(subset=["start_time", "stop_time"])
        
        # Sort by start_time, sweep and merge any overlapping windows.
        # E.g. T6-O2 at 1.96s and FP1-F7 at 21.94s both belong to one seizure
        # → merged interval: 1.96s – 65.37s
        df = df.sort_values("start_time").reset_index(drop=True)
        merged: list[dict] = []
        cur_start = float(df.iloc[0]["start_time"])
        cur_stop  = float(df.iloc[0]["stop_time"])
 
        for _, row in df.iloc[1:].iterrows():
            rs, re = float(row["start_time"]), float(row["stop_time"])
            if rs <= cur_stop:
                cur_stop = max(cur_stop, re)
            else:
                merged.append({"start_time": cur_start, "stop_time": cur_stop})
                cur_start, cur_stop = rs, re
        merged.append({"start_time": cur_start, "stop_time": cur_stop})
 
        result = pd.DataFrame(merged)
        log.info(f"  Annotations: {len(df)} channel-level rows -> "
                 f"{len(result)} recording-level seizure interval(s)")
        for _, r in result.iterrows():
            log.info(f"    seizure: {r['start_time']:.2f}s - {r['stop_time']:.2f}s "
                     f"({r['stop_time'] - r['start_time']:.1f}s)")
        return result
 
    except Exception as e:
        log.warning(f"Failed to parse {csv_path}: {e}")
        return pd.DataFrame(columns=["start_time", "stop_time"])
 
 
