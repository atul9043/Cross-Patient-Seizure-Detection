import numpy as np
import pandas as pd
WINDOW_SEC   = 1 

def label_segments(segment_times: np.ndarray, annotations: pd.DataFrame,
                   sfreq: float, window_sec: float = WINDOW_SEC) -> np.ndarray:

    labels = np.zeros(len(segment_times), dtype=np.int32)
 
    if annotations.empty:
        return labels  # no seizures annotated → all non-seizure
 
    for i, t_start in enumerate(segment_times):
        t_end = t_start + window_sec
        for _, row in annotations.iterrows():
            if t_start < row["stop_time"] and t_end > row["start_time"]:
                labels[i] = 1
                break  # no need to check further intervals
 
    return labels