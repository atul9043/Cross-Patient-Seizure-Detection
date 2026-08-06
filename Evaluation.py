import warnings
import logging
 
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 

def smooth_predictions(y_prob: np.ndarray,
                       threshold: float = 0.4,
                       window: int = 5,
                       min_run: int = 3) -> np.ndarray:

    from scipy.ndimage import uniform_filter1d
    from itertools import groupby
 
    # Stage 1: smooth probabilities
    smoothed = uniform_filter1d(y_prob.astype(np.float64), size=window)
    y_pred = (smoothed >= threshold).astype(np.int32)
 
    # Stage 2: remove runs shorter than min_run
    result = y_pred.copy()
    i = 0
    for val, group in groupby(y_pred):
        run_len = len(list(group))
        if val == 1 and run_len < min_run:
            result[i:i + run_len] = 0
        i += run_len
 
    n_removed = int(y_pred.sum()) - int(result.sum())
    log.info(f"Smoothing: window={window}s, min_run={min_run}s → "
             f"removed {n_removed} isolated positive segments")
    return result
 
 
def false_alarms_per_24h(y_pred: np.ndarray, y_test: np.ndarray,
                          total_hours: float) -> float:
    from itertools import groupby
 
    fa_count = 0
    i = 0
    for val, group in groupby(y_pred):
        run_len = len(list(group))
        if val == 1:
            # This is a positive prediction run — is it a false alarm?
            true_in_run = y_test[i:i + run_len].sum()
            if true_in_run == 0:
                fa_count += 1   # entire run contains no real seizure → FP event
        i += run_len
 
    fa_per_24h = fa_count * (24.0 / max(total_hours, 1e-6))
    return fa_per_24h
 
 
def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Random Forest",
                   threshold: float | None = None) -> dict:
    # Always use predict_proba so threshold is explicit and visible
    y_prob = model.predict_proba(X_test)[:, 1]
    effective_threshold = threshold if threshold is not None else 0.5
    y_pred = (y_prob >= effective_threshold).astype(int)
 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
 
    metrics = {
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred, zero_division=0),
        "recall":      recall_score(y_test, y_pred, zero_division=0),
        "f1":          f1_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "fpr":         fp / (fp + tn + 1e-8),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold":   effective_threshold,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
    }
 
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {model_name}  [threshold={effective_threshold:.3f}]")
    print(f"{'='*60}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall (Sens): {metrics['recall']:.4f}  <- Must be HIGH")
    print(f"  F1-score    : {metrics['f1']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  FPR         : {metrics['fpr']:.4f}  <- Must be LOW")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}  FN={fn}")
    print(f"    FP={fp}  TN={tn}")
    print(f"\n  Full Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Seizure", "Seizure"]))
    print(f"{'='*60}\n")
 
    return metrics
 
 
def adjust_threshold(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
                     target_fpr: float = 0.05) -> float:

    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.95, 200)
 
    best_thresh = 0.5
    best_recall = 0.0
 
    print(f"\n{'='*50}")
    print(f"  Threshold Tuning (target FPR ≤ {target_fpr:.2f})")
    print(f"{'='*50}")
 
    for thresh in thresholds:
        y_pred_t = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        fpr = fp / (fp + tn + 1e-8)
        recall = tp / (tp + fn + 1e-8)
 
        if fpr <= target_fpr and recall > best_recall:
            best_recall = recall
            best_thresh = thresh
 
    print(f"  Best threshold : {best_thresh:.3f}")
    print(f"  At this threshold, Recall = {best_recall:.4f}, FPR ≤ {target_fpr:.2f}")
    return best_thresh
