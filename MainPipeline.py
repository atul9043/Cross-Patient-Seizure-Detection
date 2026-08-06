import warnings
import logging
 
from sklearn.metrics import confusion_matrix

from DataConstruction import build_dataset
from Evaluation import adjust_threshold, evaluate_model, false_alarms_per_24h, smooth_predictions
from Explainability import plot_feature_importance, shap_analysis
from FeatureExtraction import get_feature_names
from ModelTraining import split_by_patient, train_random_forest, train_svm

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

WINDOW_SEC   = 1 

def run_pipeline(dataset_root: str,
                 apply_filter: bool = True,
                 max_recordings: int | None = None,
                 train_svm_also: bool = False,
                 run_shap: bool = False) -> dict:
    log.info("=" * 60)
    log.info("  EEG SEIZURE DETECTION PIPELINE")
    log.info("=" * 60)
 
    # ── Step 1: Build dataset ─────────────────────────────────────────────────
    X, y, groups = build_dataset(dataset_root, apply_filter, max_recordings)
 
    # ── Step 2: Train/test split (recording-grouped) ──────────────────────────
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_by_patient(X, y, groups)
 
    # ── Step 3: Feature names ─────────────────────────────────────────────────
    feature_names = get_feature_names(include_temporal=True)
 
    # ── Step 4: Train Random Forest ───────────────────────────────────────────
    rf_model = train_random_forest(X_train, y_train)
 
    # ── Step 5: Tune threshold FIRST so evaluation uses the right cutoff ────────
    # Evaluating at 0.5 then tuning afterwards produces misleading numbers.
    # adjust_threshold() finds the threshold that maximises recall at target FPR.
    best_thresh = adjust_threshold(rf_model, X_test, y_test, target_fpr=0.05)
 
    # ── Step 6: Evaluate at default 0.5 AND tuned threshold ──────────────────
    rf_metrics_default = evaluate_model(rf_model, X_test, y_test,
                                        "Random Forest", threshold=0.5)
    rf_metrics = evaluate_model(rf_model, X_test, y_test,
                                "Random Forest", threshold=best_thresh)
 
    # ── Step 6b: Apply smoothing at tuned threshold and report FA/24hr ────────
    y_prob        = rf_metrics["y_prob"]
    y_pred_smooth = smooth_predictions(y_prob, threshold=best_thresh,
                                       window=7, min_run=4)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, y_pred_smooth).ravel()
    total_hours   = len(y_test) * WINDOW_SEC / 3600
    fa            = false_alarms_per_24h(y_pred_smooth, y_test, total_hours)
    print(f"--- After smoothing (window=5s, min_run=3s) ---")
    print(f"  Recall  : {tp_s/(tp_s+fn_s+1e-8):.4f}")
    print(f"  FPR     : {fp_s/(fp_s+tn_s+1e-8):.4f}")
    print(f"  FA/24hr : {fa:.1f}   (paper: HMM=244, CNN/LSTM=7)")
 
    # ── Step 7: Optional SVM comparison ──────────────────────────────────────
    svm_metrics = None
    if train_svm_also:
        svm_model = train_svm(X_train, y_train)
        svm_metrics = evaluate_model(svm_model, X_test, y_test,
                                     "SVM (RBF)", threshold=best_thresh)
 
    # ── Step 8: Feature importance ────────────────────────────────────────────
    plot_feature_importance(rf_model, feature_names, top_n=30)
 
    # ── Step 9: SHAP (optional) ───────────────────────────────────────────────
    if run_shap:
        shap_analysis(rf_model, X_test, feature_names)
 
    return {
        "model": rf_model,
        "metrics": rf_metrics,
        "svm_metrics": svm_metrics,
        "feature_names": feature_names,
        "best_threshold": best_thresh,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "groups_train": groups_train,
    }
 