import warnings
import logging
 
import numpy as np
from sklearn.pipeline import Pipeline
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 

def plot_feature_importance(model: Pipeline, feature_names: list[str],
                             top_n: int = 30) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available. Skipping plot.")
        return
 
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
 
    # Trim/pad feature_names to match importances length
    n = len(importances)
    names = feature_names[:n] if len(feature_names) >= n else feature_names + [f"feat_{i}" for i in range(n - len(feature_names))]
 
    # Sort and take top-N
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[sorted_idx]
    top_names = [names[i] for i in sorted_idx]
 
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), top_importances[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — Random Forest\nSeizure Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Feature importance saved to feature_importance.png")
 
 
def shap_analysis(model: Pipeline, X_test: np.ndarray,
                  feature_names: list[str], n_samples: int = 200) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("shap or matplotlib not installed. Run: pip install shap matplotlib")
        return
 
    rf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X_test[:n_samples])
 
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)
 
    # shap_values is a list [class0, class1] for binary classification
    sv_seizure = shap_values[1] if isinstance(shap_values, list) else shap_values
 
    n = sv_seizure.shape[1]
    names = feature_names[:n] if len(feature_names) >= n else feature_names + [f"feat_{i}" for i in range(n - len(feature_names))]
 
    print("\n  SHAP Summary (seizure class):")
    shap.summary_plot(sv_seizure, X_scaled, feature_names=names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    log.info("SHAP summary saved to shap_summary.png")
 
 