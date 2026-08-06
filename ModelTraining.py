import warnings
import logging
 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

 
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def split_by_patient(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     test_size: float = 0.2, random_state: int = 42) -> tuple:
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
 
    # ── Separate patients by seizure presence ─────────────────────────────────
    seizure_patients     = np.array([g for g in unique_groups
                                     if y[groups == g].sum() > 0])
    non_seizure_patients = np.array([g for g in unique_groups
                                     if y[groups == g].sum() == 0])
 
    if len(seizure_patients) == 0:
        raise RuntimeError(
            "No seizure segments found in the entire dataset. "
            "Check that load_annotations() is parsing your CSV correctly."
        )
    if len(seizure_patients) < 2:
        raise RuntimeError(
            f"Only {len(seizure_patients)} seizure patient found. "
            "Need at least 2 to guarantee seizures in both train and test. "
            "Load more data."
        )
 
    # ── Shuffle each group independently (same seed = reproducible) ───────────
    rng.shuffle(seizure_patients)
    rng.shuffle(non_seizure_patients)
 
    # ── Split seizure patients: guarantee at least 1 in test ──────────────────
    n_seiz_test  = max(1, int(round(len(seizure_patients) * test_size)))
    n_seiz_train = len(seizure_patients) - n_seiz_test
    # Edge case: rounding gave 0 train patients
    if n_seiz_train == 0:
        n_seiz_train, n_seiz_test = 1, len(seizure_patients) - 1
 
    train_seizure_pts = seizure_patients[:n_seiz_train]
    test_seizure_pts  = seizure_patients[n_seiz_train:]
 
    # ── Split non-seizure patients proportionally ─────────────────────────────
    n_non_test    = max(1, int(round(len(non_seizure_patients) * test_size)))
    test_non_pts  = non_seizure_patients[:n_non_test]
    train_non_pts = non_seizure_patients[n_non_test:]
 
    # ── Combine and build masks ───────────────────────────────────────────────
    train_patients = np.concatenate([train_seizure_pts, train_non_pts])
    test_patients  = np.concatenate([test_seizure_pts,  test_non_pts])
 
    train_mask = np.isin(groups, train_patients)
    test_mask  = np.isin(groups, test_patients)
 
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    groups_train, groups_test = groups[train_mask], groups[test_mask]

    # ── Logging ───────────────────────────────────────────────────────────────
    log.info(f"Seizure patients  → train {list(train_seizure_pts)} "
             f"| test {list(test_seizure_pts)}")
    log.info(f"Non-seizure patients → {len(train_non_pts)} train "
             f"| {len(test_non_pts)} test")
    log.info(f"Train: {int(y_train.sum())} seizures / {len(y_train)} segments "
             f"({100*y_train.mean():.2f}%)")
    log.info(f"Test : {int(y_test.sum())} seizures / {len(y_test)} segments "
             f"({100*y_test.mean():.2f}%)")
 
    assert y_train.sum() > 0, "BUG: no seizures in train after stratified split"
    assert y_test.sum()  > 0, "BUG: no seizures in test after stratified split"
 
    return X_train, X_test, y_train, y_test, groups_train, groups_test
 
 
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 200,
                        random_state: int = 42,
                        force_balanced_rf: bool | None = None) -> Pipeline:
    
    seizure_rate = float(np.mean(y_train == 1))
    log.info(f"  Training seizure rate: {100 * seizure_rate:.2f}%")
 
    # Determine which strategy to use
    if force_balanced_rf is None:
        use_balanced = seizure_rate < 0.10   # auto: use BRF when < 10% seizure
    else:
        use_balanced = force_balanced_rf
 
    shared_params = dict(
        n_estimators=n_estimators,
        max_depth=20,           # prevent overfitting on noisy features
        min_samples_leaf=5,     # smooths decision boundary, reduces FP from artifacts
        n_jobs=-1,
        random_state=random_state,
    )
 
    if use_balanced:
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
            clf = BalancedRandomForestClassifier(
                **shared_params,
                sampling_strategy="auto",   # undersample majority to match minority
                replacement=False,          # sample without replacement per tree
                oob_score=False,            # OOB requires bootstrap; incompatible with replacement=False
            )
            strategy_name = "BalancedRandomForestClassifier (imblearn)"
        except ImportError:
            log.warning(
                "imbalanced-learn not installed — falling back to "
                "RandomForestClassifier(class_weight='balanced').\n"
                "For better recall on heavily skewed data, install it with:\n"
                "    pip install imbalanced-learn"
            )
            clf = RandomForestClassifier(
                **shared_params,
                class_weight="balanced",
                oob_score=True,
            )
            strategy_name = "RandomForestClassifier (class_weight=balanced, fallback)"
    else:
        clf = RandomForestClassifier(
            **shared_params,
            class_weight="balanced",
            oob_score=True,
        )
        strategy_name = "RandomForestClassifier (class_weight=balanced)"
 
    log.info(f"Training: {strategy_name}")
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipeline.fit(X_train, y_train)
 
    clf_step = pipeline.named_steps["clf"]
    if hasattr(clf_step, "oob_score_"):
        log.info(f"OOB accuracy: {clf_step.oob_score_:.4f}  "
                 f"(note: OOB accuracy is misleading with imbalance — check Recall)")
    else:
        log.info("OOB scoring disabled (BalancedRF with replacement=False) — evaluate via test set metrics")
    return pipeline
 
 
def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              random_state: int = 42) -> Pipeline:
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", svm),
    ])
    log.info("Training SVM (this may take a while on large datasets)...")
    pipeline.fit(X_train, y_train)
    return pipeline
 
 
