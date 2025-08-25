import os
import json
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt

DATA_PATHS = {
    "parkinsons": r"C:\Users\Shruthilaya\GUVI\data\parkinsons - parkinsons.csv",       
    "kidney": r"C:\Users\Shruthilaya\GUVI\data\kidney_disease - kidney_disease.csv",
    "liver": r"C:\Users\Shruthilaya\GUVI\data\indian_liver_patient - indian_liver_patient.csv",
}

TARGETS = {
    "parkinsons": {"target": "status", "drop": ["name"], "positive_label": None},  
    "kidney": {"target": "classification", "drop": ["id"], "positive_label": "ckd"},
    "liver": {"target": "Dataset", "drop": [], "positive_label": 1},  # 1=disease, 2=no disease
}

MODEL_KIND = {
    "parkinsons": "logistic",
    "kidney": "rf",
    "liver": "gbdt",
}

OUTPUT_DIR = "./output"
SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5

def ensure_dirs():
    for sub in ["models", "reports", "plots", "cv"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def build_estimator(kind: str):
    kind = (kind or "logistic").lower()
    if kind == "logistic":
        return LogisticRegression(max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, random_state=SEED)
    if kind == "gbdt":
        return GradientBoostingClassifier(random_state=SEED)
    return LogisticRegression(max_iter=1000)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)])

def binarize_y(y: pd.Series, positive_label):
    if positive_label is None:
        return y.astype(int)
    return (y == positive_label).astype(int)

def evaluate_classification(y_true, y_prob, y_pred) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    return {"accuracy": acc, "precision": float(pr), "recall": float(rc), "f1": float(f1), "roc_auc": auc}

def plot_confusion(cm: np.ndarray, labels=("No", "Yes"), title="Confusion Matrix"):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig

def run_one(disease: str):
    cfg = TARGETS[disease]
    path = DATA_PATHS[disease]
    df = pd.read_csv(path)

    # dataset-specific cleaning
    if disease == "liver":
        # Map 1->1 (disease), 2->0 (no disease)
        df[cfg["target"]] = df[cfg["target"]].map({1: 1, 2: 0}).astype(int)
        # optional renames (if column names vary)
        df = df.rename(columns={
            "Albumin_and_Globulin_Ratio": "A/G_Ratio",
            "Total_Protiens": "Total_Proteins",
            "Alkaline_Phosphotase": "Alkaline_Phosphatase",
        })
    if disease == "kidney":
        # standardize some categorical text fields
        cat_like = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "classification"]
        for c in cat_like:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        # clean common trailing tabs or whitespace in labels
        if cfg["target"] in df.columns:
            df[cfg["target"]] = df[cfg["target"]].replace({"ckd\t": "ckd"})

    # drop id-like columns if any
    if cfg["drop"]:
        df = df.drop(columns=cfg["drop"], errors="ignore")

    target = cfg["target"]
    X = df.drop(columns=[target])
    y = binarize_y(df[target], cfg["positive_label"])

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    # pipeline
    pre = build_preprocessor(X_train)
    clf = build_estimator(MODEL_KIND[disease])
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # cross-validate on train
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False)
    cv_summary = {k.replace("test_",""): float(np.nanmean(v)) for k,v in cvres.items() if k.startswith("test_")}

    # fit final on train
    pipe.fit(X_train, y_train)

    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        scores = pipe.decision_function(X_test).reshape(-1, 1)
        smin, smax = scores.min(), scores.max()
        y_prob = ((scores - smin) / (smax - smin + 1e-9)).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_classification(y_test, y_prob, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # save data
    ensure_dirs()
    dump(pipe, os.path.join(OUTPUT_DIR, "models", f"{disease}.joblib"))
    with open(os.path.join(OUTPUT_DIR, "reports", f"{disease}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "cv", f"{disease}_cv_metrics.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    fig = plot_confusion(cm, labels=("No", "Yes"), title=f"{disease.capitalize()} – Confusion Matrix")
    fig.savefig(os.path.join(OUTPUT_DIR, "plots", f"{disease}_confusion_matrix.png"))
    plt.close(fig)

    return {
        "disease": disease,
        "test_metrics": metrics,
        "cv_mean_metrics": cv_summary,
        "test_size": int(y_test.shape[0]),
        "train_size": int(y_train.shape[0]),
        "n_features": X.shape[1],
    }

def main():
    results = {}
    for disease in ["parkinsons", "kidney", "liver"]:
        res = run_one(disease)
        results[disease] = res
        print(f"\n=== {disease.upper()} ===")
        print(json.dumps(res, indent=2))
    with open(os.path.join(OUTPUT_DIR, "reports", "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
