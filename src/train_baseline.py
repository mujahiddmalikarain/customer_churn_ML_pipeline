import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.common import to_str_array  # ✅ stable import (fixes uvicorn pickle issue)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.xlsx"
MODEL_PATH = ROOT / "src" / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

TARGET_COL = "Churn Label"
LEAKAGE_COLS = ["Churn Reason", "Churn Score", "Churn Value"]


def ensure_dirs():
    (ROOT / "src" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(to_str_array, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])


def main():
    ensure_dirs()

    print("Training baseline...")
    print("DATA_PATH:", DATA_PATH)
    print("MODEL_PATH:", MODEL_PATH)

    df = pd.read_excel(DATA_PATH)

    # Drop leakage columns
    for c in LEAKAGE_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Target
    y = df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), TARGET_COL].unique()
        raise ValueError(f"Unexpected values in '{TARGET_COL}': {bad}")

    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline(X)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "model": "LogisticRegression",
        "threshold_default": 0.5,
    }

    print("\n=== Baseline Metrics (threshold=0.5) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred, digits=4))

    joblib.dump(clf, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Model saved ->", MODEL_PATH)
    print("✅ Metrics saved ->", METRICS_PATH)


if __name__ == "__main__":
    main()
