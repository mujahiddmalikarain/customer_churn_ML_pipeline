import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.xlsx"
MODEL_PATH = ROOT / "src" / "models" / "churn_model.joblib"
OUT_PATH = ROOT / "reports" / "metrics_threshold_030.json"

TARGET_COL = "Churn Label"
LEAKAGE_COLS = ["Churn Reason", "Churn Score", "Churn Value"]

THRESHOLD = 0.30  # ✅ chosen threshold

# Keep for unpickling if needed
def to_str_array(x):
    return x.astype(str)

def main():
    df = pd.read_excel(DATA_PATH)
    for c in LEAKAGE_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    y = df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COL])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = joblib.load(MODEL_PATH)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "threshold": THRESHOLD,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

    print("\n=== Metrics @ threshold =", THRESHOLD, "===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred, digits=4))

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved ->", OUT_PATH)

if __name__ == "__main__":
    main()
