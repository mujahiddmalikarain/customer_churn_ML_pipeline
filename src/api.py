from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# -----------------------------
# Paths (robust)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src" / "models" / "churn_model.joblib"

THRESHOLD = 0.30  # from your tuning (best F1)

# If your saved model requires this function during unpickling
def to_str_array(x):
    return x.astype(str)


# -----------------------------
# App + Model load
# -----------------------------
app = FastAPI(title="Churn Prediction API", version="1.0.0")

model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    # Send ANY set of features as key/value.
    # We'll align them to the model's expected columns.
    features: Dict[str, Any] = Field(..., description="Customer features as JSON key/value pairs")


class PredictResponse(BaseModel):
    churn_probability: float
    threshold: float
    churn_prediction: int


# -----------------------------
# Helpers
# -----------------------------
def get_expected_feature_names(pipeline) -> list:
    """
    Try to fetch training-time feature names from the ColumnTransformer.
    Works when pipeline step name is 'preprocess' like in your training.
    """
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None:
        raise ValueError("Pipeline missing 'preprocess' step. Please confirm training pipeline step names.")

    # ColumnTransformer stores original column names in transformers_
    # transformers_: list of tuples (name, transformer, columns)
    cols = []
    for _, _, c in preprocess.transformers_:
        if isinstance(c, list):
            cols.extend(c)
        else:
            # could be array-like or slice; handle minimal
            try:
                cols.extend(list(c))
            except Exception:
                pass
    return cols


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise RuntimeError("Model not loaded")

    # Make a 1-row dataframe from incoming JSON
    incoming = req.features
    X = pd.DataFrame([incoming])

    # Align columns to expected features (missing -> NaN, extra ignored)
    expected_cols = get_expected_feature_names(model)
    X = X.reindex(columns=expected_cols)

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    return PredictResponse(
        churn_probability=proba,
        threshold=THRESHOLD,
        churn_prediction=pred,
    )
