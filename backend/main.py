import io
import math
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


MODEL_PATH = "models/best_bearing_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_columns.joblib"
FRONTEND_DIR = "frontend"
FRONTEND_PATH = os.path.join(FRONTEND_DIR, "index.html")

app = FastAPI(title="Bearing Intelligence API", version="2.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# These mounts are required because the HTML document is served separately from
# its CSS and JavaScript files. Without them, /frontend/styles.css and
# /frontend/app.js return 404 even though the files exist in the repository.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


def load_artifacts():
    paths = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {missing}")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), list(joblib.load(FEATURES_PATH))


try:
    MODEL, SCALER, FEATURE_COLUMNS = load_artifacts()
except Exception as error:
    MODEL = SCALER = None
    FEATURE_COLUMNS = []
    ARTIFACT_ERROR = str(error)
else:
    ARTIFACT_ERROR = None


def json_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def safe_records(frame: pd.DataFrame):
    return [
        {str(key): json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def numeric_summary(frame: pd.DataFrame):
    numeric = frame.select_dtypes(include=np.number)
    rows = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce")
        rows.append({
            "feature": str(column),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "mean": json_value(series.mean()),
            "std": json_value(series.std()),
            "min": json_value(series.min()),
            "max": json_value(series.max()),
            "median": json_value(series.median()),
        })
    return rows


def find_label_column(frame: pd.DataFrame):
    preferred = ["label", "target", "class", "fault", "failure", "bearing_state", "status"]
    lowered = {str(column).lower(): column for column in frame.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    return None


def evaluate_labels(actual, predicted, label_column):
    actual = pd.to_numeric(actual, errors="coerce")
    mask = actual.notna()
    if not mask.any():
        return None
    actual = actual[mask].astype(int)
    predicted = pd.Series(predicted, index=actual.index).astype(int)
    labels = sorted(set(actual.tolist()) | set(predicted.tolist()))
    matrix = confusion_matrix(actual, predicted, labels=labels).tolist()
    average = "binary" if set(labels).issubset({0, 1}) else "weighted"
    return {
        "available": True,
        "label_column": str(label_column),
        "samples": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, average=average, zero_division=0)),
        "recall": float(recall_score(actual, predicted, average=average, zero_division=0)),
        "f1": float(f1_score(actual, predicted, average=average, zero_division=0)),
        "labels": labels,
        "confusion_matrix": matrix,
    }


def analyze_frame(frame: pd.DataFrame):
    numeric = frame.select_dtypes(include=np.number)
    correlation_frame = numeric.corr().round(5) if not numeric.empty else pd.DataFrame()
    correlation = {
        str(column): {str(key): json_value(value) for key, value in values.items()}
        for column, values in correlation_frame.to_dict().items()
    }
    result = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(column) for column in frame.columns],
        "numeric_columns": [str(column) for column in numeric.columns],
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "summary": numeric_summary(frame),
        "correlation": correlation,
        "preview": safe_records(frame.head(12)),
        "model": {
            "ready": MODEL is not None,
            "name": type(MODEL).__name__ if MODEL else None,
            "required_features": FEATURE_COLUMNS,
            "error": ARTIFACT_ERROR,
        },
        "evaluation": {
            "available": False,
            "reason": "Upload a labelled dataset to calculate dataset-specific metrics.",
        },
        "predictions": None,
    }
    if MODEL is None:
        return result

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    result["model"]["missing_features"] = missing
    if missing:
        result["evaluation"]["reason"] = f"Cannot score this dataset. Missing model features: {', '.join(missing)}"
        return result

    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = ~features.isna().any(axis=1)
    if not valid.any():
        result["evaluation"]["reason"] = "Model features contain no complete numeric rows."
        return result

    scaled = SCALER.transform(features.loc[valid])
    predictions = MODEL.predict(scaled)
    state_series = pd.Series(predictions, index=frame.index[valid])
    states = state_series.map({0: "Healthy", 1: "Faulty / Degraded"}).fillna(state_series.astype(str))
    prediction_frame = pd.DataFrame({
        "row": frame.index[valid].tolist(),
        "prediction": predictions.tolist(),
        "state": states.tolist(),
    })
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(scaled)
        classes = list(getattr(MODEL, "classes_", [0, 1]))
        if 1 in classes:
            prediction_frame["fault_probability"] = probabilities[:, classes.index(1)]
        if 0 in classes:
            prediction_frame["healthy_probability"] = probabilities[:, classes.index(0)]
    result["predictions"] = safe_records(prediction_frame)
    result["scored_rows"] = int(valid.sum())

    label_column = find_label_column(frame)
    if label_column:
        evaluation = evaluate_labels(frame.loc[valid, label_column], predictions, label_column)
        if evaluation:
            result["evaluation"] = evaluation
    else:
        result["evaluation"]["reason"] = "No label column detected. Showing live model risk and prediction distribution; accuracy metrics require verified labels."
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None, "model": type(MODEL).__name__ if MODEL else None}


@app.get("/")
def index():
    return FileResponse(FRONTEND_PATH)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        raw = await file.read()
        frame = pd.read_csv(io.BytesIO(raw))
        if frame.empty:
            raise ValueError("The uploaded CSV is empty.")
        result = analyze_frame(frame)
        result["filename"] = file.filename
        return result
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Unable to analyze CSV: {error}") from error
