import os
from typing import Any, Dict, List
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

app = FastAPI()

# Enable CORS for Tableau Desktop and Cloud environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Root routes to serve index.html directly
@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse("static/index.html")

# 2. Static directory mount
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Tableau Predictor ML Backend"}

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    selected_record: Dict[str, Any]

def clean_key(name: str) -> str:
    cleaned = str(name)
    if "." in cleaned:
        cleaned = cleaned.split(".")[-1]
    cleaned = (
        cleaned.replace("[", "")
        .replace("]", "")
        .replace("sum:", "")
        .replace("avg:", "")
        .replace("attr:", "")
        .replace(":qk", "")
        .replace(":nk", "")
        .replace(":ok", "")
        .strip()
    )
    return cleaned

@app.post("/predict")
def run_prediction(payload: PredictRequest):
    records = payload.records
    selected = payload.selected_record

    if not records or len(records) < 5:
        return {"error": "Need at least 5 rows from Tableau worksheets to train the prediction model."}

    raw_df = pd.DataFrame(records).dropna(how="all", axis=1)

    # Clean all column names
    clean_cols = {c: clean_key(c) for c in raw_df.columns}
    df = raw_df.rename(columns=clean_cols)

    selected_clean = {clean_key(k): v for k, v in selected.items() if v is not None}

    # Case 1: Only 1 column passed by Tableau
    if len(df.columns) < 2:
        col_name = df.columns[0] if len(df.columns) > 0 else "Metric"
        val = selected_clean.get(col_name, "N/A")
        return {
            "target_name": col_name,
            "likelihood": 50,
            "predictors": [
                {
                    "impact": 0,
                    "label": "Place more fields (e.g., Region, Category, Sales, Discount) on the Marks Detail card"
                }
            ],
            "improvements": [
                {
                    "lift": "N/A",
                    "recommendation": "Add dimensions and measures to your sheet to enable multi-variable ML predictions."
                }
            ],
            "summary": (
                f"Currently evaluated against single feature '{col_name}'. "
                "Tableau extensions only receive fields present on the active sheet view. "
                "Drag additional fields to Detail/Tooltip to calculate driver importance."
            )
        }

    # Case 2: Multi-column predictive evaluation
    target_keywords = ["profit", "churn", "won", "converted", "status", "sales", "discount", "target"]
    target_col = None
    for kw in target_keywords:
        for c in df.columns:
            if kw in c.lower():
                target_col = c
                break
        if target_col:
            break

    if not target_col:
        target_col = df.columns[0]

    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col]).copy()

    # Formulate binary classification target
    if pd.api.types.is_numeric_dtype(y_raw) or pd.to_numeric(y_raw, errors="coerce").notna().all():
        y_num = pd.to_numeric(y_raw, errors="coerce").fillna(0)
        median_val = y_num.median()
        y = (y_num >= median_val).astype(int)
        target_label = f"High {target_col} (≥ {round(float(median_val), 1)})"
    else:
        top_val = y_raw.value_counts().index[0]
        y = (y_raw == top_val).astype(int)
        target_label = f"{target_col}: {top_val}"

    # Handle edge case where target is uniform
    if y.nunique() < 2:
        y.iloc[0] = 1 - y.iloc[0]

    # Preprocess features
    X_processed = pd.DataFrame(index=X_raw.index)
    encoders = {}
    num_cols = []
    cat_cols = []

    for col in X_raw.columns:
        numeric_series = pd.to_numeric(X_raw[col], errors="coerce")
        if numeric_series.notna().sum() >= 0.6 * len(X_raw):
            num_cols.append(col)
            med = numeric_series.median()
            X_processed[col] = numeric_series.fillna(0 if pd.isna(med) else med)
        else:
            cat_cols.append(col)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_processed[col] = enc.fit_transform(X_raw[col].fillna("Unknown").astype(str).values.reshape(-1, 1))
            encoders[col] = enc

    feature_names = X_processed.columns.tolist()

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42)
    model.fit(X_processed, y)

    # Transform selected record
    selected_row = pd.DataFrame(index=[0])
    for col in feature_names:
        val = selected_clean.get(col, None)
        if col in num_cols:
            val_num = pd.to_numeric(val, errors="coerce")
            selected_row[col] = 0 if pd.isna(val_num) else val_num
        else:
            str_val = "Unknown" if val is None else str(val)
            selected_row[col] = encoders[col].transform([[str_val]])[0][0]

    # Compute Likelihood %
    probs = model.predict_proba(selected_row)[0]
    likelihood = int(round(probs[1] * 100)) if len(probs) > 1 else int(round(probs[0] * 100))

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:4]
    top_predictors = []
    for idx in sorted_idx:
        fname = feature_names[idx]
        actual_val = selected_clean.get(fname, "N/A")
        pts = round(float(importances[idx]) * 50, 1)
        top_predictors.append({
            "impact": pts,
            "label": f"{fname} = {actual_val} (+{pts} pts)"
        })

    improvements = []
    if top_predictors:
        lead_feature = top_predictors[0]["label"].split(" = ")[0]
        improvements.append({
            "lift": f"+{round(top_predictors[0]['impact'] / 2, 1)}%",
            "recommendation": f"Calibrate {lead_feature} to increase probability of favorable outcome."
        })

    summary_text = (
        f"This selected profile demonstrates a {likelihood}% likelihood of achieving {target_label}. "
        f"Primary influential factor: {top_predictors[0]['label'] if top_predictors else 'baseline trend'}."
    )

    return {
        "target_name": target_label,
        "likelihood": likelihood,
        "predictors": top_predictors,
        "improvements": improvements,
        "summary": summary_text
    }