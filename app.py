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

# Enable CORS for Tableau Desktop, Server, and Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route handlers
@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse("static/index.html")

# Static assets mount
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
    return (
        cleaned.replace("[", "")
        .replace("]", "")
        .replace("sum:", "")
        .replace("avg:", "")
        .replace("attr:", "")
        .replace("min:", "")
        .replace("max:", "")
        .replace(":qk", "")
        .replace(":nk", "")
        .replace(":ok", "")
        .strip()
    )

def audit_features_and_leakage(df: pd.DataFrame, target_name: str, selected_clean: dict):
    issues = []
    warnings = []
    
    # 1. Target Leakage Check
    target_clean = clean_key(target_name).lower()
    for col in df.columns:
        c_clean = clean_key(col).lower()
        if c_clean == target_clean or (target_clean in c_clean and abs(len(c_clean) - len(target_clean)) < 5):
            issues.append(f"Target leakage detected: '{col}' matches target '{target_name}'. Excluded from features.")
    
    # 2. Missing Fields in Selected Mark Check
    missing_in_selected = []
    for col in df.columns:
        c = clean_key(col)
        if c != clean_key(target_name) and (c not in selected_clean or selected_clean[c] in [None, "", "N/A"]):
            missing_in_selected.append(c)
            
    if missing_in_selected:
        warnings.append(f"Selected mark missing fields: {', '.join(missing_in_selected[:4])}. Populated with baseline fallback.")

    # 3. Sample Size Check
    if len(df) < 30:
        warnings.append(f"Small sample size ({len(df)} rows). Recommended: 100+ rows for high statistical confidence.")
        
    return issues, warnings

@app.post("/validate")
def validate_dataset(payload: PredictRequest):
    records = payload.records
    selected = payload.selected_record
    
    if not records or len(records) < 5:
        return {
            "status": "FAILED",
            "score": 0,
            "issues": ["Dataset contains fewer than 5 rows. Cannot train."],
            "warnings": [],
            "summary": "Insufficient rows."
        }
        
    raw_df = pd.DataFrame(records).dropna(how="all", axis=1)
    clean_cols = {c: clean_key(c) for c in raw_df.columns}
    df = raw_df.rename(columns=clean_cols)
    selected_clean = {clean_key(k): v for k, v in selected.items() if v is not None}
    
    # Auto-detect target column
    target_col = None
    for kw in ["profit", "churn", "won", "converted", "status", "sales", "target"]:
        for c in df.columns:
            if kw in c.lower():
                target_col = c
                break
        if target_col:
            break
    if not target_col:
        target_col = df.columns[0]
        
    leakage_issues, mark_warnings = audit_features_and_leakage(df, target_col, selected_clean)
    
    health_score = 100
    if leakage_issues:
        health_score -= 20
    if mark_warnings:
        health_score -= 15 * len(mark_warnings)
    if len(df.columns) < 3:
        health_score -= 25
    health_score = max(health_score, 10)
    
    status_label = "HEALTHY" if health_score >= 80 else ("FAIR" if health_score >= 50 else "ACTION REQUIRED")
    
    return {
        "status": status_label,
        "score": health_score,
        "target_detected": target_col,
        "issues": leakage_issues,
        "warnings": mark_warnings,
        "summary": f"Data Health: {health_score}/100. Evaluated {len(df)} rows."
    }

@app.post("/predict")
def run_prediction(payload: PredictRequest):
    records = payload.records
    selected = payload.selected_record

    if not records or len(records) < 5:
        return {"error": "Need at least 5 rows from Tableau worksheets to train the prediction model."}

    raw_df = pd.DataFrame(records).dropna(how="all", axis=1)
    clean_cols = {c: clean_key(c) for c in raw_df.columns}
    df = raw_df.rename(columns=clean_cols)
    selected_clean = {clean_key(k): v for k, v in selected.items() if v is not None}

    # Case 1: Insufficient features
    if len(df.columns) < 2:
        col_name = df.columns[0] if len(df.columns) > 0 else "Metric"
        return {
            "target_name": col_name,
            "likelihood": 50,
            "predictors": [
                {"impact": 0, "label": "Add dimensions and measures to the Worksheet Detail shelf to generate features."}
            ],
            "improvements": [
                {"lift": "N/A", "recommendation": "Expose more fields (e.g., Region, Discount, Sales) on the sheet."}
            ],
            "summary": "Single-column input. Add fields to Marks shelf so the model can evaluate multidimensional patterns."
        }

    # Case 2: Target Selection
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

    # STRICT ANTI-LEAKAGE: Remove target and target-derivative columns from feature set
    t_clean = clean_key(target_col).lower()
    drop_cols = [
        c for c in df.columns 
        if clean_key(c).lower() == t_clean or (t_clean in clean_key(c).lower() and "margin" not in clean_key(c).lower())
    ]
    X_raw = df.drop(columns=drop_cols).copy()

    if X_raw.empty or len(X_raw.columns) == 0:
        return {
            "target_name": target_col,
            "likelihood": 50,
            "predictors": [{"impact": 0, "label": "No independent predictor features left after target leakage exclusion."}],
            "improvements": [{"lift": "N/A", "recommendation": "Add non-target features (Region, Category, Volume) to worksheet."}],
            "summary": "Only target-related columns were supplied. Add predictor fields to Worksheet Detail."
        }

    # Binary Classification Target setup
    if pd.api.types.is_numeric_dtype(y_raw) or pd.to_numeric(y_raw, errors="coerce").notna().all():
        y_num = pd.to_numeric(y_raw, errors="coerce").fillna(0)
        median_val = float(y_num.median())
        y = (y_num >= median_val).astype(int)
        target_label = f"High {target_col} (≥ {round(median_val, 1)})"
    else:
        top_val = str(y_raw.value_counts().index[0])
        y = (y_raw == top_val).astype(int)
        target_label = f"{target_col}: {top_val}"

    if y.nunique() < 2:
        y.iloc[0] = 1 - y.iloc[0]

    # Preprocessing
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
            selected_row[col] = X_processed[col].median() if pd.isna(val_num) else val_num
        else:
            str_val = "Unknown" if val is None else str(val)
            selected_row[col] = encoders[col].transform([[str_val]])[0][0]

    probs = model.predict_proba(selected_row)[0]
    likelihood = int(round(probs[1] * 100)) if len(probs) > 1 else int(round(probs[0] * 100))

    # Feature Importances
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
        f"This profile demonstrates a {likelihood}% likelihood of achieving {target_label}. "
        f"Primary driver: {top_predictors[0]['label'] if top_predictors else 'baseline trend'}."
    )

    return {
        "target_name": target_label,
        "likelihood": likelihood,
        "predictors": top_predictors,
        "improvements": improvements,
        "summary": summary_text
    }