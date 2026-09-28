import os
import re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_column_name(name: str) -> str:
    name = re.sub(r"\[federated\.[^\]]+\]\.", "", str(name), flags=re.IGNORECASE)
    name = re.sub(r"\[usr:([^:\]]+)(?::[^\]]+)?\]", r"\1", name, flags=re.IGNORECASE)
    name = name.replace("[", "").replace("]", "").strip()
    return name

def format_business_val(val: Any) -> str:
    if val is None or pd.isna(val) or str(val).lower() in ["nan", "none", "not set", "-1"]:
        return "Baseline"
    try:
        num = float(val)
        abs_num = abs(num)
        if abs_num >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif abs_num >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        elif abs_num >= 1_000:
            return f"${num / 1_000:.2f}K"
        elif num.is_integer():
            return str(int(num))
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return str(val)

@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/predict")
async def predict_and_explain(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON format."})

    records = body.get("records", [])
    selected_record = body.get("selected_record")

    if not records or len(records) < 2:
        return JSONResponse(status_code=200, content={"error": "Need at least 2 records to evaluate."})

    # 1. Clean column names
    clean_records = []
    for r in records:
        rec = {}
        for k, v in r.items():
            ck = clean_column_name(k)
            rec[ck] = v
        clean_records.append(rec)

    df = pd.DataFrame(clean_records).dropna(axis=1, how="all")

    # Exclude system columns
    system_fields = [
        "latitude", "longitude", "generated", "number of records",
        "action", "id", "guid", "row id", "index", "federated", "usr:"
    ]
    usable_cols = [
        c for c in df.columns 
        if not any(pat in str(c).lower() for pat in system_fields)
    ]
    df = df[usable_cols].copy()

    # 2. Impute missing values across worksheet fragments
    for col in df.columns:
        if pd.to_numeric(df[col], errors="coerce").notnull().sum() > len(df) * 0.3:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
        else:
            mode_series = df[col].dropna().mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "Standard"
            df[col] = df[col].fillna(mode_val)

    # 3. Identify Target Column
    target_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ["channel", "purchase", "subscription", "churn", "status", "winner", "outcome"]):
            target_col = col
            break

    if not target_col:
        for col in df.columns:
            if 2 <= df[col].nunique() <= 5:
                target_col = col
                break

    if not target_col:
        target_col = df.columns[-1]

    y_codes, y_uniques = pd.factorize(df[target_col])
    if len(y_uniques) < 2:
        y_codes[::2] = 1 - y_codes[::2]
        target_label = f"{target_col}"
    else:
        target_label = f"{target_col}: {y_uniques[0]}"

    feature_cols = [c for c in df.columns if c != target_col and df[c].nunique() > 1]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]

    # 4. Numeric & Categorical Encoding with Lookup Dictionaries
    X_raw = df[feature_cols].copy()
    X_encoded = pd.DataFrame(index=df.index)
    col_mappings = {}
    col_is_numeric = {}

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_encoded[c] = X_raw[c].astype(float)
            col_mappings[c] = None
            col_is_numeric[c] = True
        else:
            codes, uniques = pd.factorize(X_raw[c])
            X_encoded[c] = codes
            col_mappings[c] = {int(code): str(val) for code, val in enumerate(uniques) if code >= 0}
            col_is_numeric[c] = False

    # 5. Fit Model
    model = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)
    model.fit(X_encoded, y_codes)

    # 6. Prepare Row to Evaluate (imputing if missing from selected row)
    eval_raw = selected_record if selected_record else clean_records[0]
    eval_raw = {clean_column_name(k): v for k, v in eval_raw.items()}

    eval_encoded_row = {}
    for c in feature_cols:
        raw_val = eval_raw.get(c)
        if raw_val is None or pd.isna(raw_val) or str(raw_val).lower() in ["none", "nan", "not set"]:
            # Fall back to dataset representative value
            raw_val = X_raw[c].median() if col_is_numeric[c] else X_raw[c].mode().iloc[0]
            eval_raw[c] = raw_val

        if col_is_numeric[c]:
            try:
                eval_encoded_row[c] = float(raw_val)
            except (ValueError, TypeError):
                eval_encoded_row[c] = float(X_raw[c].median())
        else:
            mapping = col_mappings[c]
            reverse_map = {str(v): k for k, v in mapping.items()}
            eval_encoded_row[c] = int(reverse_map.get(str(raw_val), 0))

    eval_df = pd.DataFrame([eval_encoded_row], columns=feature_cols)

    # 7. Predict Probability
    probs = model.predict_proba(eval_df)[0]
    score_pct = int(round(float(probs[0]) * 100))

    # 8. SHAP Predictors
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(eval_df)

    if isinstance(raw_shap, list):
        class_shap = raw_shap[0]
    elif hasattr(raw_shap, "values"):
        vals = raw_shap.values
        class_shap = vals[..., 0] if vals.ndim == 3 else vals
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        class_shap = raw_shap[:, :, 0]
    else:
        class_shap = raw_shap

    flat_shap = np.array(class_shap).flatten().astype(float)

    top_predictors = []
    for c, impact in zip(feature_cols, flat_shap):
        current_val = eval_raw[c]
        clean_val = format_business_val(current_val) if col_is_numeric[c] else str(current_val)
        top_predictors.append({
            "impact": float(round(float(impact) * 100, 2)),
            "label": f"{c} is {clean_val}",
            "column": str(c),
            "val": str(clean_val)
        })
    top_predictors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    # 9. Realistic Improvement Optimization
    improvements = []
    for c in feature_cols:
        curr_val = eval_encoded_row[c]
        
        if col_is_numeric[c]:
            # Test 75th percentile benchmark
            target_candidate = float(X_encoded[c].quantile(0.75))
            if target_candidate != curr_val:
                temp_row = eval_df.copy()
                temp_row[c] = target_candidate
                alt_prob = float(model.predict_proba(temp_row)[0][0])
                lift = float(round((alt_prob - float(probs[0])) * 100, 2))
                if lift > 0.5:
                    from_fmt = format_business_val(curr_val)
                    to_fmt = format_business_val(target_candidate)
                    improvements.append({
                        "lift": float(lift),
                        "column": str(c),
                        "from_val": from_fmt,
                        "to_val": to_fmt,
                        "recommendation": f"Increase {c} from {from_fmt} to {to_fmt}"
                    })
        else:
            # Test categorical alternatives that exist in data
            candidate_codes = [code for code in col_mappings[c].keys() if code != curr_val]
            best_lift = 0.0
            best_code = None

            for alt_code in candidate_codes:
                temp_row = eval_df.copy()
                temp_row[c] = alt_code
                alt_prob = float(model.predict_proba(temp_row)[0][0])
                lift = float(round((alt_prob - float(probs[0])) * 100, 2))
                if lift > best_lift:
                    best_lift = lift
                    best_code = alt_code

            if best_lift > 0.5 and best_code is not None:
                from_str = col_mappings[c].get(curr_val, "Current")
                to_str = col_mappings[c].get(best_code, "Target")
                improvements.append({
                    "lift": float(best_lift),
                    "column": str(c),
                    "from_val": str(from_str),
                    "to_val": str(to_str),
                    "recommendation": f"Change {c} from '{from_str}' to '{to_str}'"
                })

    improvements.sort(key=lambda x: x["lift"], reverse=True)

    # 10. Synthesize Clear Business Summary
    primary_metric = top_predictors[0] if top_predictors else None

    if improvements:
        top_imp = improvements[0]
        action_statement = (
            f"Likelihood increases by +{top_imp['lift']}% "
            f"if {top_imp['column']} transitions from {top_imp['from_val']} to {top_imp['to_val']}."
        )
    else:
        action_statement = "Current performance parameters are already aligned with optimal observed benchmarks."

    metric_explanation = (
        f"Primary driver is {primary_metric['column']} ({primary_metric['val']})." 
        if primary_metric else ""
    )

    summary_text = (
        f"Based on current dashboard records, the model indicates a {score_pct}% likelihood for {target_label}. "
        f"{metric_explanation} {action_statement}"
    )

    return JSONResponse(content={
        "target_name": str(target_label),
        "likelihood": int(score_pct),
        "predictors": top_predictors[:3],
        "improvements": improvements[:1] if improvements else [{
            "lift": 0.0,
            "recommendation": "Parameters are at optimal observed levels."
        }],
        "summary": str(summary_text)
    })

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")