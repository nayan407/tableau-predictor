import os
from typing import Any, Dict, List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import shap

app = FastAPI()

# Enable CORS for Tableau environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static HTML frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    selected_record: Dict[str, Any]

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Tableau Predictor ML Backend"}

@app.post("/predict")
def run_prediction(payload: PredictRequest):
    records = payload.records
    selected = payload.selected_record

    if not records or len(records) < 5:
        return {"error": "Need at least 5 rows from Tableau worksheets to train the prediction model."}

    df = pd.DataFrame(records)

    # 1. Clean data: Drop all-null columns
    df = df.dropna(how="all", axis=1)
    if df.empty:
        return {"error": "No valid feature data extracted from dashboard."}

    # 2. Automatically select or designate the target column
    # Priority keywords for binary classification, otherwise pick first binary/numeric column
    target_col = None
    target_keywords = ["churn", "converted", "won", "deal_status", "status", "target", "outcome", "churned"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break

    if not target_col:
        # Fallback: look for a column with 2 unique non-null values
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                break

    if not target_col:
        # If no binary column found, pick the last numeric/categorical column as target
        target_col = df.columns[-1]

    # 3. Separate features and target
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col]).copy()

    # Drop high-cardinality ID columns from training
    drop_cols = [c for c in X_raw.columns if X_raw[c].nunique() > 0.9 * len(X_raw) and X_raw[c].dtype == object]
    X_raw = X_raw.drop(columns=drop_cols)

    # Encode target to binary (0/1)
    if y_raw.dtype == object or y_raw.nunique() > 2:
        # Binarize categorical or multi-class target
        top_val = y_raw.value_counts().index[0]
        y = (y_raw == top_val).astype(int)
        target_label = f"{target_col} ({top_val})"
    else:
        y = (y_raw > y_raw.median()).astype(int)
        target_label = f"High {target_col}"

    # Handle single-class edge case
    if y.nunique() < 2:
        y.iloc[0] = 1 - y.iloc[0]

    # 4. Preprocess features: Numeric vs Categorical
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    X_processed = pd.DataFrame(index=X_raw.index)

    # Fill numeric columns with median
    for col in num_cols:
        X_processed[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(X_raw[col].median() if not pd.isna(X_raw[col].median()) else 0)

    # Ordinal encode categorical columns
    cat_encoders = {}
    for col in cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        col_filled = X_raw[col].fillna("Unknown").astype(str).values.reshape(-1, 1)
        X_processed[col] = encoder.fit_transform(col_filled)
        cat_encoders[col] = encoder

    feature_names = X_processed.columns.tolist()

    if not feature_names:
        return {"error": "No viable feature columns available to train model."}

    # 5. Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    model.fit(X_processed, y)

    # 6. Transform the selected record
    selected_df = pd.DataFrame([selected])
    selected_processed = pd.DataFrame(index=[0])

    for col in feature_names:
        if col in num_cols:
            val = pd.to_numeric(selected_df[col].iloc[0] if col in selected_df else 0, errors="coerce")
            selected_processed[col] = 0 if pd.isna(val) else val
        elif col in cat_cols:
            val = str(selected_df[col].iloc[0]) if col in selected_df and not pd.isna(selected_df[col].iloc[0]) else "Unknown"
            encoder = cat_encoders[col]
            encoded_val = encoder.transform([[val]])[0][0]
            selected_processed[col] = encoded_val

    # 7. Generate Prediction Likelihood
    probs = model.predict_proba(selected_processed)[0]
    likelihood = int(round(probs[1] * 100)) if len(probs) > 1 else int(round(probs[0] * 100))

    # 8. Compute Feature Importance & SHAP Attributions
    top_predictors = []
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(selected_processed)

        # Handle SHAP output shapes for binary classification
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        impact_pairs = []
        for name, val in zip(feature_names, sv):
            impact_pairs.append((name, round(float(val) * 100, 1)))

        # Sort by absolute SHAP impact
        impact_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, impact in impact_pairs[:3]:
            actual_val = selected.get(name, "N/A")
            sign = "+" if impact >= 0 else "-"
            top_predictors.append({
                "impact": impact,
                "label": f"{name} is {actual_val} ({sign}{abs(impact)} pts)"
            })
    except Exception:
        # Fallback to model feature importances if SHAP calculation encounters singular matrices
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:3]
        for idx in sorted_idx:
            fname = feature_names[idx]
            imp = round(float(importances[idx]) * 50, 1)
            top_predictors.append({
                "impact": imp,
                "label": f"{fname} contributes {imp} pts"
            })

    # 9. Generate Actionable Improvements
    improvements = []
    for pred in top_predictors:
        if pred["impact"] < 0:
            improvements.append({
                "lift": f"+{abs(pred['impact'])}%",
                "recommendation": f"Adjust {pred['label'].split(' is ')[0]} to improve outcome probability"
            })

    if not improvements:
        improvements.append({
            "lift": "+5%",
            "recommendation": "Maintain optimal feature distribution across key dimensions"
        })

    # 10. Construct Business Summary
    status_sentiment = "favorable" if likelihood >= 50 else "at-risk"
    summary_text = (
        f"This selected profile shows a {likelihood}% likelihood for {target_label}. "
        f"The outcome is currently {status_sentiment}, driven primarily by "
        f"{top_predictors[0]['label'] if top_predictors else 'overall data trends'}."
    )

    return {
        "target_name": target_label,
        "likelihood": likelihood,
        "predictors": top_predictors,
        "improvements": improvements[:2],
        "summary": summary_text
    }