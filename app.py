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

# 1. Serve index.html directly when requested via / or /index.html
@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse("static/index.html")

# 2. Mount static folder for any additional assets (css, js, images)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Tableau Predictor ML Backend"}

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    selected_record: Dict[str, Any]

# (Keep the rest of your run_prediction function exactly as it is)