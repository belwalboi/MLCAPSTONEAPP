"""
api_server.py — FastAPI Back-End Server
Exposes REST endpoints consumed by Streamlit via requests (Fetch API equivalent).

Run:
    uvicorn api_server:app --reload --port 8000

Endpoints:
    POST /api/predict     → Run ML prediction
    GET  /api/health      → Server health check
    GET  /api/model-info  → Model metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import os
import numpy as np
from datetime import datetime

from ml_model import load_model, predict_price, train_and_save_model, MODEL_PATH

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="ML House Price Predictor API",
    description="FastAPI back-end serving a trained Random Forest Regressor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (or any browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
if not os.path.exists(MODEL_PATH):
    print("⚙️  Model not found — training now...")
    train_and_save_model()

model, feature_names = load_model()
print(f"✅ Model loaded: {MODEL_PATH}")


# ──────────────────────────────────────────────
# Pydantic Schemas (Request / Response)
# ──────────────────────────────────────────────
class HouseFeatures(BaseModel):
    area:        int   = Field(..., ge=100, le=20000,  description="Area in sq ft")
    bedrooms:    int   = Field(..., ge=1,   le=10,     description="Number of bedrooms")
    bathrooms:   int   = Field(..., ge=1,   le=10,     description="Number of bathrooms")
    year_built:  int   = Field(..., ge=1900, le=2024,  description="Year the house was built")
    distance_km: int   = Field(..., ge=1,   le=100,    description="Distance to city center in km")
    location:    str   = Field(..., description="premium | urban | suburban | rural")
    condition:   str   = Field(..., description="excellent | good | average | poor")
    garage:      int   = Field(0,   ge=0,   le=1,      description="1 if has garage")
    pool:        int   = Field(0,   ge=0,   le=1,      description="1 if has pool")

    @validator("location")
    def validate_location(cls, v):
        valid = ["premium", "urban", "suburban", "rural"]
        if v.lower() not in valid:
            raise ValueError(f"location must be one of {valid}")
        return v.lower()

    @validator("condition")
    def validate_condition(cls, v):
        valid = ["excellent", "good", "average", "poor"]
        if v.lower() not in valid:
            raise ValueError(f"condition must be one of {valid}")
        return v.lower()


class PredictionResponse(BaseModel):
    price:              int
    price_low:          int
    price_high:         int
    confidence:         float
    price_per_sqft:     float
    feature_importance: list
    model_version:      str
    timestamp:          str


class HealthResponse(BaseModel):
    status:     str
    model:      str
    uptime:     str
    timestamp:  str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
_start_time = datetime.utcnow()


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    uptime = datetime.utcnow() - _start_time
    return {
        "status": "healthy",
        "model": "RandomForestRegressor (n_estimators=200)",
        "uptime": str(uptime).split(".")[0],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/model-info", tags=["System"])
async def model_info():
    """Return model metadata."""
    return {
        "algorithm": "RandomForestRegressor",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": feature_names,
        "n_features": len(feature_names),
        "model_file": MODEL_PATH,
        "framework": "scikit-learn",
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(house: HouseFeatures):
    """
    Run house price prediction using the trained Random Forest model.

    This is the core ML inference endpoint:
    1. Validate input with Pydantic
    2. Preprocess features
    3. Run RandomForest.predict()
    4. Return prediction with confidence interval
    """
    try:
        result = predict_price(model, house.dict())
        return {
            **result,
            "price_per_sqft": round(result["price"] / house.area, 2),
            "model_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/batch-predict", tags=["Prediction"])
async def batch_predict(houses: list[HouseFeatures]):
    """Predict prices for multiple houses at once (batch inference)."""
    if len(houses) > 50:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 50")
    results = []
    for house in houses:
        result = predict_price(model, house.dict())
        results.append({
            "input": house.dict(),
            "prediction": result,
        })
    return {"batch_size": len(houses), "results": results}


# ──────────────────────────────────────────────
# Run with: uvicorn api_server:app --reload
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
