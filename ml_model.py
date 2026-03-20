"""
ml_model.py — Machine Learning Model
Handles: data generation, training, saving, loading, prediction, evaluation.
Algorithm: Random Forest Regressor (Scikit-Learn)
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ──────────────────────────────────────────────
# 1. Synthetic Dataset Generator
# ──────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic housing data for training."""
    rng = np.random.default_rng(random_state)

    locations    = rng.choice(["premium", "urban", "suburban", "rural"], n_samples,
                               p=[0.15, 0.35, 0.35, 0.15])
    conditions   = rng.choice(["excellent", "good", "average", "poor"], n_samples,
                               p=[0.15, 0.45, 0.30, 0.10])
    bedrooms     = rng.integers(1, 7, n_samples)
    bathrooms    = np.clip(rng.integers(1, 5, n_samples), 1, bedrooms)
    area         = rng.integers(400, 6000, n_samples)
    year_built   = rng.integers(1970, 2023, n_samples)
    distance_km  = rng.integers(1, 50, n_samples)
    garage       = rng.integers(0, 2, n_samples)
    pool         = rng.integers(0, 2, n_samples)

    # Price model with realistic multipliers
    loc_mul  = {"premium": 1.65, "urban": 1.20, "suburban": 1.00, "rural": 0.72}
    cond_mul = {"excellent": 1.18, "good": 1.00, "average": 0.86, "poor": 0.70}

    base = area * 175.0
    prices = np.array([
        base[i]
        * loc_mul[locations[i]]
        * cond_mul[conditions[i]]
        * max(0.65, 1 - (2024 - year_built[i]) * 0.005)   # age factor
        * max(0.70, 1 - distance_km[i] * 0.009)            # distance factor
        * (1 + bedrooms[i] * 0.04)
        * (1 + bathrooms[i] * 0.03)
        * (1.05 if garage[i] else 1.0)
        * (1.08 if pool[i] else 1.0)
        for i in range(n_samples)
    ])

    # Add realistic noise
    noise = rng.normal(0, 0.06, n_samples)
    prices = prices * (1 + noise)
    prices = np.clip(prices, 60_000, 3_500_000).astype(int)

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "distance_km": distance_km,
        "location": locations,
        "condition": conditions,
        "garage": garage,
        "pool": pool,
        "price": prices,
    })
    return df


# ──────────────────────────────────────────────
# 2. Feature Preprocessing
# ──────────────────────────────────────────────
LOCATION_MAP  = {"premium": 3, "urban": 2, "suburban": 1, "rural": 0}
CONDITION_MAP = {"excellent": 3, "good": 2, "average": 1, "poor": 0}

FEATURE_NAMES = [
    "area", "bedrooms", "bathrooms", "house_age",
    "distance_km", "location_enc", "condition_enc",
    "garage", "pool"
]

def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """Convert raw dataframe into model-ready feature matrix."""
    X = df.copy()
    X["house_age"]     = 2024 - X["year_built"]
    X["location_enc"]  = X["location"].map(LOCATION_MAP).fillna(1)
    X["condition_enc"] = X["condition"].map(CONDITION_MAP).fillna(2)
    return X[FEATURE_NAMES].values


# ──────────────────────────────────────────────
# 3. Train & Save
# ──────────────────────────────────────────────
def train_and_save_model(n_samples: int = 2000):
    """Train the Random Forest model and save to disk."""
    print("📊 Generating training data...")
    df = generate_synthetic_data(n_samples)

    X = preprocess_features(df)
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🌲 Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"✅ Training complete!")
    print(f"   MAE  : ${mae:,.0f}")
    print(f"   RMSE : ${rmse:,.0f}")
    print(f"   R²   : {r2:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((rf, FEATURE_NAMES), f)
    print(f"💾 Model saved → {MODEL_PATH}")
    return rf, FEATURE_NAMES


# ──────────────────────────────────────────────
# 4. Load Model
# ──────────────────────────────────────────────
def load_model():
    """Load trained model from disk."""
    with open(MODEL_PATH, "rb") as f:
        model, feature_names = pickle.load(f)
    return model, feature_names


# ──────────────────────────────────────────────
# 5. Predict
# ──────────────────────────────────────────────
def predict_price(model, features: dict) -> dict:
    """
    Run prediction given a feature dict.
    Returns price, confidence, range, feature importance.
    """
    # Build feature vector
    house_age    = 2024 - features.get("year_built", 2005)
    location_enc = LOCATION_MAP.get(features.get("location", "urban"), 2)
    cond_enc     = CONDITION_MAP.get(features.get("condition", "good"), 2)

    x = np.array([[
        features.get("area", 1500),
        features.get("bedrooms", 3),
        features.get("bathrooms", 2),
        house_age,
        features.get("distance_km", 10),
        location_enc,
        cond_enc,
        features.get("garage", 0),
        features.get("pool", 0),
    ]])

    # Predict with all trees to get distribution
    tree_preds = np.array([tree.predict(x)[0] for tree in model.estimators_])
    price      = float(np.mean(tree_preds))
    price_std  = float(np.std(tree_preds))

    # Confidence: inverse of coefficient of variation (capped)
    cv = price_std / price if price > 0 else 0.1
    confidence = float(np.clip(1 - cv * 3, 0.70, 0.96))

    # Feature importance from model
    feat_imp = [
        {"feature": name.replace("_enc","").replace("_"," ").title(),
         "importance": round(imp * 100, 1)}
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    ]

    return {
        "price":             round(price),
        "price_low":         round(price - 1.5 * price_std),
        "price_high":        round(price + 1.5 * price_std),
        "confidence":        confidence,
        "feature_importance": sorted(feat_imp, key=lambda d: d["importance"], reverse=True),
    }


# ──────────────────────────────────────────────
# 6. Evaluate
# ──────────────────────────────────────────────
def evaluate_model(model, df: pd.DataFrame, feature_names):
    """Run test-set evaluation and return metrics."""
    X = preprocess_features(df)
    y = df["price"].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return (
        {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape},
        y_test.tolist(),
        y_pred.tolist(),
    )


# ──────────────────────────────────────────────
# CLI: python ml_model.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save_model()
