# 🏠 ML House Price Predictor — Capstone Project

A **full-stack machine learning web application** built with:
- **Streamlit** — Front-End UI
- **FastAPI** — Back-End REST API
- **Scikit-Learn** — Random Forest ML Model
- **Plotly** — Interactive Charts
- **Fetch API / requests** — Front-End ↔ Back-End communication

---

## 📁 Project Structure

```
ml_capstone/
├── app.py              ← Streamlit front-end (main UI)
├── api_server.py       ← FastAPI back-end server
├── ml_model.py         ← ML: data, training, prediction, evaluation
├── requirements.txt    ← All Python dependencies
├── model.pkl           ← Auto-generated on first run
└── README.md           ← This file
```

---

## 🚀 Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — auto-trains on first run)
```bash
python ml_model.py
```

### 3a. Run Streamlit only (standalone mode)
```bash
streamlit run app.py
```
Open: http://localhost:8501

### 3b. Run with FastAPI backend (full-stack mode)
```bash
# Terminal 1 — Start FastAPI
uvicorn api_server:app --reload --port 8000

# Terminal 2 — Start Streamlit
streamlit run app.py
```
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

---

## 🧠 Machine Learning Model

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| n_estimators | 200 |
| max_depth | 15 |
| Training samples | 2,000 |
| Features | 9 |

**Features used:**
1. Area (sq ft)
2. Bedrooms
3. Bathrooms
4. House age (derived from year built)
5. Distance to city center (km)
6. Location tier (premium/urban/suburban/rural)
7. Condition (excellent/good/average/poor)
8. Garage (yes/no)
9. Pool (yes/no)

**Typical Performance:**
- R² Score: ~0.94
- MAE: ~$18,000
- RMSE: ~$26,000

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health check |
| GET | `/api/model-info` | Model metadata |
| POST | `/api/predict` | Single prediction |
| POST | `/api/batch-predict` | Batch predictions (max 50) |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2005,
    "distance_km": 8,
    "location": "urban",
    "condition": "good",
    "garage": 1,
    "pool": 0
  }'
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click **Deploy** — live in ~2 minutes!

---

## 🌐 Deploy FastAPI to Render (Free)

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: ml-price-api
    env: python
    buildCommand: pip install -r requirements.txt && python ml_model.py
    startCommand: uvicorn api_server:app --host 0.0.0.0 --port $PORT
```
2. Push to GitHub, connect to [render.com](https://render.com)
3. Update Streamlit's API URL to your Render URL

---

## 📚 Concepts Demonstrated

- ✅ **Front-End**: Streamlit UI with custom CSS, forms, charts
- ✅ **Back-End**: FastAPI with Pydantic validation, CORS, REST endpoints
- ✅ **ML Model**: Scikit-Learn Random Forest, feature engineering, evaluation
- ✅ **API Communication**: `requests` library (Python Fetch API equivalent)
- ✅ **Data Visualization**: Plotly charts (distributions, scatter, heatmap)
- ✅ **Model Serialization**: `pickle` for saving/loading the trained model
- ✅ **Deployment-ready**: Streamlit Cloud + Render instructions included
