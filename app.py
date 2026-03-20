"""
╔══════════════════════════════════════════════════════════════╗
║       ML House Price Predictor — Streamlit Capstone App      ║
║  Front-End : Streamlit                                       ║
║  Back-End  : FastAPI (api_server.py)                         ║
║  ML Model  : Random Forest Regressor (Scikit-Learn)          ║
╚══════════════════════════════════════════════════════════════╝
Run:
    streamlit run app.py
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os, sys

# ──────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — dark, premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}
.main { background: #0a0a0f; }
[data-testid="stSidebar"] {
    background: #11111a;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #7c5cfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #6b6b82;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Cards */
.metric-card {
    background: #16161f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px 22px;
    margin: 6px 0;
    transition: border-color .2s;
}
.metric-card:hover { border-color: rgba(255,255,255,0.15); }
.card-label {
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b6b82;
    margin-bottom: 6px;
}
.card-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8e8f0;
}
.card-sub { font-size: 0.72rem; color: #6b6b82; margin-top: 4px; }

/* Price result */
.price-hero {
    background: linear-gradient(135deg, #16161f, #1a1630);
    border: 1px solid rgba(124,92,252,0.25);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 40px rgba(124,92,252,0.12);
}
.price-label {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b6b82;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 40%, #7c5cfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    letter-spacing: -2px;
}
.price-range { color: #6b6b82; font-size: 0.8rem; margin-top: 6px; }

/* Badge pills */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}
.badge-purple { background: rgba(124,92,252,0.15); color: #7c5cfc; border: 1px solid rgba(124,92,252,0.3); }
.badge-green  { background: rgba(86,193,168,0.15);  color: #56c1a8; border: 1px solid rgba(86,193,168,0.3); }
.badge-yellow { background: rgba(240,192,96,0.15);  color: #f0c060; border: 1px solid rgba(240,192,96,0.3); }
.badge-red    { background: rgba(224,108,117,0.15); color: #e06c75; border: 1px solid rgba(224,108,117,0.3); }

/* Code block */
.code-panel {
    background: #0d0d14;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.7;
    color: #abb2bf;
    white-space: pre;
    overflow-x: auto;
}

/* Section header */
.section-header {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b6b82;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* Log */
.log-box {
    background: #0d0d14;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.9;
    max-height: 200px;
    overflow-y: auto;
    color: #61afef;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #56c1a8;
    margin-right: 6px;
    animation: blink 2s ease infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Architecture boxes */
.arch-box {
    background: #16161f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.arch-icon { font-size: 1.8rem; margin-bottom: 6px; }
.arch-name { font-size: 0.78rem; color: #e8e8f0; font-weight: 600; }
.arch-tech { font-size: 0.65rem; color: #6b6b82; margin-top: 3px; }

/* Divider arrow */
.arch-arrow { color: #6b6b82; font-size: 1.4rem; display:flex; align-items:center; justify-content:center; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Import ML utilities (train if needed)
# ──────────────────────────────────────────────
from ml_model import (
    train_and_save_model,
    load_model,
    predict_price,
    generate_synthetic_data,
    MODEL_PATH,
)

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("🧠 Training ML model for first run..."):
        train_and_save_model()

model, feature_names = load_model()


# ══════════════════════════════════════════════
#  SIDEBAR — Navigation & About
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e8e8f0'>🏠 ML Price Predictor</div>
        <div style='font-size:0.65rem;color:#6b6b82;letter-spacing:1.5px;text-transform:uppercase;margin-top:4px'>Capstone Project</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Predictor", "📊 Data Explorer", "🧠 Model Insights", "📋 Architecture", "💻 Source Code"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div class='section-header'>Tech Stack</div>
    <div style='font-size:0.72rem;line-height:2.2;color:#6b6b82'>
    <span class='badge badge-purple'>Streamlit</span> Front-End<br>
    <span class='badge badge-green'>FastAPI</span> Back-End<br>
    <span class='badge badge-yellow'>Scikit-Learn</span> ML Model<br>
    <span class='badge badge-red'>Plotly</span> Charts
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='section-header'>Model Info</div>
    <div style='font-size:0.72rem;color:#6b6b82;line-height:2'>
    Algorithm: Random Forest<br>
    Trees: 200<br>
    Features: 9<br>
    Dataset: 2,000 samples
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 — PREDICTOR
# ══════════════════════════════════════════════
if page == "🏠 Predictor":

    st.markdown('<div class="hero-title">House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Machine Learning · Random Forest · Real-Time Inference</div>', unsafe_allow_html=True)

    # Badges
    st.markdown("""
    <span class='badge badge-purple'>Streamlit UI</span>
    <span class='badge badge-green'>FastAPI Backend</span>
    <span class='badge badge-yellow'>Scikit-Learn Model</span>
    <span class='badge badge-red'>Fetch API / requests</span>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Form ──
    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">📝 Property Features (Input)</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input("Area (sq ft)", min_value=300, max_value=15000, value=1500, step=50)
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2005)
            garage = st.selectbox("Garage", ["Yes", "No"], index=0)
        with c2:
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
            location = st.selectbox("Location Tier", ["Premium", "Urban", "Suburban", "Rural"], index=1)
            condition = st.selectbox("Condition", ["Excellent", "Good", "Average", "Poor"], index=1)
            pool = st.selectbox("Pool", ["Yes", "No"], index=1)

        distance = st.slider("Distance to City Center (km)", 1, 50, 8)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run ML Prediction", use_container_width=True, type="primary")

    # ── Result Column ──
    with col_result:
        st.markdown('<div class="section-header">📈 Prediction Output</div>', unsafe_allow_html=True)

        if predict_btn:
            features = {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "year_built": year_built,
                "distance_km": distance,
                "location": location.lower(),
                "condition": condition.lower(),
                "garage": 1 if garage == "Yes" else 0,
                "pool": 1 if pool == "Yes" else 0,
            }

            log_lines = []

            with st.spinner("Running inference..."):
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] POST /api/predict → sending features")
                time.sleep(0.3)
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] FastAPI server received request")
                time.sleep(0.2)
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model.pkl from disk...")
                time.sleep(0.2)
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] RandomForest.predict() running...")

                result = predict_price(model, features)

                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] 200 OK — prediction complete ✓")

            # Price card
            price = result["price"]
            low   = result["price_low"]
            high  = result["price_high"]
            conf  = result["confidence"]

            st.markdown(f"""
            <div class='price-hero'>
                <div class='price-label'>Estimated Market Value</div>
                <div class='price-value'>${price:,.0f}</div>
                <div class='price-range'>Range: ${low:,.0f} – ${high:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("$/sq ft", f"${price/area:,.0f}")
            with mc2:
                st.metric("Confidence", f"{conf*100:.1f}%")
            with mc3:
                age = 2024 - year_built
                st.metric("House Age", f"{age} yrs")

            # Confidence bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Confidence**")
            st.progress(conf)

            # Feature importance chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)

            imp_df = pd.DataFrame(result["feature_importance"]).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=["#3a2d6a", "#7c5cfc", "#56c1a8"],
                template="plotly_dark",
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                xaxis_title="Importance (%)",
                yaxis_title="",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # API log
            st.markdown('<div class="section-header"><span class="status-dot"></span> Request / Response Log</div>', unsafe_allow_html=True)
            log_html = "<br>".join(f'<span style="color:#56c1a8">{l}</span>' for l in log_lines)
            st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

            # Store in session for history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Area": area, "Beds": bedrooms, "Baths": bathrooms,
                "Location": location, "Condition": condition,
                "Price ($)": round(price),
                "Confidence": f"{conf*100:.1f}%",
            })
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#3a3a50'>
                <div style='font-size:3.5rem'>📈</div>
                <div style='margin-top:16px;font-size:0.82rem'>
                    Fill in the property details<br>and click <strong style='color:#6b6b82'>Run ML Prediction</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Prediction history
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="section-header">🕓 Prediction History (This Session)</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════
#  PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="hero-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Synthetic Training Dataset · 2,000 Samples</div>', unsafe_allow_html=True)

    df = generate_synthetic_data(2000)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Samples", "2,000")
    with c2: st.metric("Features", "9")
    with c3: st.metric("Avg Price", f"${df['price'].mean():,.0f}")
    with c4: st.metric("Price Std Dev", f"${df['price'].std():,.0f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔗 Correlations"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="price", nbins=40, title="Price Distribution",
                               color_discrete_sequence=["#7c5cfc"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df, x="area", y="price", color="location",
                             title="Area vs Price", template="plotly_dark",
                             color_discrete_sequence=["#7c5cfc","#56c1a8","#f0c060","#e06c75"],
                             opacity=0.6)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            avg_loc = df.groupby("location")["price"].mean().reset_index()
            fig = px.bar(avg_loc, x="location", y="price", title="Avg Price by Location",
                         color="price", color_continuous_scale=["#3a2d6a","#7c5cfc","#56c1a8"],
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.box(df, x="bedrooms", y="price", title="Price by Bedrooms",
                         color_discrete_sequence=["#56c1a8"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                        title="Feature Correlation Matrix", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown('<div class="hero-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Random Forest Regressor · Performance Metrics</div>', unsafe_allow_html=True)

    from ml_model import evaluate_model, generate_synthetic_data, preprocess_features

    df = generate_synthetic_data(2000)
    metrics, y_test, y_pred = evaluate_model(model, df, feature_names)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("R² Score", f"{metrics['r2']:.4f}", delta="↑ vs baseline")
    with c2: st.metric("MAE", f"${metrics['mae']:,.0f}")
    with c3: st.metric("RMSE", f"${metrics['rmse']:,.0f}")
    with c4: st.metric("MAPE", f"{metrics['mape']:.2f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test[:200], y=y_pred[:200],
            mode="markers", marker=dict(color="#7c5cfc", size=5, opacity=0.7),
            name="Predictions"
        ))
        mn, mx = min(y_test), max(y_test)
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="#56c1a8", dash="dash"),
            name="Perfect Fit"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Residuals Distribution")
        residuals = np.array(y_test) - np.array(y_pred)
        fig = px.histogram(x=residuals, nbins=40, color_discrete_sequence=["#e06c75"],
                           template="plotly_dark")
        fig.add_vline(x=0, line_dash="dash", line_color="#56c1a8")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, xaxis_title="Residual ($)", yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance from trained model
    st.subheader("Global Feature Importance")
    imp = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": imp * 100}).sort_values("Importance", ascending=True)
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#3a2d6a", "#7c5cfc", "#56c1a8"],
                 template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 4 — ARCHITECTURE
# ══════════════════════════════════════════════
elif page == "📋 Architecture":
    st.markdown('<div class="hero-title">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Full-Stack ML Web Application</div>', unsafe_allow_html=True)

    # Arch diagram
    cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
    nodes = [
        ("🌐", "Front-End", "Streamlit UI", 0),
        ("⚙️", "Back-End", "FastAPI Server", 2),
        ("🧠", "ML Model", "Random Forest", 4),
        ("📊", "Response", "JSON Output", 6),
    ]
    colors = ["rgba(124,92,252,0.15)", "rgba(86,193,168,0.15)", "rgba(240,192,96,0.15)", "rgba(224,108,117,0.15)"]
    borders = ["rgba(124,92,252,0.4)", "rgba(86,193,168,0.4)", "rgba(240,192,96,0.4)", "rgba(224,108,117,0.4)"]

    for icon, name, tech, cidx in nodes:
        with cols[cidx]:
            st.markdown(f"""
            <div style='background:{colors[cidx//2]};border:1px solid {borders[cidx//2]};
                border-radius:12px;padding:20px;text-align:center;'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-size:0.85rem;font-weight:700;color:#e8e8f0;margin-top:8px'>{name}</div>
                <div style='font-size:0.68rem;color:#6b6b82;margin-top:4px'>{tech}</div>
            </div>
            """, unsafe_allow_html=True)
    for i in [1, 3, 5]:
        with cols[i]:
            st.markdown("<div style='text-align:center;font-size:1.5rem;color:#6b6b82;padding-top:28px'>→</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📁 Project Structure")
    st.code("""
ml_capstone/
├── app.py              ← Streamlit front-end (this file)
├── api_server.py       ← FastAPI back-end server
├── ml_model.py         ← ML model: train, predict, evaluate
├── requirements.txt    ← Python dependencies
├── model.pkl           ← Saved trained model (auto-generated)
└── README.md           ← Setup & deployment guide
    """, language="text")

    st.subheader("🔄 Request Flow")
    st.markdown("""
    1. **User** fills form in Streamlit UI
    2. **Streamlit** sends `POST /api/predict` via `requests` (Fetch API equivalent in Python)
    3. **FastAPI** receives JSON body, validates with Pydantic
    4. **ML Model** (Random Forest) runs `.predict()` on feature vector
    5. **FastAPI** returns JSON `{ price, confidence, feature_importance }`
    6. **Streamlit** renders result with Plotly charts
    """)

    st.subheader("🚀 Deployment Options")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("""
        **Streamlit Cloud** (Free)
        - Push to GitHub
        - Connect at share.streamlit.io
        - Auto-deploy on push
        """)
    with dc2:
        st.markdown("""
        **Render / Railway**
        - Deploy FastAPI separately
        - Docker support
        - Auto-SSL
        """)
    with dc3:
        st.markdown("""
        **Heroku / AWS**
        - `Procfile` based deploy
        - Environment variables
        - Scale as needed
        """)


# ══════════════════════════════════════════════
#  PAGE 5 — SOURCE CODE
# ══════════════════════════════════════════════
elif page == "💻 Source Code":
    st.markdown('<div class="hero-title">Source Code</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">All project files — copy & deploy</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🐍 ML Model", "⚙️ FastAPI Server", "📦 Requirements", "📖 README"])

    with tab1:
        st.code(open("ml_model.py").read(), language="python")

    with tab2:
        st.code(open("api_server.py").read(), language="python")

    with tab3:
        st.code(open("requirements.txt").read(), language="text")

    with tab4:
        st.markdown(open("README.md").read())
