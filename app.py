# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor

# ==================== CONFIG ====================
st.set_page_config(
    page_title="World Risk Index 2025",
    page_icon="Earth",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background: linear-gradient(to right, #1e3a8a, #1e40af);}
    .stButton>button {background-color: #f72585; color: white; font-weight: bold; border-radius: 10px;}
    h1, h2, h3 {color: #f72585 !important;}
    .css-1d391kg {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("World Risk Index 2025")
st.markdown("### Enter indicators → Get instant risk prediction")

# ==================== TRAIN MODEL ONCE (OFFLINE) ====================
@st.cache_resource
def get_model():
    X = np.array([
        [56.33,64.80,42.50,85.20,66.70], [56.04,53.20,30.10,82.50,47.00],
        [45.09,57.40,36.80,83.10,52.30], [36.40,63.80,43.20,86.40,61.80],
        [38.42,56.98,37.10,79.80,53.90], [27.52,69.64,48.90,88.70,71.30],
        [32.11,57.50,38.70,80.30,53.50], [42.39,42.27,23.50,65.10,38.20],
        [40.12,44.12,28.40,68.50,35.50], [28.91,58.16,39.80,81.20,53.50],
        [18.44,87.40,68.20,99.99,94.00], [38.77,41.20,22.10,69.80,31.70],
        [22.10,71.30,52.30,91.20,70.40], [25.97,59.40,41.10,82.60,54.50],
        [26.81,56.40,37.80,79.90,51.50]
    ])
    y = np.array([36.49,29.81,25.88,23.23,21.88,19.16,18.46,17.92,17.70,16.82,16.12,15.98,15.76,15.42,15.11])

    model = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=800, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=800, random_state=42))
    ])
    model.fit(X, y)
    return model

model = get_model()

# ==================== INPUTS ====================
st.sidebar.header("Risk Indicators (0–100)")

exposure = st.sidebar.slider("Exposure", 0.0, 100.0, 45.0, 0.1)
vulnerability = st.sidebar.slider("Vulnerability", 0.0, 100.0, 60.0, 0.1)
susceptibility = st.sidebar.slider("Susceptibility", 0.0, 100.0, 40.0, 0.1)
coping = st.sidebar.slider("Lack of Coping Capabilities", 0.0, 100.0, 80.0, 0.1)
adaptive = st.sidebar.slider("Lack of Adaptive Capacities", 0.0, 100.0, 60.0, 0.1)

# ==================== PREDICT ====================
if st.sidebar.button("Predict WRI Score"):
    features = np.array([[exposure, vulnerability, susceptibility, coping, adaptive]])
    score = model.predict(features)[0]

    st.success(f"### Predicted World Risk Index: **{score:.2f}**")

    if score >= 30:
        st.error("EXTREME RISK – Like Vanuatu")
    elif score >= 25:
        st.warning("VERY HIGH RISK – Like Philippines")
    elif score >= 20:
        st.info("HIGH RISK – Like Bangladesh")
    else:
        st.success("MODERATE OR LOW RISK")

    # Chart
    countries = ["Vanuatu", "Philippines", "Bangladesh", "Your Country"]
    scores = [36.49, 25.88, 19.16, score]
    fig, ax = plt.subplots()
    ax.bar(countries, scores, color=['#ef4444', '#f97316', '#facc15', '#8b5cf6'])
    ax.set_ylabel("WRI Score")
    ax.set_title("Your Country vs Top-Risk Nations")
    st.pyplot(fig)

st.caption("Model trained on official WorldRiskReport 2023 • 100% offline • Made with love")
