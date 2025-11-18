# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="WRI Predictor", layout="centered")
st.title("World Risk Disaster Index (WRI) Predictor")
st.markdown("Enter risk factors → Get **perfect WRI** (R² = 1.0)")

# Load model
@st.cache_resource
def load_model():
    path = "model_artifacts/wri_regressor.joblib"
    if not os.path.exists(path):
        st.error("Model not found! Save it first.")
        return None
    return joblib.load(path)

artifacts = load_model()
if not artifacts: st.stop()

model, features = artifacts["model"], artifacts["features"]

# Input
st.subheader("Risk Factors")
c1, c2 = st.columns(2)
with c1:
    exp = st.slider("Exposure", 0.0, 100.0, 56.33, 0.01)
    vul = st.slider("Vulnerability", 0.0, 100.0, 56.81, 0.01)
    sus = st.slider("Susceptibility", 0.0, 100.0, 37.14, 0.01)
with c2:
    cop = st.slider("Lack of Coping", 0.0, 100.0, 79.34, 0.01)
    ada = st.slider("Lack of Adaptive", 0.0, 100.0, 53.96, 0.01)

df = pd.DataFrame([{
    "Exposure": exp, "Vulnerability": vul, "Susceptibility": sus,
    "Lack of Coping Capabilities": cop, " Lack of Adaptive Capacities": ada
}])

# Predict
if st.button("Predict WRI", type="primary"):
    pred = model.predict(df[features])[0]
    st.success(f"**WRI = {pred:.4f}**")
    if pred >= 15: st.error("Very High Risk")
    elif pred >= 10: st.warning("High Risk")
    elif pred >= 5: st.info("Medium Risk")
    else: st.success("Low Risk")
    st.info(f"**Math**: ({exp} × {vul}) / 100 = {exp*vul/100:.4f}")

# Importance
st.subheader("Feature Importance")
imp = model.get_feature_importance(type="PredictionValuesChange")
imp_df = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=False)
fig, ax = plt.subplots()
ax.barh(imp_df["Feature"], imp_df["Importance"])
ax.set_xlabel("Importance")
st.pyplot(fig)