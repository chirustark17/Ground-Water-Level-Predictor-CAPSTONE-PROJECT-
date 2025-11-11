# dashboard_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------------------------
#  MODEL LOADING SECTION (Safe + Auto Path)
# -------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model/groundwater_rf_model.pkl")
features_path = os.path.join(base_dir, "../model/model_features.pkl")
data_path = os.path.join(base_dir, "../data/groundwater_master_data.csv")

# Check that model exists
if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error(
        f" Model or feature file not found!\n\n"
        f"Expected at:\n{model_path}\n\n"
        "Please run `main_model_pipeline.py` first (in `/src/`) to train and save the model."
    )
    st.stop()

# Load model and feature list
model = joblib.load(model_path)
features = joblib.load(features_path)
st.sidebar.success(" Model loaded successfully!")

# -------------------------------------------------
#  APP CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="Karnataka Groundwater Level Predictor", layout="wide")

st.title(" **Karnataka Groundwater Level Predictor**")
st.caption("Machine Learning-based Forecasting using Environmental and Demographic Data")

# -------------------------------------------------
#  SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header(" Input Parameters")

rain = st.sidebar.slider("Average Annual Rainfall (mm)", 500, 3000, 1200)
temp = st.sidebar.slider("Average Annual Temperature (°C)", 20.0, 35.0, 27.0)
pop_dist = st.sidebar.number_input("District Population", 100000, 5000000, 2000000)
pop_block = st.sidebar.number_input("Block Population", 10000, 1000000, 200000)
lat = st.sidebar.number_input("Latitude", 11.0, 18.0, 14.5)
lon = st.sidebar.number_input("Longitude", 74.0, 78.0, 76.0)
soil = st.sidebar.selectbox(
    "Predominant Soil Type",
    ["Red Sandy to Loamy Soil", "Deep Black Soil", "Lateritic Soil", "Mixed Soil"],
)

# -------------------------------------------------
#  FEATURE TRANSFORMATION (Updated to match model)
# -------------------------------------------------
row = {
    "LATITUDE": lat,
    "LONGITUDE": lon,
    "District Population": pop_dist / 5_000_000,
    "Block Population": pop_block / 1_000_000,
    "Annual_Avg_Temp": (temp - 20) / 15,
    "Rainfall(mm/Year)": rain / 2000,
}

# Add the engineered features used during model training
row["Rainfall_Temp_Interaction"] = row["Rainfall(mm/Year)"] * row["Annual_Avg_Temp"]
row["Pop_Ratio"] = row["Block Population"] / (row["District Population"] + 1e-6)

# One-hot encode soil types dynamically
for f in features:
    if "Predominant_Soil_Type" in f:
        row[f] = 1 if soil in f else 0

# Ensure every expected feature is present (fill missing with 0)
X_input = pd.DataFrame([row])
for col in features:
    if col not in X_input.columns:
        X_input[col] = 0

# Reorder columns to match training order
X_input = X_input[features]

# -------------------------------------------------
#  MODEL PREDICTION
# -------------------------------------------------
prediction = model.predict(X_input)[0]
st.metric("Predicted Groundwater Level (m bgl)", f"{prediction:.2f}")

# -------------------------------------------------
#  MODEL INSIGHTS
# -------------------------------------------------
st.subheader(" Model Insights")

importances = model.feature_importances_
sorted_idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color="teal")
ax.set_title("Feature Importance (Groundwater Predictor)", fontsize=12)
ax.set_xlabel("Relative Importance")
ax.grid(alpha=0.3)
st.pyplot(fig)

# -------------------------------------------------
#  OPTIONAL MAP VISUALIZATION
# -------------------------------------------------
if os.path.exists(data_path) and st.checkbox(" Show Groundwater Map (Demo)"):
    df = pd.read_csv(data_path)
    fig_map = px.scatter_mapbox(
        df,
        lat="LATITUDE",
        lon="LONGITUDE",
        color="WL(mbgl)",
        size="Rainfall(mm/Year)",
        hover_name="BLOCK",
        zoom=6,
        color_continuous_scale="RdYlBu",
        mapbox_style="carto-positron",
    )
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------
#  FOOTER
# -------------------------------------------------
st.divider()
st.write(
    "**Developed by:** Arjun Athrey (20221CSD0133), Chirag K S (20221CSD0134), Shreyas V (20221CSD0117)"
)
st.caption("Supervised by Dr. Sampath A K | Presidency University, Bengaluru | © 2025")
