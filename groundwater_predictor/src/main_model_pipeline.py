# main_model_pipeline_v2.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------
# STEP 1: Load dataset safely
# --------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/groundwater_master_data.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f" Dataset not found at {data_path}")

print(f" Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# --------------------------------------------------------
# STEP 2: Basic cleaning and data type conversion
# --------------------------------------------------------
df.replace(["NA", "NaN", " ", ""], np.nan, inplace=True)
df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
df["Annual_Avg_Temp"] = pd.to_numeric(df["Annual_Avg_Temp"], errors="coerce")
df["Rainfall(mm/Year)"] = pd.to_numeric(df["Rainfall(mm/Year)"], errors="coerce")
df["District Population"] = pd.to_numeric(df["District Population"], errors="coerce")
df["Block Population"] = pd.to_numeric(df["Block Population"], errors="coerce")
df["WL(mbgl)"] = pd.to_numeric(df["WL(mbgl)"], errors="coerce")

# Drop rows with missing target
df.dropna(subset=["WL(mbgl)"], inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# --------------------------------------------------------
# STEP 3: Feature engineering
# --------------------------------------------------------
df["Rainfall_Temp_Interaction"] = df["Rainfall(mm/Year)"] * df["Annual_Avg_Temp"]
df["Pop_Ratio"] = df["Block Population"] / (df["District Population"] + 1)

# Normalize numerical features roughly
scaler = MinMaxScaler()
df[["Rainfall(mm/Year)", "Annual_Avg_Temp", "District Population",
    "Block Population", "Rainfall_Temp_Interaction", "Pop_Ratio"]] = scaler.fit_transform(
    df[["Rainfall(mm/Year)", "Annual_Avg_Temp", "District Population",
        "Block Population", "Rainfall_Temp_Interaction", "Pop_Ratio"]]
)

# One-hot encode soil types
if "Predominant_Soil_Type" in df.columns:
    df = pd.get_dummies(df, columns=["Predominant_Soil_Type"])

# --------------------------------------------------------
# STEP 4: Feature selection
# --------------------------------------------------------
X = df.drop(columns=["WL(mbgl)", "STATE_UT", "DISTRICT", "BLOCK", "VILLAGE_NA", "Date"], errors="ignore")
y = df["WL(mbgl)"]

print(f" Data ready for training. Samples: {X.shape[0]}, Features: {X.shape[1]}")

# --------------------------------------------------------
# STEP 5: Split data
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------
# STEP 6: Train model with GridSearchCV for tuning
# --------------------------------------------------------
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 12, 14],
    "min_samples_split": [2, 4, 6]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)

print(" Running grid search for best parameters...")
grid.fit(X_train, y_train)

model = grid.best_estimator_
print(f" Best parameters found: {grid.best_params_}")

# --------------------------------------------------------
# STEP 7: Evaluate model
# --------------------------------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f" Model Performance:\nRMSE = {rmse:.3f}\nR² = {r2:.3f}")

# --------------------------------------------------------
# STEP 8: Save model and features
# --------------------------------------------------------
model_dir = os.path.join(base_dir, "../model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "groundwater_rf_model.pkl")
features_path = os.path.join(model_dir, "model_features.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), features_path)

print(f" Model saved at: {model_path}")
print(f" Feature list saved at: {features_path}")

print("\n Training complete. You can now run Streamlit using:")
print("   cd ../app")
print("   streamlit run dashboard_app.py")
