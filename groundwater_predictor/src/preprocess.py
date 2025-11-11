# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path: str):
    """
    Loads groundwater dataset, cleans, encodes, and scales it.
    """
    df = pd.read_csv(file_path)
    print(f" Loaded dataset with {len(df)} records and {df.shape[1]} columns.")

    # Drop duplicates, fill missing values
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    # Encode categorical features
    if 'soil_type' in df.columns:
        df = pd.get_dummies(df, columns=['soil_type'])

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(" Preprocessing complete. Data ready for modeling.")
    return df, scaler
