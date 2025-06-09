import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = "/Users/veeralpatel/ECE284FinalProject/data"
stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
dreamt_files = glob.glob(os.path.join(DATA_DIR, "S*_PSG_df_updated.csv"))
records = []

for file in dreamt_files:
    df = pd.read_csv(file, usecols=['HR', 'Sleep_Stage'])
    df = df.dropna(subset=['HR', 'Sleep_Stage'])

    df['Sleep_Stage'] = df['Sleep_Stage'].map(stage_map)
    df = df.dropna(subset=['Sleep_Stage'])
    df['Sleep_Stage'] = df['Sleep_Stage'].astype(int)

    if df.empty or len(df) < 20:
        print(f"Skipping {file} due to insufficient or invalid data.")
        continue

    label_val = df['Sleep_Stage'].mode()[0]

    record = {
        'mean_hr': df['HR'].mean(),
        'std_hr': df['HR'].std(),
        'min_hr': df['HR'].min(),
        'max_hr': df['HR'].max(),
        'delta_hr': df['HR'].iloc[-1] - df['HR'].iloc[0],
        'morning_slope': df['HR'].diff().rolling(5).mean().iloc[-10:].mean(),
        'label': label_val
    }
    records.append(record)

features_df = pd.DataFrame(records)

if 'label' in features_df.columns and not features_df.empty:
    X = features_df.drop(columns=['label'])
    y = features_df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = "/Users/veeralpatel/ECE284FinalProject/Application/rf_sleep_stage_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
else:
    print("No valid records found. Check input files and stage labels.")
