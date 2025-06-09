import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load data ===
def load_subjects(subject_ids, data_dir="../data_100Hz"):
    dfs = []
    for sid in subject_ids:
        path = os.path.join(data_dir, f"{sid}_PSG_df_updated.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.dropna()
        df = df[df["Sleep_Stage"].isin(["W", "N1", "N2", "N3", "R"])]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# === Step 2: Preprocess ===
def preprocess(df):
    X = df.drop(columns=["TIMESTAMP", "Sleep_Stage"])
    y = df["Sleep_Stage"].map({
        "W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4
    })
    return X, y

# === Step 3: Train & Evaluate ===
def train_rf(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = []
    all_true = []
    for train_idx, test_idx in skf.split(X, y):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = rf.predict(X.iloc[test_idx])
        all_preds.extend(preds)
        all_true.extend(y.iloc[test_idx])

    print(classification_report(all_true, all_preds, target_names=["Wake", "N1", "N2", "N3", "REM"]))

    return rf, np.array(all_true), np.array(all_preds)

# === Step 4: Visuals ===
def plot_results(y_true, y_pred, rf, feature_names):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Wake", "N1", "N2", "N3", "REM"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Feature importance
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-15:]  # top 15
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

# === Run all ===
if __name__ == "__main__":
    subjects = ["S005", "S007", "S010"]  # Replace with downloaded ones
    df = load_subjects(subjects)
    X, y = preprocess(df)
    rf, y_true, y_pred = train_rf(X, y)
    plot_results(y_true, y_pred, rf, X.columns)
