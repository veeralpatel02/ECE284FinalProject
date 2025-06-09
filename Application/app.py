import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA



st.set_page_config(page_title="SmartWake", layout="wide")
st.title("ML-Based Smart Alarm")
st.markdown("Predict sleep stage and find optimal wake-up time using **heart rate only** from your Apple Watch.")
MODEL_PATH = "/Users/veeralpatel/ECE284FinalProject/Application/rf_sleep_stage_model.pkl"
model = joblib.load(MODEL_PATH)
stage_lookup = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

uploaded_file = st.file_uploader("Upload your Apple Watch CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'Heart Rate'])
    df['Date'] = df['Timestamp'].dt.date

    # Select a night
    available_dates = df['Date'].unique()
    selected_date = st.selectbox("Select a Night", available_dates[::-1])
    night_df = df[df['Date'] == selected_date].copy()
    night_df.sort_values('Timestamp', inplace=True)

    st.subheader(f"HR Trend for {selected_date}")
    st.line_chart(night_df.set_index('Timestamp')['Heart Rate'])
    mean_hr = night_df['Heart Rate'].mean()
    std_hr = night_df['Heart Rate'].std()
    min_hr = night_df['Heart Rate'].min()
    max_hr = night_df['Heart Rate'].max()
    delta_hr = max_hr - min_hr
    hr_slope = night_df['Heart Rate'].diff().rolling(5).mean().iloc[-10:].mean()

    st.markdown(f"""
    ** HR Summary:**
    - Mean HR: {mean_hr:.1f}
    - Std Dev: {std_hr:.1f}
    - Min: {min_hr:.1f}, Max: {max_hr:.1f}
    - Delta HR: {delta_hr:.1f}
    - Morning HR Slope: {hr_slope:.4f}
    """)

    feature_vector = np.array([[mean_hr, std_hr, min_hr, max_hr, delta_hr, hr_slope]])
    predicted_stage = model.predict(feature_vector)[0]
    stage_name = stage_lookup.get(predicted_stage, "Unknown")

    st.success(f"**Predicted Dominant Sleep Stage:** {stage_name}")

    st.subheader(" Wake-Up Window Estimation")
    wake_start = st.time_input("Preferred Wake Window Start", value=datetime.strptime("06:30", "%H:%M").time())
    wake_end = st.time_input("Preferred Wake Window End", value=datetime.strptime("07:30", "%H:%M").time())

    window_df = night_df[(night_df['Timestamp'].dt.time >= wake_start) &
                         (night_df['Timestamp'].dt.time <= wake_end)]

    if not window_df.empty:
        hr_diff = window_df['Heart Rate'].diff().fillna(0)
        smoothed = hr_diff.rolling(3).mean()

        if smoothed.dropna().empty:
            st.warning("ERROR FIX THIS.")
        else:
            smooth_index = smoothed.idxmax()
            if smooth_index in window_df.index:
                recommended_time = window_df.loc[smooth_index, 'Timestamp']
                st.success(f"Recommended Wake-Up Time: **{recommended_time.strftime('%H:%M:%S')}**")
                st.line_chart(window_df.set_index('Timestamp')['Heart Rate'])
            else:
                st.warning("⚠️ No peak slope found in wake window.")
    else:
        st.warning("No HR data in selected wake-up window.")
    
    # ===== Visualizer Section =====
    st.subheader("Model Visualizer")

    # 1. Feature Importance
    st.markdown("#### 🔍 Feature Importance")
    feature_names = ['mean_hr', 'std_hr', 'min_hr', 'max_hr', 'delta_hr', 'morning_slope']
    importances = model.feature_importances_
    fig1, ax1 = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax1)
    ax1.set_title("Feature Importance for Sleep Stage Prediction")
    st.pyplot(fig1)

    # 2. PCA Visualization (Optional)
    st.markdown("#### PCA Projection (Demo Visualization)")
    # Fake batch data for context in PCA
    X_demo = pd.DataFrame([feature_vector[0] for _ in range(50)], columns=feature_names)
    X_demo.iloc[:, :] += np.random.normal(0, 0.1, X_demo.shape)
    X_demo.loc[0] = feature_vector[0]

    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_demo)

    fig2, ax2 = plt.subplots()
    ax2.scatter(proj[1:, 0], proj[1:, 1], alpha=0.3, label="Reference Points")
    ax2.scatter(proj[0, 0], proj[0, 1], color='red', label='Your Night', s=100)
    ax2.set_title("2D PCA Projection of Feature Vector")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.legend()
    st.pyplot(fig2)


else:
    st.info("Upload a CSV file to begin.")
