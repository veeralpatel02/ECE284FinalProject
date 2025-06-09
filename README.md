# ECE284FinalProject

**Sleep Stage Prediction Model**

This project classifies sleep stages using wearable and biosignal data. It supports both unimodal and multimodal models, and includes a web app to visualize predictions. This repo was built for the ECE 284 final project.
---

## Project Structure

```
ECE284FinalProject/
│
├── Application/
│   ├── app.py                     # Web app using Streamlit
│   ├── model_def.py               # Defines the CNN-BiLSTM model
│   ├── rf_sleep_stage_model.pkl   # Random Forest model (trained)
│   ├── train_model.py             # Trains deep learning model
│   └── uploads/                   # Where user-uploaded data is saved
│
├── data/
│   ├── *.txt                      # Raw Apple Watch files: HR, ACC, steps
│   ├── *.csv                      # Cleaned data files (from DREAMT or Apple)
│   ├── S002_PSG_df_updated.csv
│   ├── S005_PSG_df_updated.csv
│   ├── S008_PSG_df_updated.csv
│   ├── PersonalAppleData.csv
│   └── sleep_data.csv
│
├── MultiModal/
│   ├── Final/                     # Final tuned models
│   ├── Initial/                   # First versions
│   ├── Smoothing/                 # Experiments for smoothing predictions
│   ├── best_model.pt              # Saved best model checkpoint
│   ├── model_weights.pt           # General model checkpoint
│   ├── dreamt.pt                  # Model trained on DREAMT dataset
│   ├── sleep_edf_pretrained.pt    # Pretrained on Sleep-EDF
│   ├── test_validation.py         # Test external validation subjects
│   ├── *.ipynb                    # Notebooks for training/testing
│   ├── *.pdf                      # Plots and metrics for CNN-BiLSTM
│   └── recommended_wakeup_times.csv # Smart alarm predictions
│
├── Presentation/                 # Slides, presentation materials
├── RandomForest/                 # Code related to RF models
├── sleep-cassette/               # Other experimental folder
├── sleep-telemetry/              # Signal decoding & telemetry processing
│
├── UnimodalModel/
│   ├── DifferentModalities.png   # Sensor modality comparison
│   ├── unimodal_sleep_pipeline.ipynb # Notebook for single-sensor model
│   └── Unimodal.png              # Architecture image
│
└── README.md                     # This file
```

---

## How to Use

### 1. Install Requirements
If you are using Streamlit and PyTorch:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit torch pandas matplotlib scikit-learn
```

### 2. Train Model
To train the deep learning model using data in `/data`:
```bash
cd Application
python train_model.py
```

### 3. Run the Streamlit App
This launches a local web interface:
```bash
streamlit run app.py
```

### 4. Inference Using Pretrained Model
Run inference on Apple Watch or DREAMT data:
```bash
python test_validation.py
```
Edit the script to point to your CSV files.

---

## Data Sources

- **DREAMT**: S002, S005, S008 PSG data used for training.
- **Apple Watch**: Personal CSV with HR and motion data.

## Output

- Confusion matrix images saved in MultiModal/
- Smart alarm predictions in `recommended_wakeup_times.csv`
- Model weights: `model_weights.pt`, `dreamt.pt`, etc.
