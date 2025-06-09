# ECE284FinalProject
Sleep Stage Prediction Model

# ECE284FinalProject

This project classifies sleep stages using wearable and biosignal data. It supports both unimodal and multimodal models, and includes a web app to visualize predictions. This repo was built for ECE 284 final.

## 📁 Project Structure

ECE284FinalProject/
│
├── Application/
│ ├── app.py # Web app using Streamlit
│ ├── model_def.py # Defines the CNN-BiLSTM model
│ ├── rf_sleep_stage_model.pkl # Random Forest model (trained)
│ ├── train_model.py # Trains deep learning model
│ └── uploads/ # Where user-uploaded data is saved
│
├── data/
│ ├── *.txt # Raw Apple Watch files: HR, ACC, steps
│ ├── *.csv # Cleaned data files (from DREAMT or Apple)
│ ├── S002_PSG_df_updated.csv
│ ├── S005_PSG_df_updated.csv
│ ├── S008_PSG_df_updated.csv
│ ├── PersonalAppleData.csv
│ └── sleep_data.csv
│
├── MultiModal/
│ ├── Final/ # Final tuned models
│ ├── Initial/ # First versions
│ ├── Smoothing/ # Experiments for smoothing predictions
│ ├── best_model.pt # Saved best model checkpoint
│ ├── model_weights.pt # General model checkpoint
│ ├── dreamt.pt # Model trained on DREAMT dataset
│ ├── sleep_edf_pretrained.pt # Pretrained on Sleep-EDF
│ ├── test_validation.py # Test external validation subjects
│ ├── *.ipynb # Notebooks for training/testing
│ ├── *.pdf # Plots and metrics for CNN-BiLSTM
│ └── recommended_wakeup_times.csv # Smart alarm predictions
│
├── Presentation/ # Slides, presentation materials
├── RandomForest/ # Code related to RF models
├── sleep-cassette/ # Other experimental folder
├── sleep-telemetry/ # Signal decoding & telemetry processing
│
├── UnimodalModel/
│ ├── DifferentModalities.png # Sensor modality comparison
│ ├── unimodal_sleep_pipeline.ipynb # Notebook for single-sensor model
│ └── Unimodal.png # Architecture image
│
└── README.md # This file


---

## 🚀 How to Use

### 1. Install Requirements
If you are using Streamlit and PyTorch:
```bash
pip install -r requirements.txt
You may also manually install common dependencies:

bash
Copy
Edit
pip install streamlit torch pandas matplotlib scikit-learn
2. Train Model
To train the deep learning model using data in /data:

bash
Copy
Edit
cd Application
python train_model.py
3. Run the Streamlit App
This launches a local web interface to try out sleep stage predictions:

bash
Copy
Edit
streamlit run app.py
4. Inference Using Pretrained Model
Run inference on Apple Watch or DREAMT data:

bash
Copy
Edit
python test_validation.py
Edit this script to point to your CSV files.

📚 Data Sources
DREAMT: S002, S005, S008 PSG data used for training.

Personal Apple Watch: Custom CSV with HR and motion for testing.

Data is stored in /data.

📈 Output
Confusion matrix images in MultiModal/

Predicted wake-up times: recommended_wakeup_times.csv

Model weights: model_weights.pt, dreamt.pt, etc.

