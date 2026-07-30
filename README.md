# 🔧 PredictiveAI - NASA IMS Bearing Fault Diagnosis

This project implements a complete **Predictive Maintenance** pipeline for predicting bearing failure using the **NASA IMS (Intelligent Maintenance Systems) Bearing Dataset**. 

The pipeline processes high-frequency vibration signals, extracts key time-domain statistical features, and trains machine learning models to classify the health state of the bearings into **Healthy** versus **Failing/Degraded**.

---

## 📊 Dataset & Setup

- **Source**: NASA Ames Prognostics Data Repository
- **Experimental Setup**: Four Rexnord ZA-2115 double-row bearings running at **2000 RPM** under a **6000 lbs** radial load.
- **Vibration Signals**: Sampled at **20 kHz** for 1-second snapshots (20,480 points per bearing) every 5-10 minutes.
- **Failures**: In Test Run 2, **Bearing 1** suffered an outer race failure at the end of the run (files after index 700).

---

## 🛠️ Features Extracted (Time-Domain)

For each 1-second snapshot file, the following features are calculated for all 4 bearings:
1. **Mean**: Average signal amplitude.
2. **Standard Deviation (Std)**: Variance in signal energy.
3. **Root Mean Square (RMS)**: Total energy of the vibration signal.
4. **Peak Amplitude**: Maximum absolute amplitude level.
5. **Kurtosis**: Probability density of sharp spikes (impulse indicator).
6. **Skewness**: Symmetry of the vibration profile.
7. **Crest Factor**: Ratio of peak value to RMS (early wear indicator).
8. **Shape Factor**: Ratio of RMS to Mean Absolute Deviation.

---

## 📈 Pipeline Architecture

The pipeline in `main.py` performs the following steps:
1. **Feature Extraction & Caching**: Auto-detects and loads all Test 2 files, extracts 32 features per file, and caches them to `bearing_features.csv` for fast future launches.
2. **Exploratory Data Analysis (EDA)**: Generates degradation trend plots for RMS and Kurtosis, plus sensor correlation heatmaps (saved in `plots/`).
3. **Model Selection**: Splits data into Train/Test sets, normalizes features, and trains 4 models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Classifier (SVM)
   - Random Forest Classifier
4. **Evaluation**: Displays accuracy, precision, recall, and F1-score comparisons, and saves confusion matrices (saved in `plots/model_confusion_matrices.png`).

---

## 🏆 Model Performance Results

Below are the evaluation metrics for the classifiers:

| Classifier Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **98.98%** | **100.00%** | **96.49%** | **0.9821** |
| **SVM (RBF Kernel)** | 97.46% | 98.15% | 92.98% | 0.9550 |
| **Logistic Regression** | 96.95% | 98.11% | 91.23% | 0.9455 |
| **KNN** | 93.91% | 100.00% | 78.95% | 0.8824 |

*The **Random Forest** classifier achieved the best performance with an **F1-Score of 0.9821**.*

---

## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline:
   ```bash
   python main.py
   ```

---

## 📁 Artifacts Generated
- `bearing_features.csv`: Cached tabular feature matrix (984 rows × 35 columns).
- `plots/`:
  - `bearing_rms_trend.png`: Vibration energy progression over time.
  - `bearing_kurtosis_trend.png`: Wearing indicators showing spike levels.
  - `bearing_correlation.png`: Correlation matrix of bearing readings.
  - `model_confusion_matrices.png`: Performance matrix of the 4 models.
- `models/`:
  - `scaler.joblib`: StandardScaler normalization parameters.
  - `best_bearing_model.joblib`: The trained Random Forest classifier.
  - `feature_columns.joblib`: Feature layout specification.
