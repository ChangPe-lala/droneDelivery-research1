# Drone Energy Consumption Prediction

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-blue.svg)](https://lightgbm.readthedocs.io/)

## Abstract
This research repository focuses on predicting the energy consumption of Unmanned Aerial Vehicles (UAVs/Drones) based on flight kinematics and environmental weather conditions. Accurate energy estimation is critical for safe path planning, optimizing payload delivery operations, and preventing mid-air battery depletion. 

This project processes raw telemetry and weather data to extract meaningful physical features (such as Haversine distance, flight duration, relative wind angle, and energy consumption in Watt-hours). Several machine learning models, ranging from baseline Linear Regression to advanced tree-based ensembles (Random Forest, XGBoost, LightGBM), are evaluated using robust cross-validation techniques (GroupKFold) to prevent data leakage.

## Repository Structure

```text
Energy_Predict_abs/
├── data/               # Contains raw and intermediate datasets (See Data Access)
├── figures/            # Exploratory Data Analysis (EDA) and Model Evaluation plots
├── logs/               # Execution logs and text reports from the pipelines
├── models/             # Serialized pre-trained machine learning models (.pkl)
├── output/             # Processed datasets (clean data, train/test splits)
├── results/            # Performance metrics and cross-validation results (.csv)
├── src/                # Source code directory
│   ├── 00_check_clean_data.py         # Data cleaning, feature engineering, filtering
│   ├── 01_train_test_split.py         # Data splitting using GroupKFold by date
│   ├── 02_train_all_models.py         # Training and evaluating baseline & ensemble models
│   ├── 03_train_model_B_no_duration.py# Training models excluding duration features
│   ├── analyze_data.py                # Exploratory Data Analysis script
│   └── check_total_flight.py          # Data validation utilities
├── .gitignore          # Git ignore file
├── requirements.txt    # Python dependencies
└── README.md           # This document
```

## Dataset & Data Access

Due to GitHub's file size limitations, the large raw dataset `flight_with_weather.csv` is securely hosted externally.

**[Download Raw Dataset (Google Drive)](https://drive.google.com/drive/folders/1CLWcVrpuZaHS277lQgkgCQuWYt3UuZY4?usp=drive_link)**

*Instructions:* Download the dataset from the link above and place it in the `data/` directory as `data/flight_with_weather.csv` before running the pipeline.

### Features Description
The dataset aggregates high-frequency telemetry data into per-flight metrics.
- **Kinematic Features:**
  - `distance` (m): Total Haversine distance derived from GPS coordinates.
  - `flight_duration` (s): Total time elapsed during the flight.
  - `speed` (m/s): Average flight speed.
  - `altitude` (m): Cruising altitude.
  - `payload` (g): Weight of the payload carried by the drone.
- **Environmental & Weather Features:**
  - `temperature` (°C), `humidity` (%), `pressure` (hPa), `cloud_cover` (%).
  - `wind_speed` (m/s), `wind_gust` (m/s), `wind_dir` (°).
  - `relative_wind_angle` (°): The angle between the drone's heading vector and the wind direction, bounded [0, 180].
- **Target Variable:**
  - `energy_consumption_wh` (Wh): Total energy consumed, calculated by integrating power over time: `Σ (V × |I| × Δt) / 3600`.

## Methodology

### 1. Data Preprocessing & Cleaning (`00_check_clean_data.py`)
- **Noise Filtering:** Flights with distance < 30m, duration < 30s, or energy < 1 Wh are filtered out as hover/calibration tests.
- **Feature Engineering:** Calculated vectorised Haversine distance and relative wind angle.
- **Target Calculation:** Energy consumption derived via Riemann sum of battery voltage and current.

### 2. Train-Test Splitting (`01_train_test_split.py`)
- Standard random splitting can cause **data leakage** because flights from the same day share identical weather conditions. 
- Implemented `GroupKFold` grouped by `date` to ensure that flights occurring on the same day are entirely isolated into either the training or the testing set.

### 3. Model Training & Evaluation (`02_train_all_models.py`)
Five models are benchmarked:
1. **Linear Regression** (Baseline)
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **XGBoost Regressor**
5. **LightGBM Regressor**

Models are evaluated across multiple metrics: $R^2$, Adjusted $R^2$, MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE. 
SHAP (SHapley Additive exPlanations) values are used for model interpretability to quantify feature contributions.

## Getting Started

### Prerequisites
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipeline
Execute the scripts sequentially from the root directory:

1. **Clean and Prepare Data:**
```bash
python src/00_check_clean_data.py
```
2. **Split Data (Train/Test):**
```bash
python src/01_train_test_split.py
```
3. **Train Models and Generate Results:**
```bash
python src/02_train_all_models.py
```

## Results & Artifacts
Upon successful execution, the pipeline generates the following artifacts:
- **`figures/`**: EDA distributions, Model Prediction vs. Actual scatter plots, Residual plots, and SHAP Summary plots.
- **`results/`**: `cv_results.csv` and `test_results.csv` detailing the cross-validation and hold-out test performance of all models.
- **`models/`**: Serialized `.pkl` files of the trained pipelines for future deployment/inference.
