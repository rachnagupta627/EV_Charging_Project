# ⚡ EV Charging Demand Prediction and Usage Pattern Analysis

**Domain:** Electric Vehicles (EV) & Sustainable Transportation  
**Skills Covered:** Data Cleaning & Preprocessing • Exploratory Data Analysis (EDA) • Feature Engineering • Regression & Classification • Time Series Analysis & Forecasting • Model Evaluation & Comparison • Explainable AI (SHAP & LIME) • Data Visualization • Python & Scikit-Learn • XGBoost & LightGBM • Streamlit Dashboard Development

---

## 📌 Problem Statement

Develop an end-to-end machine learning system to analyze EV charging station usage patterns and predict charging demand based on factors such as charger type, location, charging company, time of charging, and historical usage behavior.

This project addresses three core machine learning tasks alongside interactive explainability and deployment:
1. **Regression — Charging Demand Prediction:** Predict the exact energy demand (`Demand`) of an EV charging session using session-start and historical charger features, evaluated with `MAE`, `MSE`, `RMSE`, and `R²`.
2. **Classification — High-Demand Session Prediction:** Classify whether a charging session falls into **High Demand** ($\ge 14.2$) or **Normal/Low Demand** ($< 14.2$), evaluated with `Accuracy`, `Precision`, `Recall`, `F1-Score`, and `ROC-AUC`.
3. **Time Series — Future Charging Demand Forecasting:** Analyze daily charging demand trends and weekly seasonality using Moving Averages and Holt-Winters Exponential Smoothing to forecast future demand, evaluated with `MAE` and `RMSE`.
4. **Explainable AI & Streamlit Dashboard:** Interpret model predictions globally and locally using **Feature Importance**, **SHAP**, and **LIME**, presented in an interactive multi-page Streamlit dashboard.

---

## 💼 Business Use Cases

- **Charging Station Demand Forecasting:** Anticipate busy and low-demand periods across times of day, days of the week, and months.
- **EV Charging Station Planning:** Identify high-demand locations (`apartment`, `public area`, `resort`, `hotel`, `company`) where additional charging infrastructure is needed.
- **Resource & Capacity Planning:** Optimize grid load allocation and charger availability based on expected energy demand.
- **Peak Demand Management:** Manage evening peak loads (`17:00–20:00`) and reduce station congestion.
- **Charger Utilization Analysis:** Compare fast vs. slow charger types (`ChargerType 0` vs. `1`) and companies (`ChargerCompany 0` vs. `1`).
- **Operational Optimization & Customer Experience:** Improve maintenance scheduling, reduce waiting times, and support long-term capacity planning.

---

## 📂 Dataset Overview

- **Source:** EV Charging Transaction Records (`Data_row/ChargingRecords.csv`)
- **Raw Records:** `72,856` sessions (`13` columns)
- **Cleaned Records:** `72,843` sessions (`19` columns)
- **Historical Coverage:** `2021-09-30` to `2022-09-30` (`2,337` unique users across `2,119` unique chargers and `14` location types)

### Key Variables
| Column | Description |
| :--- | :--- |
| `UserID` | Unique identifier for the EV user |
| `ChargerID` | Unique identifier for the charging station/unit |
| `ChargerCompany` | Charging network operator code (`0` or `1`) |
| `Location` | Charging station category (`hotel`, `apartment`, `public area`, `resort`, `company`, etc. — 14 categories) |
| `ChargerType` | Charger type category (`0` = standard/slow ~79.5%, `1` = fast ~20.5%) |
| `StartDay` / `StartTime` | Session start date and timestamp |
| `EndDay` / `EndTime` | Session end date and timestamp |
| `StartDatetime` / `EndDatetime` | Combined start and end timestamps |
| `Duration` / `CleanedDuration` | Charging session duration in minutes (cleaned using second-level precise timestamps) |
| `Demand` | **Target Variable (Regression):** Energy demand for the charging session (mean `17.44`, median `14.10`, range `0.01–97.00`) |
| `HighDemand` | **Target Variable (Classification):** `1` if `Demand >= 14.2` (training median), `0` if `Demand < 14.2` |

---

## 🔄 Project Workflow & Architecture

```
Data Understanding ➔ Data Cleaning ➔ Exploratory Data Analysis (EDA) ➔ Feature Engineering
       ➔ Regression Modeling ➔ Classification Modeling ➔ Time-Series Forecasting
       ➔ Explainable AI (SHAP & LIME) ➔ Streamlit Dashboard Deployment
```

### 1. Data Understanding & Cleaning (`01_data_understanding.ipynb`, `02_data_cleaning.ipynb`)
- Removed `13` exact duplicate rows (`72,856` $\rightarrow$ `72,843` records).
- Investigated `718` zero-duration and `17` negative-duration sessions by computing second-level `PreciseDuration` from `StartDay`/`StartTime` and `EndDay`/`EndTime`.
- Recovered `691` valid sessions that had positive elapsed time and marked the remaining `44` invalid/unresolved durations (`17` negative + `27` zero) as `NaN` in `CleanedDuration`.
- Saved cleaned dataset to `Data_row/cleaned data/ChargingRecords_Cleaned.csv`.

### 2. Exploratory Data Analysis (`03_EDA.ipynb`)
- **Hourly Patterns:** Charging sessions and total demand peak during evening hours (`17:00–20:00`, with hour `18` highest at `6,509` sessions and `119,394.82` demand).
- **Location Patterns:** `public area` (`14,076` sessions) and `apartment` (`14,036` sessions; highest total demand at `263,103.12`) dominate volume, while `accommodation` (`29.06`) and `bus garage` (`28.59`) have the highest average demand per session.
- **Charger Type Patterns:** `ChargerType 0` has longer average duration (`181.42` min, avg demand `17.05`), whereas `ChargerType 1` delivers slightly higher average demand (`18.96`) in a much shorter average duration (`36.05` min).
- **Duration vs. Demand:** Strong positive correlation (`r = 0.6585`).
- **Temporal Growth:** Average daily demand more than doubled from October 2021 (`2,094.63`/day) to August 2022 (`4,614.23`/day).

### 3. Feature Engineering (`04_feature_engineering.ipynb` & `05_regression.ipynb`)
- Extracted temporal features: `StartHour`, `DayOfWeek`, `Month`, `IsWeekend`, and `TimeOfDay` (`Morning`, `Afternoon`, `Evening`, `Night`).
- Applied One-Hot Encoding to `Location`, `ChargerType`, and `TimeOfDay` (`Data_row/feature_engineered/EV_Features.csv`).
- Engineered leak-free historical feature `PreviousChargerDemand` using `pd.merge_asof` to match each session with the most recent completed session on the same `ChargerID`.

---

## 📊 Model Evaluation & Results

All supervised models use an **80/20 chronological train-test split** (`58,274` training rows from Sep 2021–Jul 2022; `14,569` testing rows from Jul 2022–Sep 2022) to prevent future data leakage.

### 1. Regression Models (`05_regression.ipynb`)
| Model | MAE | MSE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM (with Historical Demand)** | **9.5494** | **159.9609** | **12.6476** | **0.1738** |
| LightGBM | 9.9771 | 170.9370 | 13.0743 | 0.1171 |
| XGBoost | 10.1632 | 172.6865 | 13.1410 | 0.1081 |
| Gradient Boosting | 10.0443 | 173.0961 | 13.1566 | 0.1060 |
| Decision Tree | 10.1560 | 176.7558 | 13.2950 | 0.0871 |
| Random Forest | 10.1109 | 177.6567 | 13.3288 | 0.0824 |
| Linear Regression | 10.3957 | 182.6671 | 13.5154 | 0.0566 |

### 2. Classification Models (`06_classification.ipynb`)
Target threshold: `Demand >= 14.2` (derived from training median; naturally balanced at `50.03%` High Demand vs. `49.97%` Normal Demand).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.6436** | **0.6152** | **0.7180** | **0.6626** | **0.6932** |
| XGBoost | 0.6372 | 0.6167 | 0.6756 | 0.6448 | 0.6839 |
| Logistic Regression | 0.6171 | 0.5914 | 0.6942 | 0.6387 | 0.6622 |
| SVM | 0.6149 | 0.5891 | 0.6942 | 0.6373 | 0.6607 |
| Decision Tree | 0.6158 | 0.5934 | 0.6732 | 0.6308 | 0.6630 |
| Random Forest | 0.6298 | 0.6140 | 0.6478 | 0.6305 | 0.6756 |
| KNN | 0.6163 | 0.6013 | 0.6318 | 0.6162 | 0.6568 |

### 3. Time-Series Forecasting Models (`07_demand_forecasting.ipynb`)
Evaluated on `365` complete days (`292` training days / `73` testing days), followed by a 30-day future forecast (`2022-10-01` to `2022-10-30`).

| Forecasting Model | MAE | RMSE |
| :--- | :---: | :---: |
| **Moving Average (7-Day)** | **588.84** | **810.42** |
| Exponential Smoothing | 773.73 | 1014.07 |
| Weekly Seasonal Forecasting (Holt-Winters) | 784.61 | 1018.28 |

---

## 🗂️ Project Structure

```text
EV_Charging_Project/
├── 01_data_understanding.ipynb        # Initial exploration, schema, missing/duplicate/duration checks
├── 02_data_cleaning.ipynb             # Duplicate removal & timestamp-based duration recovery
├── 03_EDA.ipynb                       # Exploratory Data Analysis across hours, locations, types, dates
├── 04_feature_engineering.ipynb       # Temporal extraction & one-hot encoding pipeline
├── 05_regression.ipynb                # Regression model training, historical lag engineering & evaluation
├── 06_classification.ipynb            # High/Normal demand classification training & evaluation
├── 07_demand_forecasting.ipynb        # Daily time-series analysis, smoothing & 30-day forecasting
├── app.py                             # Interactive Streamlit Dashboard with SHAP & LIME explainability
├── README.md                          # Project documentation and setup guide
├── Data_row/
│   ├── ChargingRecords.csv            # Original raw dataset (72,856 records)
│   ├── cleaned data/
│   │   └── ChargingRecords_Cleaned.csv# Cleaned dataset (72,843 records)
│   └── feature_engineered/
│       └── EV_Features.csv            # Engineered feature dataset (72,843 × 25)
├── models/
│   ├── lightgbm_regression.pkl        # Trained LightGBM regression pipeline
│   ├── lightgbm_classification.pkl    # Trained LightGBM classification pipeline
│   └── high_demand_threshold.pkl      # Saved median threshold (14.2)
└── forecasts/
    ├── future_charging_demand.csv     # 30-day future demand predictions
    └── forecast_model_comparison.csv  # Forecasting evaluation metrics (MAE, RMSE)
```

---

## 🖥️ Streamlit Dashboard Features (`app.py`)

The Streamlit application (`app.py`) includes sidebar filters (`Location` and `ChargerType`) and 6 interactive pages:
1. **Overview:** High-level KPI metrics (`Total Sessions`, `Total Demand`, `Average Demand`, `Unique Chargers`), demand by location, demand by charger type, and daily demand trend line.
2. **Data Analysis:** Hourly session distribution (`0–23`) and total charging demand by day of the week (`Monday–Sunday`).
3. **Regression Prediction:** Interactive calculator to predict session charging demand using `lightgbm_regression.pkl`.
4. **Classification Prediction:** Interactive classifier predicting **HIGH** ($\ge 14.2$) vs. **LOW** ($< 14.2$) charging demand using `lightgbm_classification.pkl`.
5. **Demand Forecasting:** Displays the 30-day future demand table and interactive trend chart.
6. **Model Insights:**
   - **Model Performance Comparison:** Full evaluation tables for Regression, Classification, and Time-Series models.
   - **Feature Importance:** Split-importance bar charts and tables for LightGBM Regression and Classification.
   - **SHAP Explainability:** Global `Mean |SHAP Value|` impact charts and SHAP summary beeswarm plots.
   - **LIME Local Explanation:** Interactive session-level local explanations with contribution charts and feature rules.

---

## 🚀 Installation & How to Run

### 1. Clone or Navigate to the Project Directory
```bash
cd /Users/USERNAME/Desktop/EV_Charging_Project
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn lightgbm xgboost statsmodels matplotlib shap lime joblib
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will launch in your browser at:
- **Local URL:** `http://localhost:8501`
