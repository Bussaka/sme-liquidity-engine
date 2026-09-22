# SME Liquidity Risk Engine

## Overview
This project is an automated machine learning pipeline designed to detect early signs of financial hardship and liquidity stress in Small and Medium Enterprises (SMEs) in Kenya. By analyzing mobile money transactions, cash runways, and momentum shifts, this system flags high-risk accounts before a critical cash bleed occurs.

## Architecture & Methodology
The system is built on strict machine learning principles focusing on data integrity, model calibration, and generalization to unseen data.

* Gradient Tree Boosting: The core predictor uses Scikit-Learn's `HistGradientBoostingClassifier`, selected for its high performance on tabular financial data and native handling of missing values.
* Leak-Free Encoding: Categorical variables were processed using Out-of-Fold (OOF) target encoding to prevent conditional prediction shifts and target leakage.
* Adversarial Validation: A secondary classifier was deployed to detect covariate shift between the training and production data distributions, proving the model generalizes safely (ROC-AUC 0.4990).
* F1-Optimized Thresholding: The decision boundary was mathematically shifted from the default 0.50 to an aggressive 0.25 to maximize the F1-Score, prioritizing early risk detection over conservative precision.

## Model Interpretability
Black-box algorithms are insufficient for forensic financial tooling. This engine utilizes SHAP (SHapley Additive exPlanations) to calculate exact feature contributions. If an account is flagged for liquidity stress, the system can instantly isolate the specific behaviors (e.g., shrinking cash buffers or rapid outflow ratios) driving the alarm.

## Repository Structure
* `data/`: Contains sample datasets (CSV files ignored via `.gitignore`).
* `notebooks/`: Jupyter notebooks detailing the EDA, model training, SHAP analysis, and adversarial validation.
* `api.py`: The FastAPI deployment script serving the prediction endpoint.
* `liquidity_risk_engine.pkl`: The serialized Scikit-Learn model and preprocessing package.
* `requirements.txt`: Python package dependencies.

## Local Setup & Deployment
The intelligence engine is decoupled from the training data and serialized via `joblib`. It is wrapped in a lightweight **FastAPI** microservice, allowing it to ingest live JSON payloads and instantly return a binary risk flag.

1. Clone the repository and navigate to the project folder.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Boot up the FastAPI server:
   `python api.py`
4. The model will be live at `http://127.0.0.1:8000/evaluate_risk`.

## API Usage Example
Endpoint: `POST /evaluate_risk`

Example Request Payload:
```json
{
  "features": {
    "m1_daily_avg_bal": 12500.50,
    "m2_daily_avg_bal": 14000.00,
    "m3_daily_avg_bal": 15500.25,
    "m4_daily_avg_bal": 16000.00,
    "m5_daily_avg_bal": 16200.00,
    "m1_deposit_total_value": 45000.00,
    "m2_deposit_total_value": 48000.00,
    "bal_ratio_m1_m6": 0.85,
    "liquidity_buffer_ratio": 1.2
  }
}