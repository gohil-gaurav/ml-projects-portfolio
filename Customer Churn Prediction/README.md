# Customer Churn Prediction Dashboard

An end-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset, with an interactive Streamlit app for real-time predictions and business insights.

## Project Overview

Customer churn prediction helps businesses identify users who are likely to leave, so they can take proactive retention actions.

This project includes:
- Data cleaning and preprocessing
- Feature encoding and scaling
- Model training and evaluation
- Saved model artifacts for inference
- A production-style Streamlit web app

## Dataset

- Name: Telco Customer Churn
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib, Seaborn (used in notebook analysis)

## Project Structure

```text
Customer Churn Prediction/
├── app.py               # Streamlit application
├── notebook.ipynb       # Data analysis + model training workflow
├── customer_churn.csv   # Dataset (local copy)
├── churn_model.pkl      # Trained Random Forest model
├── scaler.pkl           # Trained StandardScaler
└── requirements.txt     # Python dependencies
```

## Model Details

The final app uses a Random Forest classifier with preprocessing compatible with the training notebook:
- Target: `Churn` (1 = churn, 0 = no churn)
- Categorical encoding: LabelEncoder (column-wise)
- Feature scaling: StandardScaler
- Class balancing approach explored in notebook (SMOTE)

Saved inference artifacts:
- `churn_model.pkl`
- `scaler.pkl`

## Key App Features

- Interactive form to enter customer profile data
- Churn probability prediction
- Risk band output (Low / Medium / High)
- Retention-oriented recommendation message
- Visual insights tab (churn rate, contract/payment trends)
- Dark-themed dashboard UI

## How to Run Locally

### 1) Clone the repository

```bash
git clone https://github.com/gohil-gaurav/ml-projects-portfolio.git
cd "Customer Churn Prediction"
```

### 2) Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app

```bash
streamlit run app.py
```

### 5) Open in browser

By default:
- http://localhost:8501

## Reproducibility Notes

- If `churn_model.pkl` and `scaler.pkl` are present, the app loads them directly.
- If model files are missing, the app can train a fallback model from the dataset.
- For best compatibility, use the same or close Scikit-learn version as training when loading `.pkl` files.

## Typical Business Use Cases

- Identify high-risk customers for targeted retention campaigns
- Prioritize customer success outreach
- Improve contract and payment strategy using churn insights
- Support data-driven decisions in product and CRM teams

## Future Improvements

- Batch prediction from uploaded CSV
- Explainability (feature importance / SHAP)
- Model versioning and experiment tracking
- Docker deployment
- CI checks for quality and reproducibility

## Author

Built as an ML portfolio project for practical churn prediction and deployment demonstration.
