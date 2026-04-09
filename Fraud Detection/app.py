from pathlib import Path
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_PATH = Path(__file__).with_name("fraud_detection_model.pkl")
DATA_PATH = Path(__file__).with_name("Fraud_Detection_Dataset.csv")
MODEL_FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def _is_pickle_compat_error(exc):
    text = f"{type(exc).__name__}: {exc}"
    return "_RemainderColsList" in text or "sklearn.compose._column_transformer" in text


def rebuild_model_from_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_cols = MODEL_FEATURES + ["isFraud"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[MODEL_FEATURES].copy()
    y = df["isFraud"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return rebuild_model_from_dataset()

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded_model = joblib.load(MODEL_PATH)

        if any("InconsistentVersionWarning" in str(w.category.__name__) for w in caught):
            return rebuild_model_from_dataset()

        return loaded_model
    except Exception as exc:
        if _is_pickle_compat_error(exc):
            return rebuild_model_from_dataset()
        raise


@st.cache_data
def get_transaction_types():
    defaults = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

    if not DATA_PATH.exists():
        return defaults

    try:
        values = pd.read_csv(DATA_PATH, usecols=["type"])["type"].dropna().astype(str).unique().tolist()
        values = sorted(values)
        return values if values else defaults
    except Exception:
        return defaults


def build_input_frame(
    tx_type,
    amount,
    old_balance_sender,
    new_balance_sender,
    old_balance_receiver,
    new_balance_receiver,
):
    return pd.DataFrame(
        [
            {
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_balance_sender,
                "newbalanceOrig": new_balance_sender,
                "oldbalanceDest": old_balance_receiver,
                "newbalanceDest": new_balance_receiver,
            }
        ]
    )


st.set_page_config(page_title="Fraud Detection", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(1200px 700px at 15% -20%, #1a2138 0%, #0d1222 42%, #070b17 100%);
            color: #e6ebff;
        }
        .stSelectbox label, .stNumberInput label {
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #3a4466;
            background: linear-gradient(180deg, #1b2340 0%, #141a30 100%);
            color: #eaf0ff;
            padding: 0.55rem 1.25rem;
            font-weight: 700;
        }
        .stButton > button:hover {
            border-color: #4f5f8f;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fraud Detection Predictor")
st.caption("Enter transaction details and click Predict.")

transaction_types = get_transaction_types()

transaction_type = st.selectbox("Transaction Type", options=transaction_types, index=0)
amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
old_balance_sender = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
new_balance_sender = st.number_input("New Balance (Sender)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
old_balance_receiver = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
new_balance_receiver = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0, format="%.2f")

if st.button("Predict"):
    try:
        model = load_model()
        features = build_input_frame(
            transaction_type,
            amount,
            old_balance_sender,
            new_balance_sender,
            old_balance_receiver,
            new_balance_receiver,
        )

        pred = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            fraud_prob = float(model.predict_proba(features)[0][1])
            st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")

        if pred == 1:
            st.error("This transaction looks related to fraud. Please review it carefully.")
        else:
            st.success("This transaction does not look like fraud.")

    except Exception as exc:
        st.error("Prediction failed. Please check model file and inputs.")
        st.exception(exc)
