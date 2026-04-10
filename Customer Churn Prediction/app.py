from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="centered",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;700&display=swap');

            :root {
                --bg: #0a0c10;
                --ink: #e9edf6;
                --muted: #9aa6bf;
                --card: #121721;
                --card-2: #171d29;
                --brand: #5aa2ff;
                --brand-2: #2dd4bf;
                --ok: #22c55e;
                --warn: #f59e0b;
                --bad: #ef4444;
            }

            html, body, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 16% 12%, rgba(90, 162, 255, 0.22), transparent 34%),
                    radial-gradient(circle at 88% 86%, rgba(45, 212, 191, 0.15), transparent 36%),
                    linear-gradient(145deg, #090b0f 0%, #0f1219 46%, #101521 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: rgba(10, 12, 16, 0.62);
                backdrop-filter: blur(4px);
            }

            [data-testid="stToolbar"] {
                right: 1rem;
            }

            [data-testid="stMainBlockContainer"] {
                max-width: 1040px;
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4 {
                font-family: 'Space Grotesk', sans-serif !important;
                color: var(--ink);
                letter-spacing: -0.02em;
            }

            p, label, li, div[data-testid="stMarkdownContainer"] {
                font-family: 'Manrope', sans-serif !important;
                color: var(--ink);
            }

            small, span, .stCaption {
                color: var(--muted) !important;
            }

            .hero {
                background: linear-gradient(125deg, rgba(18, 24, 35, 0.95), rgba(13, 18, 28, 0.96));
                border-radius: 22px;
                padding: 26px 28px;
                margin-bottom: 14px;
                box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
                border: 1px solid rgba(123, 153, 204, 0.3);
            }

            .hero h1 {
                color: #ffffff !important;
                margin: 0;
                font-size: 2rem;
            }

            .hero p {
                color: rgba(233, 237, 246, 0.88);
                margin-top: 8px;
                margin-bottom: 0;
                font-size: 1rem;
            }

            .data-chip {
                display: inline-block;
                background: rgba(90, 162, 255, 0.12);
                color: #cfe4ff;
                border: 1px solid rgba(90, 162, 255, 0.38);
                border-radius: 999px;
                padding: 5px 11px;
                font-size: 0.83rem;
                margin-right: 8px;
                margin-top: 8px;
            }

            .result-card {
                background: var(--card);
                border: 1px solid rgba(112, 138, 184, 0.25);
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3);
                margin-top: 8px;
            }

            .result-title {
                font-weight: 700;
                color: var(--ink);
                margin-bottom: 6px;
            }

            .risk-low { color: var(--ok); font-weight: 700; }
            .risk-mid { color: var(--warn); font-weight: 700; }
            .risk-high { color: var(--bad); font-weight: 700; }

            div[data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid rgba(112, 138, 184, 0.2);
                border-radius: 14px;
                padding: 10px 12px;
            }

            [data-baseweb="tab-list"] {
                gap: 0.25rem;
                background: transparent;
            }

            [data-baseweb="tab"] {
                color: #a8b5ce !important;
            }

            [aria-selected="true"] {
                color: #f5f9ff !important;
                border-bottom-color: var(--brand) !important;
            }

            [data-baseweb="select"] > div,
            .stNumberInput > div > div,
            .stTextInput > div > div,
            .stDateInput > div > div,
            [data-testid="stSlider"] {
                background: var(--card-2) !important;
            }

            [data-baseweb="select"] > div,
            .stNumberInput > div > div,
            .stTextInput > div > div,
            .stDateInput > div > div {
                border: 1px solid rgba(129, 152, 191, 0.34) !important;
                color: var(--ink) !important;
            }

            [data-baseweb="select"] span,
            .stSelectbox label,
            .stSlider label,
            .stNumberInput label {
                color: #c7d3ea !important;
            }

            .stButton > button {
                border: none;
                border-radius: 12px;
                padding: 0.58rem 1.1rem;
                font-weight: 700;
                background: linear-gradient(90deg, #2f6fbd 0%, #1c4f8e 100%);
                color: white;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 20px rgba(18, 55, 103, 0.45);
            }

            @media (max-width: 900px) {
                .hero h1 { font-size: 1.45rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()
    df = df.drop(columns=["customerID"])
    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)
    return df


@st.cache_resource(show_spinner=False)
def build_preprocessing_objects():
    df = load_clean_data().copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_columns = [c for c in df.columns if c != "Churn"]
    return df, encoders, feature_columns


@st.cache_resource(show_spinner=False)
def get_model_and_scaler():
    encoded_df, _, feature_columns = build_preprocessing_objects()

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, "Loaded from saved files"

    X = encoded_df[feature_columns]
    y = encoded_df["Churn"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, "Trained in app session"


def classify_risk(probability: float) -> tuple[str, str]:
    if probability < 0.30:
        return "Low", "risk-low"
    if probability < 0.60:
        return "Medium", "risk-mid"
    return "High", "risk-high"


def render_header(df: pd.DataFrame, model_note: str) -> None:
    churn_rate = df["Churn"].mean() * 100
    st.markdown(
        f"""
        <div class="hero">
            <h1>Customer Churn Intelligence Dashboard</h1>
            <p>
                Predict churn probability for individual customers, explore churn patterns,
                and drive retention decisions with your trained model.
            </p>
            <span class="data-chip">Rows: {len(df):,}</span>
            <span class="data-chip">Features: {df.shape[1] - 1}</span>
            <span class="data-chip">Churn Rate: {churn_rate:.2f}%</span>
            <span class="data-chip">Model: {model_note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def numeric_input(col_name: str, series: pd.Series):
    min_v = float(series.min())
    max_v = float(series.max())
    median_v = float(series.median())

    if pd.api.types.is_integer_dtype(series):
        return st.slider(
            col_name,
            min_value=int(min_v),
            max_value=int(max_v),
            value=int(median_v),
            step=1,
        )

    return st.slider(
        col_name,
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(round(median_v, 2)),
        step=float(max((max_v - min_v) / 100, 0.01)),
    )


def build_input_form(
    clean_df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
) -> dict:
    st.subheader("Customer Profile Input")
    st.caption("Fill customer details and click Predict Churn.")

    col_left, col_right = st.columns(2)
    input_data = {}

    for idx, col in enumerate(feature_columns):
        target_col = col_left if idx % 2 == 0 else col_right
        with target_col:
            if col in encoders:
                options = list(encoders[col].classes_)
                default_index = 0
                if "Month-to-month" in options:
                    default_index = options.index("Month-to-month")
                input_data[col] = st.selectbox(col, options=options, index=default_index)
            elif col == "SeniorCitizen":
                input_data[col] = st.selectbox(
                    "SeniorCitizen",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
            else:
                input_data[col] = numeric_input(col, clean_df[col])

    return input_data


def encode_input(
    input_data: dict,
    encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
) -> pd.DataFrame:
    model_row = {}
    for col in feature_columns:
        value = input_data[col]
        if col in encoders:
            model_row[col] = int(encoders[col].transform([str(value)])[0])
        else:
            model_row[col] = float(value)

    return pd.DataFrame([model_row], columns=feature_columns)


def render_prediction_result(probability: float, prediction: int):
    risk_label, risk_css = classify_risk(probability)
    churn_text = "Churn Likely" if prediction == 1 else "Likely to Stay"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Prediction Result</div>
            <p><strong>Status:</strong> {churn_text}</p>
            <p><strong>Risk Level:</strong> <span class="{risk_css}">{risk_label}</span></p>
            <p><strong>Churn Probability:</strong> {probability * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability >= 0.60:
        st.error(
            "High-risk customer: prioritize retention actions like personalized offers, support follow-up, and contract incentives."
        )
    elif probability >= 0.30:
        st.warning(
            "Moderate risk: monitor engagement and consider a targeted loyalty campaign."
        )
    else:
        st.success("Low risk: customer is currently stable.")


def render_insights(clean_df: pd.DataFrame):
    st.subheader("Portfolio Insights")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Customers", f"{len(clean_df):,}")
    with c2:
        st.metric("Churned Customers", f"{int(clean_df['Churn'].sum()):,}")
    with c3:
        st.metric("Overall Churn Rate", f"{clean_df['Churn'].mean() * 100:.2f}%")

    viz1, viz2 = st.columns(2)

    with viz1:
        contract_stats = (
            clean_df.groupby("Contract", as_index=False)["Churn"].mean().sort_values("Churn")
        )
        contract_stats["ChurnPct"] = contract_stats["Churn"] * 100
        st.markdown("##### Churn Rate by Contract")
        st.bar_chart(contract_stats.set_index("Contract")["ChurnPct"])

    with viz2:
        payment_stats = (
            clean_df.groupby("PaymentMethod", as_index=False)["Churn"]
            .mean()
            .sort_values("Churn")
        )
        payment_stats["ChurnPct"] = payment_stats["Churn"] * 100
        st.markdown("##### Churn Rate by Payment Method")
        st.bar_chart(payment_stats.set_index("PaymentMethod")["ChurnPct"])

    st.markdown("##### Monthly Charges vs Churn")
    sample_df = clean_df[["MonthlyCharges", "Churn"]].copy()
    sample_df["ChurnLabel"] = np.where(sample_df["Churn"] == 1, "Churn", "No Churn")
    st.scatter_chart(sample_df, x="MonthlyCharges", y="Churn")


def main():
    inject_styles()

    clean_df = load_clean_data()
    encoded_df, encoders, feature_columns = build_preprocessing_objects()
    model, scaler, model_note = get_model_and_scaler()

    render_header(clean_df, model_note)

    tab_predict, tab_insights, tab_about = st.tabs(["Predictor", "Insights", "About Project"])

    with tab_predict:
        input_data = build_input_form(clean_df, encoders, feature_columns)

        if st.button("Predict Churn", use_container_width=True):
            try:
                input_frame = encode_input(input_data, encoders, feature_columns)
                scaled_input = scaler.transform(input_frame)

                prediction = int(model.predict(scaled_input)[0])
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(scaled_input)[0][1])
                else:
                    probability = float(prediction)

                render_prediction_result(probability, prediction)

                with st.expander("See Encoded Input Sent to Model"):
                    st.dataframe(input_frame, use_container_width=True)

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    with tab_insights:
        render_insights(clean_df)

    with tab_about:
        st.subheader("How This App Matches Your Notebook")
        st.markdown(
            """
            - Uses the same dataset file and cleaning logic.
            - Encodes categorical features using LabelEncoder column-by-column.
            - Uses StandardScaler before prediction.
            - Loads your saved files churn_model.pkl and scaler.pkl when available.
            - Falls back to in-app training if saved files are missing.
            """
        )
        st.info(
            "Run with: streamlit run app.py"
        )


if __name__ == "__main__":
    main()
