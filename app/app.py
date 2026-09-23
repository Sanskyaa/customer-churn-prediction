import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.joblib"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "the probability of customer churn."
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)


# -------------------------
# Column 1
# -------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


# -------------------------
# Column 2
# -------------------------

with col2:

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.header("Financial Information")

col3, col4 = st.columns(2)

with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col4:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -------------------------
    # Feature Engineering
    # -------------------------

    if tenure > 0:
        avg_monthly_spend = total_charges / tenure
    else:
        avg_monthly_spend = monthly_charges


    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Growing"
    elif tenure <= 48:
        tenure_group = "Mature"
    else:
        tenure_group = "Loyal"


    service_columns = [
        phone_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    num_services = sum(
        service == "Yes"
        for service in service_columns
    )


    # IMPORTANT:
    # This threshold should match the value used
    # during training.
    high_value_threshold = 70.35

    is_high_value = int(
        monthly_charges > high_value_threshold
    )


    protection_columns = [
        online_security,
        online_backup,
        device_protection,
        tech_support
    ]

    has_protection = int(
        any(service == "Yes" for service in protection_columns)
    )


    # -------------------------
    # Create input DataFrame
    # -------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],

        # Engineered features
        "AvgMonthlySpend": [avg_monthly_spend],
        "TenureGroup": [tenure_group],
        "NumServices": [num_services],
        "IsHighValue": [is_high_value],
        "HasProtection": [has_protection]
    })


    # -------------------------
    # Prediction
    # -------------------------

    churn_probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        churn_probability >= threshold
    )


    # -------------------------
    # Display result
    # -------------------------

    st.header("Prediction Result")

    probability_percent = churn_probability * 100

    st.metric(
        "Churn Probability",
        f"{probability_percent:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ High Churn Risk — Customer is predicted to churn."
        )

    else:

        st.success(
            "✅ Low Churn Risk — Customer is predicted to stay."
        )