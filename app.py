import streamlit as st
import pickle
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Intelligent Credit Risk Scoring",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("dt_model.pkl", "rb"))

package = load_model()
model = package["model"]
scaler = package["scaler"]
encoders = package["encoders"]

# -------------------------
# FEATURES (MUST MATCH TRAINING)
# -------------------------
FEATURES = [
    'person_age',
    'person_income($)',
    'person_home_ownership',
    'person_emp_length',
    'loan_intent',
    'loan_amnt($)',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_default_on_file',
    'cb_person_cred_hist_length'
]

# -------------------------
# HEADER
# -------------------------
st.title("💳 Intelligent Credit Risk Scoring System")
st.markdown("""
This ML-powered system evaluates borrower profiles and predicts 
**loan default probability** using a trained Decision Tree model.
""")

st.divider()

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("📌 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income ($)", 0, 1_000_000, 50000)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    person_emp_length = st.number_input("Employment Length (Years)", 0.0, 40.0, 5.0)

with col2:
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
         "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    loan_amnt = st.number_input("Loan Amount ($)", 0, 1_000_000, 10000)
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
    loan_percent_income = st.number_input("Loan to Income Ratio", 0.0, 1.0, 0.2)
    cb_person_default_on_file = st.selectbox("Historical Default", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History (Years)", 0, 50, 3)

st.divider()

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("🚀 Predict Credit Risk"):

    input_data = pd.DataFrame([[
        person_age,
        person_income,
        person_home_ownership,
        person_emp_length,
        loan_intent,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_default_on_file,
        cb_person_cred_hist_length
    ]], columns=FEATURES)

    try:
        # Encode categorical columns
        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        st.subheader("📊 Risk Assessment Result")

        if prediction == 1:
            st.error(f"⚠ High Risk of Default\n\nProbability: **{probability:.2f}%**")
        else:
            st.success(f"✅ Low Risk Applicant\n\nProbability: **{probability:.2f}%**")

        st.progress(int(probability))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------
# BATCH PROCESSING
# -------------------------
st.divider()
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    if st.button("Process Batch File"):
        try:
            processed_df = batch_df.copy()

            # Encode
            for col, encoder in encoders.items():
                if col in processed_df.columns:
                    processed_df[col] = encoder.transform(processed_df[col])

            # Ensure correct column order
            processed_data = scaler.transform(processed_df[FEATURES])

            predictions = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)[:, 1]

            batch_df["Risk_Class"] = [
                "High Risk" if p == 1 else "Low Risk"
                for p in predictions
            ]
            batch_df["Default_Probability"] = probabilities

            st.dataframe(batch_df)

            st.download_button(
                "Download Results",
                batch_df.to_csv(index=False),
                "credit_risk_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Batch Processing Error: {e}")

# -------------------------
# MODEL PERFORMANCE (Optional for Milestone 1)
# -------------------------
st.divider()
st.subheader("📈 Model Performance Summary")

st.write("""
- Decision Tree Classifier (max_depth tuned)
- Training Accuracy ≈ 92.9%
- Testing Accuracy ≈ 91.0%
- Balanced generalization performance
""")

st.caption("Milestone 1 – ML-Based Credit Risk Scoring System")