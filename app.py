import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Feature Definition ---
FEATURES = [
    'person_age', 'person_income($)', 'person_home_ownership', 
    'person_emp_length', 'loan_intent', 'loan_amnt($)', 
    'loan_int_rate', 'loan_percent_income', 
    'cb_person_default_on_file', 'cb_person_cred_hist_length'
]

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-attachment: fixed;
        color: #ffffff;
    }

    /* Glassmorphism Containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Input Styling Override */
    .stNumberInput, .stSelectbox {
        background-color: transparent !important;
    }
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    div[data-baseweb="base-input"] input {
        color: white !important;
    }
    label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #1a1a1a;
        font-weight: 700;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.6);
        color: #000;
    }

    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Typography */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(to right, #ffffff, #b3e5fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
    }
    h2, h3 {
        color: #ffffff !important;
    }
    .subtitle {
        color: #b0bec5;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("dt_model.pkl", "rb"))
    except FileNotFoundError:
        return None

package = load_model()

if package:
    model = package["model"]
    scaler = package["scaler"]
    encoders = package["encoders"]
else:
    model = None

# --- Header Section ---
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Credit Risk AI Assessment")
    st.markdown('<p class="subtitle">Advanced machine learning model for predicting loan default probability with high precision.</p>', unsafe_allow_html=True)

# --- Main Content ---
if not model:
    st.error("⚠️ Model file 'dt_model.pkl' not found. Please upload it to the directory.")
else:
    # Form Container
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Applicant Details")
    
    with st.container():
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            person_age = st.number_input("Age", 18, 120, 30, help="Applicant's age in years")
            person_income = st.number_input("Annual Income ($)", 0, None, 50000, 1000, help="Gross annual income")
        with row1_col2:
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            person_emp_length = st.number_input("Employment Length (Years)", 0.0, None, 5.0, 0.5)
        with row1_col3:
            loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            loan_amnt = st.number_input("Loan Amount ($)", 0, None, 10000, 500)
            loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0, 0.1)
        with row2_col2:
            loan_percent_income = st.number_input("Loan to Income Ratio (0-1)", 0.0, 1.0, 0.20, 0.01)
            cb_person_default_on_file = st.selectbox("Historical Default", ["Y", "N"])
        with row2_col3:
            cb_person_cred_hist_length = st.number_input("Credit History (Years)", 0, None, 3, 1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Calculate Button
    if st.button("🚀 Analyze Risk Profile"):
        # Prepare Data with correct column order
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
            # Apply Encoders
            if encoders:
                for col, encoder in encoders.items():
                    if col in input_data.columns:
                        input_data[col] = encoder.transform(input_data[col])
            
            # Apply Scaler
            if scaler:
                input_data_scaled = scaler.transform(input_data)
            else:
                input_data_scaled = input_data

            # Prediction
            prediction = model.predict(input_data_scaled)[0]
            probability = model.predict_proba(input_data_scaled)[0][1] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Results Display
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            with res_col2:
                if prediction == 1:
                    status_color = "#ff4b4b"
                    status_icon = "⚠️"
                    status_text = "High Risk Identified"
                    desc = "The model predicts a high likelihood of default."
                else:
                    status_color = "#00c853"
                    status_icon = "✅"
                    status_text = "Low Risk Profile"
                    desc = "The applicant demonstrates a safe credit profile."

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 50px; margin-bottom: 10px;">{status_icon}</div>
                    <h2 style="color: {status_color} !important; margin: 0;">{status_text}</h2>
                    <p style="color: #666; margin-top: 10px;">{desc}</p>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee;">
                        <span style="font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px;">Default Probability</span>
                        <div style="font-size: 36px; font-weight: 700; color: #333;">{probability:.2f}%</div>
                        <div style="width: 100%; background: #eee; height: 8px; border-radius: 4px; margin-top: 10px; overflow: hidden;">
                            <div style="width: {probability}%; background: {status_color}; height: 100%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# --- Batch Processing Section ---
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📂 Batch Processing (CSV Upload)"):
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file and model:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if st.button("Process Batch Data"):
                # Copy for transformation
                processed_df = batch_df.copy()
                
                # Apply Encoders
                if encoders:
                    for col, encoder in encoders.items():
                        if col in processed_df.columns:
                            processed_df[col] = encoder.transform(processed_df[col])
                
                # Apply Scaler
                if scaler:
                    processed_data = scaler.transform(processed_df[FEATURES]) # Ensure column order
                else:
                    processed_data = processed_df
                
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]
                
                batch_df['Risk_Class'] = ["High Risk" if p == 1 else "Low Risk" for p in predictions]
                batch_df['Default_Probability'] = probabilities
                
                st.dataframe(batch_df.style.applymap(
                    lambda v: 'color: red; font-weight: bold' if v == 'High Risk' else 'color: green; font-weight: bold', 
                    subset=['Risk_Class']
                ))
                
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "risk_assessment_results.csv",
                    "text/csv",
                    key='download-csv'
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: rgba(255,255,255,0.5); font-size: 12px;">
    Powered by Advanced Machine Learning • Corporate Credit Risk Division
</div>
""", unsafe_allow_html=True)