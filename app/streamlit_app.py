
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path to allow src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_engineered_features
from src.evaluation import get_individual_force_plot

# --- CONFIGURATION & ASSETS ---

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    """Load and cache pre-trained ML models and preprocessors."""
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/final_model.pkl')
        return preprocessor, model
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the training notebooks first.")
        st.stop()

# --- VALIDATION LOGIC ---

def validate_inputs(data):
    """
    Performs domain-specific validation on the input fields.
    Returns (is_valid, error_message).
    """
    validations = [
        (data['income'] > 0, "Monthly income must be greater than 0."),
        (data['loan_amount'] > 0, "Loan amount must be greater than 0."),
        (300 <= data['credit_score'] <= 850, "Credit score must be between 300 and 850."),
        (18 <= data['age'] <= 120, "Applicant age must be between 18 and 120."),
        (data['loan_duration_months'] >= 1, "Loan duration must be at least 1 month."),
        (data['loan_amount'] / (data['income'] + 1e-6) <= 1000, "The requested loan amount is unrealistic relative to income.")
    ]
    
    for condition, msg in validations:
        if not condition:
            return False, msg
    return True, ""

# --- UI COMPONENTS ---

def render_sidebar_form():
    """Renders the applicant data entry form in the sidebar."""
    with st.sidebar:
        st.header("Applicant Information")
        with st.form("applicant_data"):
            data = {
                'income': st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, help="Monthly gross income."),
                'loan_amount': st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0, help="Total amount requested."),
                'loan_duration_months': st.number_input("Duration (Months)", min_value=1, value=36),
                'age': st.number_input("Age", min_value=0, max_value=150, value=34),
                'employment_type': st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed", "Other"]),
                'credit_score': st.number_input("Credit Score", min_value=0, max_value=1000, value=710),
                'previous_defaults': st.number_input("Previous Defaults", min_value=0, value=0)
            }
            submitted = st.form_submit_button("Analyze Risk Profile")
    return submitted, data

def render_assessment_summary(prob, risk_label, credit_score, defaults, income, loan_amount):
    """Renders the left column results (Metrics and Indicators)."""
    st.subheader("Assessment Summary")
    color = "red" if risk_label == "HIGH RISK" else "green"
    st.markdown(f"### Result: :{color}[{risk_label}]")
    st.metric("Probability of Default", f"{prob*100:.1f}%")
    
    st.write("---")
    st.write("**Quick Indicators:**")
    if credit_score < 600:
        st.warning("⚠️ Low Credit Score")
    if defaults > 0:
        st.warning("⚠️ History of Defaults")
    if (loan_amount/income) > 5:
        st.info("ℹ️ High Loan-to-Income ratio")

def render_explainability_section(model, processed_data, feature_names):
    """Renders the SHAP force plot and interpretability info."""
    st.subheader("Individual Risk Breakdown")
    st.info("Visualizing how each feature pushes the risk prediction away from the model's average base value.")
    
    with st.spinner("Generating SHAP explanation..."):
        force_plot_html = get_individual_force_plot(model, processed_data, feature_names)
        components.html(force_plot_html, height=200, scrolling=True)

def render_global_insights():
    """Renders global feature importance charts."""
    st.divider()
    st.subheader("Global Model Insights")
    if os.path.exists('models/top_features.pkl'):
        top_features = joblib.load('models/top_features.pkl')
        st.write("Top overall predictors across the training population:")
        st.bar_chart(pd.Series(top_features))
    else:
        st.info("Global feature importance artifact not found.")

# --- MAIN EXECUTION ---

def run_prediction_pipeline(raw_data):
    """Handles the end-to-end data transformation and prediction."""
    preprocessor, model = load_artifacts()
    
    # 1. Transform input
    input_df = pd.DataFrame([raw_data])
    input_df_eng = add_engineered_features(input_df)
    
    # 2. Preprocess
    processed_data = preprocessor.transform(input_df_eng)
    feature_names = preprocessor.get_feature_names_out()
    
    # 3. Inference
    prob = model.predict_proba(processed_data)[0][1]
    risk_label = "HIGH RISK" if prob > 0.5 else "LOW RISK"
    
    return prob, risk_label, processed_data, feature_names, model

def main():
    st.title("🛡️ Credit Risk Prediction System")
    st.markdown("Automated classification and explainable AI for loan applicant risk assessment.")
    
    submitted, raw_data = render_sidebar_form()

    if submitted:
        is_valid, error_msg = validate_inputs(raw_data)
        
        if not is_valid:
            st.error(f"❌ Input Validation Error: {error_msg}")
            return

        try:
            prob, label, proc_data, feat_names, model = run_prediction_pipeline(raw_data)
            
            # Display Results
            col1, col2 = st.columns([1, 2])
            with col1:
                render_assessment_summary(
                    prob, label, 
                    raw_data['credit_score'], 
                    raw_data['previous_defaults'], 
                    raw_data['income'], 
                    raw_data['loan_amount']
                )
            with col2:
                render_explainability_section(model, proc_data, feat_names)
                
            render_global_insights()
                    
        except Exception as e:
            st.error(f"Prediction Pipeline Failed: {e}")
            st.exception(e)
    else:
        st.info("👈 Fill in the applicant details on the sidebar and click 'Analyze' to start.")

if __name__ == "__main__":
    main()
