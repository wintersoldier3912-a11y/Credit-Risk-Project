
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path to allow src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_engineered_features

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦")

@st.cache_resource
def load_artifacts():
    # In a real scenario, paths would be models/preprocessor.pkl etc.
    # Using relative paths for streamlit run from root
    preprocessor = joblib.load('models/preprocessor.pkl')
    model = joblib.load('models/final_model.pkl')
    return preprocessor, model

def main():
    st.title("🛡️ Credit Risk Prediction System")
    st.markdown("Enter applicant details to determine creditworthiness.")
    
    # Input Form
    with st.form("applicant_data"):
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=20000.0)
            loan_duration = st.number_input("Duration (Months)", min_value=1, value=36)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            
        with col2:
            employment_type = st.selectbox("Employment Type", 
                                           ["Salaried", "Self-employed", "Unemployed", "Other"])
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            defaults = st.number_input("Previous Defaults", min_value=0, value=0)
            
        submit = st.form_submit_button("Predict Risk")

    if submit:
        # 1. Load artifacts
        try:
            preprocessor, model = load_artifacts()
            
            # 2. Prepare Data
            input_df = pd.DataFrame([{
                'income': income,
                'employment_type': employment_type,
                'loan_amount': loan_amount,
                'loan_duration_months': loan_duration,
                'credit_score': credit_score,
                'age': age,
                'previous_defaults': defaults
            }])
            
            # Feature engineering (needs to happen before ColumnTransformer if manual)
            input_df = add_engineered_features(input_df)
            
            # 3. Transform
            processed_data = preprocessor.transform(input_df)
            
            # 4. Predict
            prob = model.predict_proba(processed_data)[0][1]
            risk_label = "HIGH RISK" if prob > 0.5 else "LOW RISK"
            
            # 5. Display Results
            st.divider()
            st.subheader("Prediction Result")
            
            color = "red" if risk_label == "HIGH RISK" else "green"
            st.markdown(f"### Assessment: :{color}[{risk_label}]")
            st.metric("Probability of Default", f"{prob*100:.1f}%")
            
            # Explanation Mock (In real case use SHAP)
            st.subheader("Key Risk Drivers")
            drivers = [
                ("Credit Score", "Lower than median" if credit_score < 650 else "Stable"),
                ("Debt to Income", "High" if (loan_amount/income) > 5 else "Normal"),
                ("History", "Has defaults" if defaults > 0 else "Clean")
            ]
            for feat, status in drivers:
                st.write(f"**{feat}:** {status}")
                
        except Exception as e:
            st.error(f"Error loading model or processing data. Ensure artifacts are trained and present. {e}")

if __name__ == "__main__":
    main()
