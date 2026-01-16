
import pandas as pd
import numpy as np

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transformer function for feature engineering.
    Calculates debt_to_income and monthly_payment.
    """
    X_out = X.copy()
    
    # debt_to_income = loan_amount / (income + 1e-6)
    X_out['debt_to_income'] = X_out['loan_amount'] / (X_out['income'] + 1e-6)
    
    # monthly_payment = loan_amount / loan_duration_months
    X_out['monthly_payment'] = X_out['loan_amount'] / (X_out['loan_duration_months'] + 1e-6)
    
    return X_out
