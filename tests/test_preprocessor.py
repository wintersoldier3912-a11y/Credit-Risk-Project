
import pytest
import pandas as pd
import numpy as np
import joblib
import os

def test_preprocessor_output():
    """
    Tests if preprocessor returns correct shape and no NaNs.
    """
    model_path = 'models/preprocessor.pkl'
    if not os.path.exists(model_path):
        pytest.skip("Preprocessor artifact not found. Run training first.")
        
    preprocessor = joblib.load(model_path)
    
    # Create dummy row
    sample = pd.DataFrame([{
        'income': 5000.0,
        'employment_type': 'Salaried',
        'loan_amount': 10000.0,
        'loan_duration_months': 24,
        'credit_score': 700,
        'age': 35,
        'previous_defaults': 0,
        'debt_to_income': 2.0,
        'monthly_payment': 400.0
    }])
    
    out = preprocessor.transform(sample)
    assert out.shape[0] == 1
    assert not np.isnan(out).any()
