import pytest
import pandas as pd
import numpy as np
import joblib
import os

def test_model_prediction():
    """
    Tests if the model can produce valid probabilities.
    """
    model_path = 'models/final_model.pkl'
    # Fallback for notebook runs or different root contexts
    if not os.path.exists(model_path):
        model_path = '../models/final_model.pkl'
        
    if not os.path.exists(model_path):
        pytest.skip("Final model artifact not found.")
        
    model = joblib.load(model_path)
    
    # Generate 5 random samples (assuming 9 input features based on dummy preprocessor output)
    # The dummy preprocessor produces 9 features (8 numeric + 1 OHE category result)
    # We'll just check if it accepts the transformed input shape
    X_dummy = np.random.rand(5, 9)
    
    probs = model.predict_proba(X_dummy)
    
    # Check shape (n_samples, n_classes)
    assert probs.shape == (5, 2)
    # Check probability range
    assert np.all((probs >= 0) & (probs <= 1))
