
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, model_name='xgboost', params=None):
    """
    Initializes and trains a model based on name.
    """
    if params is None:
        params = {}
        
    if model_name == 'xgboost':
        model = XGBClassifier(random_state=42, **params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, **params)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(class_weight='balanced', random_state=42, **params)
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    model.fit(X_train, y_train)
    return model

def save_model(obj, path: str):
    """
    Saves a serialized object to path.
    """
    joblib.dump(obj, path)

def load_model(path: str):
    """
    Loads a serialized object from path.
    """
    return joblib.load(path)
