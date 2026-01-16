
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw CSV data.
    """
    return pd.read_csv(path)

def train_val_test_split(df: pd.DataFrame, target_col: str, seed: int = 42):
    """
    Splits data into train (70%), validation (15%), and test (15%) sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: Train vs Remainder
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    # Second split: Val vs Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=seed, stratify=y_rem
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
