
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, roc_curve
)

def evaluate_model(model, X_test, y_test):
    """
    Calculates key classification metrics.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_probs),
        'report': classification_report(y_test, y_preds, output_dict=True)
    }
    
    return metrics

def plot_confusion_matrix(y_test, y_preds):
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def generate_shap_plots(model, X_processed, feature_names, output_path='reports/shap_summary.png'):
    """
    Computes SHAP values and saves a summary plot.
    """
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use TreeExplainer for XGBoost/RF, otherwise KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
    except:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return shap_values
