# src/explain_model.py
import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample):
    """
    Generates SHAP summary plot for model explainability.
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
