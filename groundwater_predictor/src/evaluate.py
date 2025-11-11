# src/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """
    Visualize feature importance from Random Forest model.
    """
    importance = model.feature_importances_
    sorted_idx = importance.argsort()

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Groundwater Predictor – Feature Importance")
    plt.tight_layout()
    plt.show()
