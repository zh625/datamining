
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.metrics import roc_curve, auc

# Load the results CSV file
results_path = 'model_results_with_class_report.csv'
model_results = pd.read_csv(results_path)

# --- Bar Plot for Comparative Metrics ---
def plot_metrics(metrics):
    # Extract metrics for plotting
    metrics = metrics[["Unnamed: 0", "Accuracy", "F1-Score (Macro)", "F1-Score (Weighted)", "ROC-AUC"]]
    metrics = metrics.rename(columns={"Unnamed: 0": "Model"})
    metrics.set_index("Model", inplace=True)

    # Plot comparative metrics
    metrics.plot(kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black")
    plt.title("Model Performance Metrics")
    plt.ylabel("Scores")
    plt.xlabel("Models")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Confusion Matrix Heatmap ---
def plot_confusion_matrix(conf_matrix_str, model_name):
    try:
        # Convert string to list using ast.literal_eval
        conf_matrix = ast.literal_eval(conf_matrix_str)
    except ValueError as e:
        print(f"Error parsing confusion matrix for {model_name}: {e}")
        return

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --- ROC Curve Plot ---
def plot_roc_curve(model_results, model_name, y_test_binarized, y_pred_proba):
    n_classes = y_test_binarized.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Example usage:
plot_metrics(model_results)
plot_confusion_matrix(model_results.loc[0, "Confusion Matrix"], "Random Forest")

# Note: For ROC curve, provide y_test_binarized and y_pred_proba when calling the plot_roc_curve function.
