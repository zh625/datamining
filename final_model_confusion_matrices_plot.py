import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the RFE-selected dataset
file_path = 'rfe_selected_features_dataset.csv'  # Adjust path as needed
data = pd.read_csv(file_path)

# Remove the QV2M-Au feature
if 'QV2M-Au' in data.columns:
    data = data.drop(columns=['QV2M-Au'])

# Separate features (X) and target (y)
X = data.drop(columns=['label_encoded'])  # Adjust target columns as needed
y = data['label_encoded']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for models that require it (SVM, KNN, Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Refined hyperparameter grids (provided earlier)
param_grids_refined = {
    "Random Forest": {
        "n_estimators": [140, 150, 160],
        "max_depth": [None, 15, 20],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 3],
    },
    "KNN": {
        "n_neighbors": [3, 4, 5],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVM": {
        "C": [5, 10, 15],
        "kernel": ["linear", "rbf"],
        "gamma": ["auto", 0.1, 0.5],
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50, 50), (100, 50), (100, 100)],
        "activation": ["relu"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["adaptive"],
        "learning_rate_init": [0.005, 0.01, 0.02],
        "batch_size": [32, 64, 128],
        "max_iter": [1000],
        "early_stopping": [True],
    },
    "XGBoost": {
        "n_estimators": [140, 150, 160],
        "learning_rate": [0.08, 0.1, 0.12],
        "max_depth": [8, 10, 12],
        "subsample": [0.8, 0.85, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
}

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

# Perform GridSearchCV with refined hyperparameters
results_refined = []
for model_name, model in models.items():
    print(f"Running Refined GridSearch for {model_name}...")
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grids_refined[model_name],
        scoring="accuracy",
        cv=10,
        n_jobs=-1,
    )
    
    # Select scaled or unscaled data
    X_train_data = X_train_scaled if model_name in ["KNN", "SVM", "Neural Network"] else X_train
    X_test_data = X_test_scaled if model_name in ["KNN", "SVM", "Neural Network"] else X_test
    
    # Fit the model
    search.fit(X_train_data, y_train)
    
    # Get the best parameters and evaluate on the test set
    best_params = search.best_params_
    y_pred = search.best_estimator_.predict(X_test_data)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"model_plot/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Store the results as a dictionary
    results_refined.append({
        "Model": model_name,
        "Search Type": "Refined GridSearchCV",
        "Best Parameters": best_params,
        "Test Accuracy": test_accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"]
    })

    print(f"Test Accuracy for {model_name}: {test_accuracy:.2f}")

# Convert refined results to a DataFrame and save to CSV
results_refined_df = pd.DataFrame(results_refined)
results_refined_csv_path = 'refined_model_results_with_confusion_matrices.csv'
results_refined_df.to_csv(results_refined_csv_path, index=False)

# Visualization of Metrics
metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_refined_df)
    plt.title(f"Comparison of {metric} Across Models")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"model_plot/{metric}_comparison.png")
    plt.close()

print("Refined results and visualizations have been saved.")

# Feature Importance Visualization for Tree-Based Models
tree_based_models = ["Random Forest", "XGBoost"]

for model_name in tree_based_models:
    if model_name in results_refined_df["Model"].values:
        # Retrieve the best estimator for the model
        best_estimator = models[model_name].set_params(**results_refined_df.loc[
            results_refined_df["Model"] == model_name, "Best Parameters"
        ].iloc[0])
        
        # Fit the model on the full training set for visualization
        X_train_data = X_train if model_name == "Random Forest" else X_train_scaled
        best_estimator.fit(X_train_data, y_train)

        # Get feature importances
        importances = best_estimator.feature_importances_
        feature_names = X.columns

        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title(f"Feature Importance for {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(f"model_plot/{model_name}_feature_importance.png")
        plt.close()

        print(f"Feature importance plot for {model_name} saved as '{model_name}_feature_importance.png'.")
