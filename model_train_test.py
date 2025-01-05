# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:54:53 2025

@author: realm
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Load pre-split datasets
X_train_one_hot = pd.read_csv('X_train_one_hot.csv')
X_test_one_hot = pd.read_csv('X_test_one_hot.csv')
y_train_one_hot = pd.read_csv('y_train_one_hot.csv')
y_test_one_hot = pd.read_csv('y_test_one_hot.csv')

X_train_ordinal = pd.read_csv('X_train_ordinal.csv')
X_test_ordinal = pd.read_csv('X_test_ordinal.csv')
y_train_ordinal = pd.read_csv('y_train_ordinal.csv').squeeze()  # Convert to Series
y_test_ordinal = pd.read_csv('y_test_ordinal.csv').squeeze()    # Convert to Series

# Encode target variables for ordinal datasets
def encode_target(y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder

# Apply encoding to ordinal target variables
y_train_ordinal_encoded, y_test_ordinal_encoded, label_encoder = encode_target(y_train_ordinal, y_test_ordinal)

# Replace y_train_ordinal and y_test_ordinal in the function
def get_datasets(encoding_type):
    if encoding_type == "One-Hot":
        return X_train_one_hot, X_test_one_hot, y_train_one_hot, y_test_one_hot
    elif encoding_type == "Ordinal":
        return X_train_ordinal, X_test_ordinal, y_train_ordinal_encoded, y_test_ordinal_encoded
    else:
        raise ValueError("Invalid encoding type specified.")

# Define models and their encoding type
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42, probability=True),
    "KNN": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=300)
}

model_encoding = {
    "Random Forest": "Ordinal",         # Tree-based model
    "Gradient Boosting": "Ordinal",     # Tree-based model
    "SVM": "One-Hot",                   # Distance-based model
    "KNN": "One-Hot",                   # Distance-based model
    "Neural Network": "One-Hot"         # Continuous and normalized input
}

# Hyperparameter grids for tuning
param_grids = {
    "Random Forest": {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]},
    "Gradient Boosting": {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    "KNN": {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']},
    "Neural Network": {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
}

# Function to train models and evaluate performance for both training and testing sets
def train_and_evaluate(models, results_dict):
    for name, model in models.items():
        encoding_type = model_encoding[name]
        print(f"Training {name} ({encoding_type} Encoding)...")
        
        # Select appropriate dataset
        X_train, X_test, y_train, y_test = get_datasets(encoding_type)
        
        # Convert one-hot encoded labels to single-label format if necessary
        if encoding_type == "One-Hot" and len(y_train.shape) > 1:
            y_train = np.argmax(y_train.values, axis=1)  # Convert one-hot to class indices
            y_test = np.argmax(y_test.values, axis=1)    # Convert one-hot to class indices
        
        # Grid Search for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Training Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
        train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
        
        # Testing Metrics
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        
        # ROC-AUC (for models supporting probability outputs)
        roc_auc = None
        if hasattr(best_model, "predict_proba"):
            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
            y_pred_proba = best_model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, average="macro", multi_class="ovr")
        
        # Store results
        results_dict[name] = {
            "Best Params": grid_search.best_params_,
            "Train Accuracy": train_acc,
            "Train F1-Score (Macro)": train_f1_macro,
            "Train F1-Score (Weighted)": train_f1_weighted,
            "Test Accuracy": test_acc,
            "Test F1-Score (Macro)": test_f1_macro,
            "Test F1-Score (Weighted)": test_f1_weighted,
            "ROC-AUC": roc_auc,
        }
        
        # Print results
        print(f"{name} ({encoding_type} Encoding) Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
        print(f"Train F1-Score (Macro): {train_f1_macro:.2f}, Test F1-Score (Macro): {test_f1_macro:.2f}")
        print(f"Train F1-Score (Weighted): {train_f1_weighted:.2f}, Test F1-Score (Weighted): {test_f1_weighted:.2f}")
        print(f"ROC-AUC: {roc_auc:.2f}" if roc_auc else "ROC-AUC: Not Applicable")
        print("\n")

# Initialize results dictionary
results = {}

# Train and evaluate models
train_and_evaluate(models, results)

print("Model development completed.")
# Save separate train and test results 
results_df_separate = pd.DataFrame({model: metrics for model, metrics in results.items()}).T
results_df_separate.to_csv("model_results_with_train_test.csv")
print("Train and test metrics saved to 'model_results_with_train_test.csv'.")




