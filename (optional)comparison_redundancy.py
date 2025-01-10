import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score

# Load Boruta and RFE-selected datasets
boruta_file_path = 'boruta_selected_features_dataset.csv'
rfe_file_path = 'rfe_selected_features_dataset.csv'

boruta_data = pd.read_csv(boruta_file_path)
rfe_data = pd.read_csv(rfe_file_path)

# Define correlated feature pairs to test
correlated_pairs = [('QV2M-Au', 'QV2M-W'), ('PRECTOTCORR-W', 'WD10M')]

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

# Define hyperparameter grids for all models
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "KNN": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],  # Simplified for faster computation
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001],
    },
    "XGBoost": {
        "n_estimators": [100, 150],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}

# Function to evaluate models after redundancy testing
def evaluate_models(X_train, X_test, y_train, y_test, feature_set_name, removed_feature):
    results = []
    for model_name, model in models.items():
        print(f"Evaluating {model_name} for {feature_set_name} (Removed: {removed_feature})...")
        
        # Use Standard Scaling for models that require it
        if model_name in ["KNN", "SVM", "Neural Network"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train and evaluate model
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Store the results
        results.append({
            "Feature Set": feature_set_name,
            "Removed Feature": removed_feature,
            "Model": model_name,
            "Test Accuracy (%)": round(test_accuracy * 100, 2),
        })
    return results

# Redundancy testing function
def redundancy_test(data, feature_set_name):
    results = []
    X = data.drop(columns=['label_encoded'])
    y = data['label_encoded']

    # Baseline accuracy with all features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    baseline_results = evaluate_models(X_train, X_test, y_train, y_test, feature_set_name, "None")
    results.extend(baseline_results)

    # Iterate through correlated pairs and remove one feature at a time
    for feature1, feature2 in correlated_pairs:
        for feature_to_remove in [feature1, feature2]:
            if feature_to_remove in X.columns:
                X_reduced = X.drop(columns=[feature_to_remove])
                X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
                reduced_results = evaluate_models(X_train, X_test, y_train, y_test, feature_set_name, feature_to_remove)
                results.extend(reduced_results)
    return results

# Perform redundancy testing on Boruta features
boruta_results = redundancy_test(boruta_data, "Boruta Features")

# Perform redundancy testing on RFE features
rfe_results = redundancy_test(rfe_data, "RFE Features")

# Combine and save results
all_results = pd.DataFrame(boruta_results + rfe_results)
all_results.to_csv("additional/redundancy_test_results.csv", index=False)

# Display results
print("\nRedundancy Testing Results:")
print(all_results)
