import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the processed dataset
file_path = 'transformed_resampled_crop_data_without_outliers.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['label', 'label_encoded'])  # Drop target columns
y = data['label_encoded']  # Use the encoded target variable

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=150)

# Function to perform RFE and determine the optimal number of features
def optimal_features_rfe(X, y, estimator, cv_splits=10):
    accuracy_scores = []
    feature_counts = []

    for n_features in range(1, X.shape[1] + 1):
        rfe = RFE(estimator, n_features_to_select=n_features)
        X_rfe = rfe.fit_transform(X, y)

        # Cross-validation with accuracy scoring
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(estimator, X_rfe, y, cv=cv, scoring='accuracy')
        mean_accuracy = np.mean(scores)

        accuracy_scores.append(mean_accuracy)
        feature_counts.append(n_features)

        print(f"Features: {n_features}, Accuracy: {mean_accuracy:.4f}")

    # Identify the optimal number of features
    optimal_features = feature_counts[np.argmax(accuracy_scores)]
    print(f"\nOptimal number of features: {optimal_features}")

    return optimal_features, accuracy_scores, feature_counts

# Run RFE with Random Forest Classifier
optimal_features, accuracy_scores, feature_counts = optimal_features_rfe(X, y, rf_classifier)

# Plot the Accuracy vs Number of Features
plt.figure(figsize=(10, 6))
plt.plot(feature_counts, accuracy_scores, marker='o', linestyle='-', color='b')
plt.title('RFE: Accuracy vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig('plot/rfe_accuracy_vs_feature.png')
plt.show()

# Select the optimal feature subset using RFE
rfe = RFE(rf_classifier, n_features_to_select=optimal_features)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]

print(f"Selected Features by RFE: {list(selected_features)}")

# Save the dataset with selected features
X_rfe_df = pd.DataFrame(X_rfe, columns=selected_features)
X_rfe_df['label_encoded'] = y
X_rfe_df.to_csv('rfe_selected_features_dataset.csv', index=False)
print("Dataset with RFE-selected features saved as 'rfe_selected_features_dataset.csv'.")
