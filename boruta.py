import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# Load your dataset
file_path = 'transformed_resampled_crop_data_without_outliers.csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=['label', 'label_encoded'])  # Features
y = data['label_encoded']  # Target variable

# Define the Random Forest Classifier for Boruta
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize Boruta
boruta_selector = BorutaPy(
    estimator=rf_model,
    n_estimators='auto',
    random_state=42,
    verbose=2
)

# Fit Boruta to your dataset
boruta_selector.fit(X.values, y.values)

# Create a DataFrame to summarize Boruta results
feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.fit(X, y).feature_importances_,
    'Ranking': boruta_selector.ranking_,
    'Selected': boruta_selector.support_
})

# Sort by feature importance for better visualization
feature_ranking = feature_ranking.sort_values(by='Importance', ascending=False)

# Print selected features
selected_features = feature_ranking.loc[feature_ranking['Selected'], 'Feature'].tolist()
print("\nSelected Features by Boruta:")
print(selected_features)

# Filter the selected features and create a copy to avoid SettingWithCopyWarning
boruta_selected_features = X[selected_features].copy()  # Create a copy of the selected features
boruta_selected_features['label_encoded'] = y.values  # Add the target variable explicitly

# Save the dataset with Boruta-selected features
boruta_selected_features.to_csv('boruta_selected_features_dataset.csv', index=False)
print("Dataset with Boruta-selected features saved as 'boruta_selected_features_dataset.csv'.")

# Visualize Boruta results
plt.figure(figsize=(12, 8))
plt.barh(
    feature_ranking['Feature'],
    feature_ranking['Importance'],
    color=feature_ranking['Selected'].apply(lambda x: 'green' if x else 'red')
)
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Importance and Boruta Selection', fontsize=16)
plt.gca().invert_yaxis()  # Ensure the highest importance is at the top
plt.tight_layout()

# Save and display the plot
plt.savefig('boruta_feature_importance_plot.png')
plt.show()
