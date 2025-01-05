
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
file_path = 'Crop_recommendation.csv'
crop_data = pd.read_csv(file_path)

# ====== LOG TRANSFORMATION ====== #
crop_data['K_log'] = np.log1p(crop_data['K'])  # log(K + 1)
crop_data['Rainfall_log'] = np.log(crop_data['rainfall'] + 100)  # log(rainfall + 100)
crop_data.drop(columns=['K'], inplace=True)  # Drop the original K column

# ====== STANDARDIZATION ====== #
scaler = StandardScaler()
all_vars = ['N', 'P', 'K_log', 'Rainfall_log', 'temperature', 'humidity', 'ph']
crop_data_standardized = crop_data.copy()
crop_data_standardized[all_vars] = scaler.fit_transform(crop_data[all_vars])

# ====== RFE ====== #
# Prepare the data for RFE
X_all = crop_data_standardized[all_vars]  # Features for RFE
y_all = OrdinalEncoder().fit_transform(crop_data[['label']]).ravel()  # Ordinal encoded target

# Initialize the model for RFE
rfe_model_all = RandomForestClassifier(random_state=42, n_jobs=-1)
rfe_all = RFE(estimator=rfe_model_all, n_features_to_select=6)  # Select 6 features

# Fit the RFE model
rfe_all.fit(X_all, y_all)

# Feature ranking for all variables
feature_ranking_all = pd.DataFrame({
    'Feature': all_vars,
    'Ranking': rfe_all.ranking_,
    'Selected': rfe_all.support_
}).sort_values(by='Ranking')

# Feature importance for all variables
rfe_model_all.fit(X_all, y_all)
feature_importances_all = pd.DataFrame({
    'Feature': all_vars,
    'Importance': rfe_model_all.feature_importances_
}).sort_values(by='Importance', ascending=False)

# ====== VISUALIZATION ====== #
# Bar plot for feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_all, palette='viridis')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Output feature ranking and importance
print("Feature Ranking (RFE):")
print(feature_ranking_all)

print("\nFeature Importance (Random Forest):")
print(feature_importances_all)
