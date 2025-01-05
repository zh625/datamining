
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load your dataset
# Replace 'file_path' with the path to your dataset
file_path = 'Crop_recommendation.csv'
crop_data = pd.read_csv(file_path)

# Log transformation for Potassium (K) and Rainfall
crop_data['K_log'] = np.log1p(crop_data['K'])  # log(K + 1)
crop_data['Rainfall_log'] = np.log(crop_data['rainfall'] + 100)  # log(rainfall + 100)

# Selecting interval variables for standardization
interval_vars = ['N', 'P', 'K_log', 'Rainfall_log', 'temperature', 'humidity', 'ph']

# Standardizing the variables using z-score normalization
scaler = StandardScaler()
crop_data_standardized = crop_data.copy()  # Create a copy to preserve original data
crop_data_standardized[interval_vars] = scaler.fit_transform(crop_data[interval_vars])

# Visualizing standardized variables (e.g., histogram of one or two variables)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Standardized distribution for Nitrogen (N)
sns.histplot(crop_data_standardized['N'], kde=True, ax=axs[0, 0], color='blue', bins=20)
axs[0, 0].set_title('Standardized Distribution of Nitrogen (N)')
axs[0, 0].set_xlabel('Standardized N')

# Standardized distribution for Phosphorus (P)
sns.histplot(crop_data_standardized['P'], kde=True, ax=axs[0, 1], color='green', bins=20)
axs[0, 1].set_title('Standardized Distribution of Phosphorus (P)')
axs[0, 1].set_xlabel('Standardized P')

# Standardized distribution for log-transformed Potassium (K_log)
sns.histplot(crop_data_standardized['K_log'], kde=True, ax=axs[1, 0], color='orange', bins=20)
axs[1, 0].set_title('Standardized Distribution of Potassium (K_log)')
axs[1, 0].set_xlabel('Standardized K_log')

# Standardized distribution for log-transformed Rainfall (Rainfall_log)
sns.histplot(crop_data_standardized['Rainfall_log'], kde=True, ax=axs[1, 1], color='red', bins=20)
axs[1, 1].set_title('Standardized Distribution of Rainfall (Rainfall_log)')
axs[1, 1].set_xlabel('Standardized Rainfall_log')

# Adjust layout
plt.tight_layout()
plt.show()
