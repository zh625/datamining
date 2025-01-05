
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'file_path' with the path to your dataset
file_path = 'crop_raw_data'
crop_data = pd.read_csv(file_path)

# Log transformation for Potassium (K) and Rainfall
crop_data['K_log'] = np.log1p(crop_data['K'])  # log(K + 1)
crop_data['Rainfall_log'] = np.log(crop_data['rainfall'] + 100)  # log(rainfall + 100)

# Plotting the transformed distributions
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Original vs Transformed for Potassium (K)
sns.histplot(crop_data['K'], kde=True, ax=axs[0, 0], color='blue', bins=20)
axs[0, 0].set_title('Original Distribution of Potassium (K)')
axs[0, 0].set_xlabel('K (Potassium)')

sns.histplot(crop_data['K_log'], kde=True, ax=axs[0, 1], color='green', bins=20)
axs[0, 1].set_title('Log-Transformed Distribution of Potassium (K)')
axs[0, 1].set_xlabel('log(K + 1)')

# Original vs Transformed for Rainfall
sns.histplot(crop_data['rainfall'], kde=True, ax=axs[1, 0], color='blue', bins=20)
axs[1, 0].set_title('Original Distribution of Rainfall')
axs[1, 0].set_xlabel('Rainfall (mm)')

sns.histplot(crop_data['Rainfall_log'], kde=True, ax=axs[1, 1], color='green', bins=20)
axs[1, 1].set_title('Log-Transformed Distribution of Rainfall')
axs[1, 1].set_xlabel('log(Rainfall + 100)')

# Adjust layout
plt.tight_layout()
plt.show()
