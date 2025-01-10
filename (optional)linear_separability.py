import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the RFE-selected dataset
file_path = 'rfe_selected_features_dataset.csv'  # Adjust path as needed
data = pd.read_csv(file_path)

# Features to scale
features_to_scale = ['Ph','K', 'P', 'N','Zn','S']

# Apply StandardScaler
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Plot scaled features to visualize linear separability
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features_to_scale):
    plt.subplot(1, len(features_to_scale), i + 1)  # Create subplots
    sns.boxplot(x='label_encoded', y=feature, data=data_scaled)  # Boxplot for scaled features
    plt.title(f'{feature} vs Label (Scaled)')  # Title
    plt.xlabel('Label')  # X-axis label
    plt.ylabel(feature)  # Y-axis label

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
