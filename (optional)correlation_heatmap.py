import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'transformed_resampled_crop_data_without_outliers.csv'  # Adjust the file path as needed
data = pd.read_csv(file_path)

# List of selected features (from Boruta and RFE)
selected_features = ['Ph', 'K', 'P', 'N', 'Zn', 'S', 
                     'QV2M-W', 'QV2M-Au', 'PRECTOTCORR-W', 
                     'WD10M', 'Soilcolor_encoded']

# Subset the dataset to include only the selected features
subset_data = data[selected_features]

# Compute the correlation matrix
correlation_matrix = subset_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Annotate the correlation coefficients
    fmt=".2f",  # Format the coefficients to 2 decimal places
    cmap='coolwarm',  # Color map
    cbar=True,  # Show color bar
    square=True  # Make heatmap squares
)
plt.title("Correlation Heatmap of Selected Features", fontsize=16)
plt.tight_layout()
plt.savefig('plot/correlation_heatmap_graph.png')
# Show the plot
plt.show()
