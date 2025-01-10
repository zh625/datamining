import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'crop_raw_data.csv'  # Adjust the file path as needed
data = pd.read_csv(file_path)

# Function to remove outliers using manual thresholds
def remove_outliers_manual(df, thresholds):
    for column, threshold in thresholds.items():
        if column in df.columns:
            original_count = len(df)
            df = df[df[column] <= threshold]  # Keep only rows below or equal to the threshold
            removed_count = original_count - len(df)
            print(f"Removed {removed_count} outliers in {column}: Above {threshold}.")
    return df

# Define manual thresholds for outlier removal
manual_thresholds = {
    'Zn': 40,  # Remove values above 40
    'S': 100,  # Remove values above 100
    'P': 600   # Remove values above 600
}

# Apply manual outlier removal
data = remove_outliers_manual(data, manual_thresholds)

# Save the original data for "before" comparison
original_data = data.copy()

# Features to be transformed
log_transform_features = ['Ph', 'K', 'N']  # Features for log transformation
yeo_johnson_features = ['P', 'S', 'Zn', 'CLOUD_AMT', 'WD10M', 'WS2M_RANGE', "T2M_MAX-W"]  # Features for Yeo-Johnson transformation

# Apply Log Transformation
for feature in log_transform_features:
    if feature in data.columns:
        if (data[feature] > 0).all():  # Check for positive values
            data[feature] = np.log1p(data[feature])
            print(f"Log transformation applied to: {feature}")
        else:
            print(f"Skipped log transformation for {feature} (contains zero or negative values)")

# Plot "before" distributions for Log Transformation
fig, axes = plt.subplots(1, len(log_transform_features), figsize=(15, 5))
for i, feature in enumerate(log_transform_features):
    axes[i].hist(original_data[feature], bins=30, alpha=0.7, label='Before', color='blue')
    axes[i].set_title(f'Before Log Transformation: {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot "after" distributions for Log Transformation
fig, axes = plt.subplots(1, len(log_transform_features), figsize=(15, 5))
for i, feature in enumerate(log_transform_features):
    axes[i].hist(data[feature], bins=30, alpha=0.7, label='After', color='orange')
    axes[i].set_title(f'After Log Transformation: {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Apply Yeo-Johnson Transformation
transformer = PowerTransformer(method='yeo-johnson')
for feature in yeo_johnson_features:
    if feature in data.columns:
        data[feature] = transformer.fit_transform(data[[feature]])
        print(f"Yeo-Johnson transformation applied to: {feature}")

# Plot "before" distributions for Yeo-Johnson Transformation
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(yeo_johnson_features):
    if feature in original_data.columns:
        axes[i].hist(original_data[feature], bins=30, alpha=0.7, label='Before', color='blue')
        axes[i].set_title(f'Before Yeo-Johnson: {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
# Remove unused subplots
for j in range(len(yeo_johnson_features), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Plot "after" distributions for Yeo-Johnson Transformation
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(yeo_johnson_features):
    if feature in data.columns:
        axes[i].hist(data[feature], bins=30, alpha=0.7, label='After', color='orange')
        axes[i].set_title(f'After Yeo-Johnson: {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
# Remove unused subplots
for j in range(len(yeo_johnson_features), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Frequency Encoding for 'Soilcolor'
if 'Soilcolor' in data.columns:
    soilcolor_counts = data['Soilcolor'].value_counts()
    data['Soilcolor_encoded'] = data['Soilcolor'].map(soilcolor_counts)
    print("Frequency encoding applied to 'Soilcolor'.")

# Drop the original 'Soilcolor' column after encoding
data = data.drop(columns=['Soilcolor'])

# Encode the target variable (label)
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

# Separate features (X) and target (y)
X = data.drop(columns=['label', 'label_encoded'])
y = data['label_encoded']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and target into a single DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['label_encoded'] = y_resampled
resampled_data['label'] = label_encoder.inverse_transform(y_resampled)

# Save the transformed and resampled dataset to a new CSV file
resampled_data.to_csv('transformed_resampled_crop_data_without_outliers.csv', index=False)
print("Transformed and resampled dataset saved as 'transformed_resampled_crop_data_without_outliers.csv'.")
