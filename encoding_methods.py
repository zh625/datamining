
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load your dataset
# Replace 'file_path' with the path to your dataset
file_path = 'Crop_recommendation.csv'
crop_data = pd.read_csv(file_path)

# ====== ONE-HOT ENCODING ====== #
# Apply One-Hot Encoding to the 'label' column
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded_labels = one_hot_encoder.fit_transform(crop_data[['label']])

# Create a DataFrame for One-Hot Encoded labels
one_hot_encoded_columns = one_hot_encoder.get_feature_names_out(['label'])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_labels, columns=one_hot_encoded_columns, index=crop_data.index)

# Combine the original dataset with One-Hot Encoded labels
crop_data_one_hot = pd.concat([crop_data.drop(columns=['label']), one_hot_encoded_df], axis=1)

# ====== ORDINAL ENCODING ====== #
# Apply Ordinal Encoding to the 'label' column
ordinal_encoder = OrdinalEncoder()
ordinal_encoded_labels = ordinal_encoder.fit_transform(crop_data[['label']])

# Add Ordinal Encoded labels back to the dataset
crop_data_ordinal = crop_data.copy()
crop_data_ordinal['label_ordinal'] = ordinal_encoded_labels

# ====== SAVE THE DATASETS ====== #
# Save One-Hot Encoded and Ordinal Encoded datasets to CSV for inspection
crop_data_one_hot.to_csv('crop_data_one_hot.csv', index=False)
crop_data_ordinal.to_csv('crop_data_ordinal.csv', index=False)

# Output for confirmation
print("One-Hot Encoded Dataset Sample:")
print(crop_data_one_hot.head())

print("\nOrdinal Encoded Dataset Sample:")
print(crop_data_ordinal.head())
