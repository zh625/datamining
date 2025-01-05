import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np

# Load the dataset
file_path = 'crop_raw_data.csv'
crop_data_raw = pd.read_csv(file_path)

# Log transformation for selected variables
crop_data_raw['K_log'] = (crop_data_raw['K'] + 1).apply(lambda x: np.log(x))
crop_data_raw['Rainfall_log'] = (crop_data_raw['rainfall'] + 100).apply(lambda x: np.log(x))
crop_data_raw = crop_data_raw.drop(columns=['K', 'rainfall'])  # Drop original columns if transformed

# Standardization of continuous variables
scaler = StandardScaler()
scaled_features = ['N', 'P', 'K_log', 'Rainfall_log', 'temperature', 'humidity', 'ph']
crop_data_raw[scaled_features] = scaler.fit_transform(crop_data_raw[scaled_features])

# One-Hot Encoding for the target variable
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(crop_data_raw[['label']])
one_hot_columns = [f'label_{category}' for category in one_hot_encoder.categories_[0]]
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)

# Combine with the original dataset (excluding the original target column)
crop_data_one_hot = pd.concat([crop_data_raw.drop(columns=['label']), one_hot_df], axis=1)

# Ordinal Encoding for the target variable
ordinal_encoder = OrdinalEncoder()
crop_data_raw['label_ordinal'] = ordinal_encoder.fit_transform(crop_data_raw[['label']]).astype(int)

# Prepare final datasets
crop_data_ordinal = crop_data_raw.drop(columns=['label'])

# Save the preprocessed datasets
crop_data_one_hot.to_csv('crop_data_standardized_one_hot.csv', index=False)
crop_data_ordinal.to_csv('crop_data_standardized_ordinal.csv', index=False)

# Confirm preprocessing
print("Preprocessing completed. Datasets saved as:")
print("- crop_data_standardized_one_hot.csv")
print("- crop_data_standardized_ordinal.csv")
