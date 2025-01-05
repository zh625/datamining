# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:42:55 2025

@author: realm
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the standardized one-hot and ordinal datasets
file_one_hot = 'crop_data_standardized_one_hot.csv'
file_ordinal = 'crop_data_standardized_ordinal.csv'
crop_data_one_hot = pd.read_csv(file_one_hot)
crop_data_ordinal = pd.read_csv(file_ordinal)

# Prepare features (X) and target (y) for both datasets
X_one_hot = crop_data_one_hot.drop(columns=[col for col in crop_data_one_hot.columns if 'label_' in col])
y_one_hot = crop_data_one_hot[[col for col in crop_data_one_hot.columns if 'label_' in col]]

X_ordinal = crop_data_ordinal.drop(columns=['label_ordinal'])
y_ordinal = crop_data_ordinal['label_ordinal']

# Perform 80/20 stratified train-test split for One-Hot Encoded data
X_train_one_hot, X_test_one_hot, y_train_one_hot, y_test_one_hot = train_test_split(
    X_one_hot, y_one_hot, test_size=0.2, stratify=y_one_hot, random_state=42
)

# Perform 80/20 stratified train-test split for Ordinal Encoded data
X_train_ordinal, X_test_ordinal, y_train_ordinal, y_test_ordinal = train_test_split(
    X_ordinal, y_ordinal, test_size=0.2, stratify=y_ordinal, random_state=42
)

# Save the final datasets to CSV for inspection
X_train_one_hot.to_csv('X_train_one_hot.csv', index=False)
X_test_one_hot.to_csv('X_test_one_hot.csv', index=False)
y_train_one_hot.to_csv('y_train_one_hot.csv', index=False)
y_test_one_hot.to_csv('y_test_one_hot.csv', index=False)

X_train_ordinal.to_csv('X_train_ordinal.csv', index=False)
X_test_ordinal.to_csv('X_test_ordinal.csv', index=False)
y_train_ordinal.to_csv('y_train_ordinal.csv', index=False)
y_test_ordinal.to_csv('y_test_ordinal.csv', index=False)

# Confirm successful split
print("Stratified train-test split completed for both datasets.")
print("One-Hot Encoded Dataset:")
print(f"Train shape: {X_train_one_hot.shape}, Test shape: {X_test_one_hot.shape}")
print("Ordinal Encoded Dataset:")
print(f"Train shape: {X_train_ordinal.shape}, Test shape: {X_test_ordinal.shape}")
