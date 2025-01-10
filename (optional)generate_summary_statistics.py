import pandas as pd
import numpy as np

# Load the dataset
file_path = 'crop_raw_data.csv'  # Adjust the file name as needed
data = pd.read_csv(file_path)

# Initialize the summary table
summary_stats = pd.DataFrame()
summary_stats['Variable'] = data.columns  # Include all columns
summary_stats['Role'] = ['Input' if col != 'label' else 'Target' for col in data.columns]  # Assign roles

# Handle numerical statistics
numerical_cols = data.select_dtypes(include=np.number).columns  # Only numerical columns
summary_stats['Mean'] = [data[col].mean() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Standard Deviation'] = [data[col].std() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Minimum'] = [data[col].min() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Median'] = [data[col].median() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Maximum'] = [data[col].max() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Skewness'] = [data[col].skew() if col in numerical_cols else np.nan for col in data.columns]
summary_stats['Kurtosis'] = [data[col].kurt() if col in numerical_cols else np.nan for col in data.columns]

# Add missing/non-missing value counts
summary_stats['Non Missing'] = data.notnull().sum().values
summary_stats['Missing'] = data.isnull().sum().values

# Save the summary statistics to a CSV file
summary_stats.to_csv('additional/summary_statistics.csv', index=False)

print("The summary statistics have been saved as 'summary_statistics.csv' in your working directory.")

