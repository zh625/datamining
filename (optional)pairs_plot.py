# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:39:37 2025

@author: realm
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (ensure it includes the selected features and target variable)
file_path = 'transformed_resampled_crop_data_without_outliers.csv'
data = pd.read_csv(file_path)

# List of selected features by Boruta and RFE
selected_features = ['Ph', 'K', 'P', 'N', 'Zn', 'S', 
                     'QV2M-W', 'QV2M-Au', 'PRECTOTCORR-W', 
                     'WD10M', 'Soilcolor_encoded']

# Subset the dataset to include only the selected features and the target variable
subset_data = data[selected_features + ['label_encoded']]

# Draw the pair plot
sns.set(style="ticks", font_scale=1.1)
pairplot = sns.pairplot(
    subset_data,
    hue='label_encoded',  # Color by the target variable
    diag_kind='kde',  # Kernel density estimation on diagonal
    plot_kws={'alpha': 0.6, 's': 20}  # Transparency and point size for scatter plots
)

# Adjust layout and display the plot
pairplot.fig.suptitle("Pairwise Relationships of Selected Features", y=1.02, fontsize=16)
plt.savefig('plot/pairs_plot_graph.png')
plt.show()
