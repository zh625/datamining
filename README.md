Crop Recommendation System
This project implements a machine learning pipeline to build a crop recommendation system using various machine learning models. The system predicts the most suitable crop for cultivation based on environmental and soil conditions.

Project Structure
The project is divided into multiple steps, each with its corresponding script for clarity and modularity:

1. Dataset
File: crop_raw_data.csv
Description: Contains the raw dataset with environmental and soil conditions, which serves as the input to the pipeline.

2. Preprocessing
Script: preprocessing.py
Purpose: Initial preprocessing of the raw dataset.
Key Steps:
1)Remove outliers.
2)Apply log and Yeo-Johnson transformations.
3)Perform frequency encoding.
4)Use SMOTE for handling class imbalances.
Output: transformed_resampled_crop_data_without_outliers.csv

3. Feature Selection
(a) Boruta
Script: boruta.py
Purpose: Selects relevant features using the Boruta algorithm.
Output: boruta_selected_features_dataset.csv
Plot: plot/boruta_feature_importance_plot.png

(b) Recursive Feature Elimination (RFE)
Script: rfe.py
Purpose: Selects relevant features using RFE.
Output:rfe_selected_features_dataset.csv
Plot: plot/rfe_accuracy_vs_features.png

4. Model Training and Evaluation
(a) Train and Test Metrics
Script: final_model_train_test.py
Purpose: Builds models and evaluates their performance on train and test datasets.
Output:refined_model_results_with_train_test.csv

(b) Confusion Matrices and Metrics Comparison
Script: final_model_confusion_matrices_plot.py
Purpose: Visualizes performance metrics and feature importance.
Outputs:refined_model_results_with_confusion_matrices.csv
Plots: In folder model_plot

5. Optional Scripts
(a) Generate Summary Statistics
Script: (optional)generate_summary_statistics.py
Purpose: Generates descriptive statistics for the dataset.
Output: additional/summary_statistics.csv

(b) Correlation Heatmap
Script: (optional)correlation_heatmap.py
Purpose: Creates a heatmap showing correlations between features.
Output: plot/correlation_heatmap_graph.png

(c) Feature Redundancy Comparison
Script: (optional)comparison_redundancy.py
Purpose: Compares performance with reduced selected features.
Output: additional/redundancy_test_results.csv

(d) Linear Separability
Script: (optional)linear_separability.py
Purpose: Visualizes the linear separability of features.
Output: plot/linear_separability.png

(e) Pairs Plot
Script: (optional)pairs_plot.py
Purpose: Generates pairwise plots for feature interactions.
Output: plot/pairs_plot_graph.png

Folders and Outputs Summary
Main Outputs
Processed Data Files:
1)transformed_resampled_crop_data_without_outliers.csv
2)boruta_selected_features_dataset.csv
3)rfe_selected_features_dataset.csv

Model Results:
1)refined_model_results_with_train_test.csv
2)refined_model_results_with_confusion_matrices.csv

Plots and Visualizations:
Folder: plot
1)Boruta Feature Importance: boruta_feature_importance_plot.png
2)RFE Accuracy vs Features: rfe_accuracy_vs_features.png
3)Correlation Heatmap: correlation_heatmap_graph.png
4)Linear Separability: linear_separability.png
5)Pairs Plot: pairs_plot_graph.png

Folder: model_plot
Includes confusion matrices and performance metric comparisons.
Additional Outputs:
Folder: additional
1)Summary Statistics: summary_statistics.csv
2)Redundancy Test Results: redundancy_test_results.csv

Usage Instructions
1)Start with preprocessing.py to preprocess the raw dataset.
2)Run feature selection scripts (boruta.py and/or rfe.py) to identify important features.
3)Use final_model_train_test.py to train models and evaluate their performance.
4)Visualize performance using final_model_confusion_matrices_plot.py.
5)Optionally, run the additional scripts for further analysis and insights.

Contact

For any issues or further information, please contact the project maintainer.
