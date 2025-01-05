Crop Recommendation System
This project implements a machine learning pipeline to build a crop recommendation system using various machine learning models. The system predicts the most suitable crop for cultivation based on environmental and soil conditions.

Project Structure
The project is divided into multiple steps, each with its corresponding script for clarity and modularity:

1. crop_raw_data.csv:

This file contains the raw dataset with environmental and soil conditions, which serves as the input to the pipeline.

2. data_preprocessing.py:

Purpose: Initial preprocessing of the raw dataset.
Key Steps:
Missing value handling (if any).
Data cleaning and basic feature engineering.
Saving the cleaned dataset for further processing.

3. data_splitting.py:

Purpose: Splits the cleaned data into training and testing sets.
Key Steps:
Performs an 80/20 stratified train-test split.
Saves the resulting training and testing datasets for both one-hot and ordinal encodings.

4. log_transformation_and_visualization.py:

Purpose: Applies log transformations to specific variables and visualizes their distributions.
Key Steps:
Log transformation of highly skewed variables like Potassium (K) and Rainfall.
Boxplot and histogram visualizations to verify transformations.
Identifies and addresses data normalization issues.

5. standardization_and_visualization.py:

Purpose: Standardizes the continuous features and visualizes the results.
Key Steps:
Standardization of key features (e.g., N, P, K_log, Rainfall_log).
Visualization of standardized variables to confirm uniform scaling.
Prepares data for machine learning models.

6. model.py:

Purpose: Builds, tunes, and evaluates machine learning models.
Key Steps:
Models included: Random Forest, Gradient Boosting, SVM, KNN, Neural Network.
Hyperparameter tuning using GridSearchCV.
Evaluates models using performance metrics (accuracy, F1-score, ROC-AUC).
Results are saved in model_results_with_class_report.csv.

7. model_train_test.py (Optional):

Purpose: Evaluates the models separately for training and testing datasets.
Key Steps:
Splits performance metrics into training and testing.
Saves the results in a detailed CSV file for further analysis.

8. visualize_results.py:

Purpose: Visualizes the results from the trained models.
Key Steps:
Generates heatmaps for confusion matrices.
Plots model comparison metrics (e.g., accuracy, F1-scores, ROC-AUC).
Visualizes the ROC curve for multi-class problems.
