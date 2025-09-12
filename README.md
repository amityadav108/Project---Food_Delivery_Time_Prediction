# Project---Food_Delivery_Time_Prediction
## Objective
Predict whether a food delivery will be Fast or Delayed based on customer location, restaurant location, weather, traffic conditions, and other factors. This is a binary classification problem where the model outputs:
Fast → Delivered within the expected time
Delayed → Delivered later than expected

## Features
Data Preprocessing: Missing value handling, categorical encoding, feature scaling.
Feature Engineering: Binary classification target creation, distance calculations.
Modeling:
Gaussian Naive Bayes
K-Nearest Neighbors (KNN) with hyperparameter tuning
Decision Tree with pruning to avoid overfitting
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC curves.
Model Comparison: Identifies the best-performing classifier.

## Requirements
Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn

### Workflow
1. Data Preprocessing
Load Data: Read the CSV file.
Handle Missing Values: Imputed numeric columns with mean; categorical with 'Unknown'.
Encode Categorical Features: Used LabelEncoder.
Feature Scaling: Applied StandardScaler.

2. Feature Engineering
Target Creation: If Delivery_Time > 30 minutes → Delayed, else Fast.
Optional: Calculate geographic distance using Haversine formula if latitude/longitude available.

3. Model Training and Evaluation
Naive Bayes: Simple baseline for continuous features.
K-Nearest Neighbors: Tuned k using GridSearchCV.
Decision Tree: Tuned max_depth and min_samples_split.

4. Reporting and Insights
Compared accuracy, precision, recall, and F1-score.
Visualized confusion matrices using Seaborn.
Recommended the best classifier based on performance and interpretability.

## Results Summary
Decision Tree achieved the highest accuracy among the three classifiers.
KNN performed well but was sensitive to the choice of k.
Naive Bayes was fastest to train but less accurate.
Recommended Model: Decision Tree (easy to interpret and high accuracy).
