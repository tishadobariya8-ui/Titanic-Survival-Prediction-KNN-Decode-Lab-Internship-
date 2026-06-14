# Titanic-Survival-Prediction-KNN-Decode-Lab-Internship-
Project Overview

This project predicts whether a passenger survived the Titanic disaster using the K-Nearest Neighbors (KNN) classification algorithm.
The project demonstrates the complete Machine Learning workflow, including data preprocessing, feature engineering, model training, hyperparameter tuning, model evaluation, and cross-validation.

# Dataset
Titanic Dataset

Features used:
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked

Target Variable:
0 = Did Not Survive
1 = Survived

# Project Workflow

1. Data Preprocessing
Removed unnecessary columns:
PassengerId
Name
Ticket
Cabin
Handled missing values in:
Age
Fare
Embarked

2. Feature Engineering
Converted categorical features into numerical format.
Applied one-hot encoding to the Embarked column.

3. Data Splitting
The dataset was divided into:
Training Set
Validation Set
Testing Set
Stratified sampling was used to preserve class distribution.

4. Feature Scaling
StandardScaler was applied to normalize feature values before training the KNN model.

5. Hyperparameter Tuning
The optimal K value was selected by evaluating K values from 1 to 20 using the validation dataset.

6. Model Training
A K-Nearest Neighbors (KNN) classifier was trained using the optimal K value.

7. Model Evaluation
Performance was evaluated using:
Accuracy Score
Confusion Matrix
F1 Score
Classification Report
5-Fold Cross Validation

# Results
Optimal K Value: 3
Training Accuracy: 0.997
Testing Accuracy: 0.9643
Average 5-Fold Cross Validation Accuracy: 0.9833
F1 Score: 0.9508

The high cross-validation score confirms that the model generalizes well across different data splits.
