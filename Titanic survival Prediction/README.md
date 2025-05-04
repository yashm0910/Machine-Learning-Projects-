# Titanic Survival Prediction

## Project Overview
This project implements a logistic regression model to predict passenger survival on the Titanic. It serves as my introduction to machine learning classification techniques.

## Dataset
The model uses the famous Titanic dataset containing passenger information such as:
- Passenger class (Pclass)
- Gender (Sex)
- Age
- Fare
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Port of embarkation (Embarked)

## Data Preprocessing
The following preprocessing steps were performed:
- Handled missing age values by filling with median age
- Filled missing embarkation values with the most common port
- Encoded categorical variables (Sex and Embarked) using LabelEncoder

## Model Implementation
- Used scikit-learn's LogisticRegression classifier
- Split data with 80% training and 20% testing
- Set max_iter=200 to ensure convergence

## Results
The model achieved:
- Accuracy: 81.0%
- Precision for survival prediction: 79%
- Recall for survival prediction: 74%

## Visualization
- Generated confusion matrix to visualize prediction results
- Created feature importance chart to understand which factors most influenced survival predictions

## Future Improvements
I plan to enhance this project by:
- Implementing more advanced algorithms like Random Forest
- Performing additional feature engineering
- Exploring hyperparameter tuning
- Adding cross-validation techniques

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn