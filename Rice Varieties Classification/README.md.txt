# Rice Variety Classification

## Project Overview
This project implements and compares two machine learning algorithms (Random Forest and Decision Tree) to classify rice varieties (Cammeo and Osmancik) based on their morphological features.

## Dataset
The dataset contains 3810 rice grain samples with the following features:
- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent
- Class (target variable: Cammeo or Osmancik)

## Data Preprocessing
- Verified no missing values in the dataset
- Encoded the categorical target variable using LabelEncoder
- Split data into training (80%) and testing (20%) sets

## Model Implementation and Comparison

### Random Forest Classifier
- Implemented using scikit-learn's RandomForestClassifier
- Used default hyperparameters with a fixed random state for reproducibility

**Results:**
- Accuracy: 92.52%
- Precision: 92% (Cammeo), 93% (Osmancik)
- Recall: 91% (Cammeo), 93% (Osmancik)
- F1-score: 92% (Cammeo), 93% (Osmancik)

### Decision Tree Classifier
- Implemented using scikit-learn's DecisionTreeClassifier
- Used default hyperparameters with a fixed random state for reproducibility

**Results:**
- Accuracy: 87.80%
- Precision: 88% (Cammeo), 88% (Osmancik)
- Recall: 85% (Cammeo), 90% (Osmancik)
- F1-score: 87% (Cammeo), 89% (Osmancik)

## Key Findings
- Random Forest outperformed the Decision Tree classifier by approximately 5% in overall accuracy
- Random Forest showed more balanced precision and recall metrics for both rice varieties
- The confusion matrices revealed that Random Forest had fewer misclassifications overall

## Visualization
- Generated confusion matrices for both models to visualize classification performance
- Used the inverse transform to convert numeric predictions back to variety names

## Future Improvements
- Hyperparameter tuning for both models
- Feature importance analysis to identify most influential grain characteristics
- Implementation of additional classification algorithms
- Cross-validation to ensure model stability

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn