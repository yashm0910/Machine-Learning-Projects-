# House Price Prediction

A simple machine learning project to predict house prices using linear regression.

## Overview

This project analyzes housing data to build a predictive model for house prices. Using scikit-learn's SGDRegressor, I trained a linear regression model on a dataset containing various house features. The model focuses on Overall Quality, Living Area, and Year Built as the key predictors of sale price.

## Data Exploration

The dataset contains information about 1460 houses with 81 different attributes including:
- Physical attributes (square footage, rooms)
- Quality ratings
- Building features
- Sale information

After correlation analysis, I selected these predictors:
- `OverallQual`: Overall quality rating (1-10)
- `GrLivArea`: Above ground living area (square feet)
- `YearBuilt`: Original construction year
- `GarageCars`: Size of garage in car capacity

## Data Analysis

Key statistics:
- Houses built between 1872 and 2010
- Mean sale price: $180,921
- Strong correlation between Overall Quality and Sale Price (0.79)
- Good correlation between Living Area and Sale Price (0.71)

## Model Training

The model was trained with:
- Learning rate: 0.0000001
- Epochs: 50
- Batch size: 50

## Results

Final model equation:
```
SalePrice = -30.386*YearBuilt + 69.265*GrLivArea + 282.419*OverallQual - 3.371
```

Interpretation:
- Overall Quality has the strongest positive effect on price
- Living Area has a moderate positive effect
- Year Built has a small negative effect

## Functions Explained

### `train_sklearn_model()`
Trains the SGDRegressor model for a specified number of epochs and tracks the RMSE. Returns weights, bias, and training history.

### `run_experiment()`
Sets up and runs a training experiment with specified features and label. Handles training, reporting, and visualization.

### `plot_data()`, `plot_model()`, `plot_loss_curve()`
Visualization functions that create different plots:
- Data points scatter plot
- Model prediction line/plane
- RMSE loss curve during training

### `make_plots()`
Creates a combined visualization with model fit and loss curve subplots.

### `model_info()`
Formats and prints information about the trained model parameters and equation.


## Requirements

- Python 3
- pandas, numpy, matplotlib
- scikit-learn
- plotly