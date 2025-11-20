"""
Supervised Learning Final Project: White Wine Quality Prediction
Contributors: [Your Names]

This script executes the end-to-end training pipeline:
1. Loads the White Wine dataset.
2. Splits data into Train/Test sets (80/20).
3. Establishes a Baseline (DummyRegressor).
4. Optimizes a Ridge Regression model using GridSearchCV.
5. Evaluates the best model and reports business improvement.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def run_pipeline():
    # 1. Load Data
    # We use sep=';' as identified during EDA
    print("Loading dataset...")
    df = pd.read_csv('winequality-white.csv', sep=';')
    
    X = df.drop('quality', axis=1)
    y = df['quality']

    # 2. Train/Test Split (Rigorous Validation)
    # We hold out 20% of data for the final "Business Value" test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: {X_train.shape[0]} Training samples, {X_test.shape[0]} Test samples")

    # 3. Establish Baseline (The "Floor")
    # We use a DummyRegressor to predict the mean quality for everyone.
    # Any useful model MUST beat this error.
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_mse = mean_squared_error(y_test, y_pred_dummy)

    print(f"\n--- Baseline Performance ---")
    print(f"Baseline MSE (Guessing Average): {baseline_mse:.4f}")

    # 4. Build Pipeline (Preprocessing + Model)
    # We use StandardScaler to handle different feature ranges (e.g. Sugar vs Density)
    # We use Ridge to handle the Multicollinearity detected during EDA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    # 5. Optimization (Grid Search)
    # We test various regularization strengths (alpha) to find the best balance
    param_grid = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    
    print(f"\n--- Starting Grid Search Optimization ---")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Optimization Complete. Best Alpha: {grid_search.best_params_['ridge__alpha']}")

    # 6. Final Evaluation
    # We test the champion model on the held-out Test set
    y_pred = best_model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    
    improvement = 100 * (baseline_mse - final_mse) / baseline_mse

    print(f"\n--- Final Ridge Model Results ---")
    print(f"Final MSE:      {final_mse:.4f}")
    print(f"R2 Score:       {final_r2:.4f}")
    print(f"---------------------------------------")
    print(f"BUSINESS VALUE: Improved prediction accuracy by {improvement:.2f}% over baseline")

if __name__ == "__main__":
    run_pipeline()