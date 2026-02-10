"""
Evaluation metrics for regression models.

This module provides functions to compute common regression metrics.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics: MAE, RMSE, and R².
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mae': Mean Absolute Error
        - 'rmse': Root Mean Squared Error
        - 'r2': R² Score (coefficient of determination)
    
    Examples
    --------
    >>> y_true = np.array([100, 200, 300])
    >>> y_pred = np.array([110, 190, 310])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"MAE: {metrics['mae']:.2f}")
    MAE: 10.00
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print metrics in a readable format.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing metrics (mae, rmse, r2).
    model_name : str, optional
        Name of the model for display purposes.
    
    Examples
    --------
    >>> metrics = {"mae": 1234.56, "rmse": 2345.67, "r2": 0.8543}
    >>> print_metrics(metrics, "Linear Regression")
    """
    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"MAE (Mean Absolute Error):  ${metrics['mae']:,.2f}")
    print(f"RMSE (Root Mean Squared):   ${metrics['rmse']:,.2f}")
    print(f"R² Score:                   {metrics['r2']:.4f}")
    print(f"{'='*50}\n")
