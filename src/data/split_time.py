"""
Time-based data splitting module.

This module handles splitting time-series data into training and test sets
based on date ranges, which is crucial for proper time-series evaluation.
"""

import pandas as pd
from typing import Tuple, Union
from datetime import datetime

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def split_by_date(
    df: pd.DataFrame,
    date_column: str = "Date",
    train_start: Union[str, datetime] = None,
    train_end: Union[str, datetime] = None,
    test_start: Union[str, datetime] = None,
    test_end: Union[str, datetime] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets based on date ranges.
    
    For time-series data, we CANNOT use random splitting (like train_test_split)
    because it would leak future information into the training set.
    Instead, we split chronologically: train on earlier data, test on later data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a date column.
    date_column : str, optional
        Name of the date column. Default is "Date".
    train_start : str or datetime, optional
        Start date for training period (inclusive).
        Format: 'YYYY-MM-DD'. If None, uses earliest date in data.
    train_end : str or datetime, optional
        End date for training period (inclusive).
        Format: 'YYYY-MM-DD'.
    test_start : str or datetime, optional
        Start date for test period (inclusive).
        Format: 'YYYY-MM-DD'.
    test_end : str or datetime, optional
        End date for test period (inclusive).
        Format: 'YYYY-MM-DD'. If None, uses latest date in data.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) - Training and test dataframes.
    
    Raises
    ------
    ValueError
        If date ranges are invalid or if there's overlap between
        train and test periods.
    
    Examples
    --------
    >>> # Train on 2010-2011, test on 2012
    >>> train_df, test_df = split_by_date(
    ...     df,
    ...     train_start="2010-01-01",
    ...     train_end="2011-12-31",
    ...     test_start="2012-01-01",
    ...     test_end="2012-12-31"
    ... )
    """
    # Make a copy
    df = df.copy()
    
    # Validate date column
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Convert string dates to datetime
    def to_datetime(date_val):
        if date_val is None:
            return None
        if isinstance(date_val, str):
            return pd.to_datetime(date_val)
        return date_val
    
    train_start = to_datetime(train_start)
    train_end = to_datetime(train_end)
    test_start = to_datetime(test_start)
    test_end = to_datetime(test_end)
    
    # Use data range if not specified
    if train_start is None:
        train_start = df[date_column].min()
    if test_end is None:
        test_end = df[date_column].max()
    
    # Validate date ranges
    if train_end is None or test_start is None:
        raise ValueError(
            "Must specify either (train_end and test_start) or both complete ranges"
        )
    
    if train_start >= train_end:
        raise ValueError(f"train_start ({train_start}) must be before train_end ({train_end})")
    
    if test_start >= test_end:
        raise ValueError(f"test_start ({test_start}) must be before test_end ({test_end})")
    
    if train_end >= test_start:
        raise ValueError(
            f"Training period ({train_end}) must end before test period starts ({test_start})\n"
            "This ensures no data leakage in time-series forecasting."
        )
    
    # Split data
    logger.info(f"Splitting data by date:")
    logger.info(f"  Train period: {train_start.date()} to {train_end.date()}")
    logger.info(f"  Test period:  {test_start.date()} to {test_end.date()}")
    
    train_df = df[
        (df[date_column] >= train_start) & 
        (df[date_column] <= train_end)
    ].copy()
    
    test_df = df[
        (df[date_column] >= test_start) & 
        (df[date_column] <= test_end)
    ].copy()
    
    # Log results
    logger.info(f"  Train set: {len(train_df):,} rows")
    logger.info(f"  Test set:  {len(test_df):,} rows")
    logger.info(f"  Split ratio: {len(train_df)/(len(train_df)+len(test_df)):.1%} train, "
                f"{len(test_df)/(len(train_df)+len(test_df)):.1%} test")
    
    # Validate splits
    if len(train_df) == 0:
        raise ValueError("Training set is empty. Check your date ranges.")
    if len(test_df) == 0:
        raise ValueError("Test set is empty. Check your date ranges.")
    
    return train_df, test_df


def prepare_X_y(
    df: pd.DataFrame,
    target_column: str = "Weekly_Sales",
    drop_columns: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix (X) and target vector (y) for modeling.
    
    This function separates the target variable from the features
    and removes columns that shouldn't be used for prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    target_column : str, optional
        Name of the target column. Default is "Weekly_Sales".
    drop_columns : list, optional
        Additional columns to drop (e.g., 'Date', 'Store').
        Default drops ['Date', 'Store'].
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where:
        - X is the feature matrix
        - y is the target vector
    
    Examples
    --------
    >>> train_df, test_df = split_by_date(df, ...)
    >>> X_train, y_train = prepare_X_y(train_df)
    >>> X_test, y_test = prepare_X_y(test_df)
    """
    # Default columns to drop
    if drop_columns is None:
        drop_columns = ['Date', 'Store']
    
    # Make a copy
    df = df.copy()
    
    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Extract target
    y = df[target_column].copy()
    
    # Drop target and unwanted columns
    columns_to_drop = [target_column] + [col for col in drop_columns if col in df.columns]
    X = df.drop(columns=columns_to_drop)
    
    logger.info(f"Prepared X and y:")
    logger.info(f"  X shape: {X.shape} ({X.shape[1]} features)")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  Features: {list(X.columns)}")
    
    return X, y


def get_train_test_split(
    df: pd.DataFrame,
    target_column: str = "Weekly_Sales",
    train_start: Union[str, datetime] = None,
    train_end: Union[str, datetime] = None,
    test_start: Union[str, datetime] = None,
    test_end: Union[str, datetime] = None,
    drop_columns: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete pipeline: split by date and prepare X, y for both sets.
    
    This is a convenience function that combines split_by_date and
    prepare_X_y for a streamlined workflow.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    target_column : str, optional
        Name of the target column. Default is "Weekly_Sales".
    train_start : str or datetime, optional
        Start date for training period.
    train_end : str or datetime, optional
        End date for training period.
    test_start : str or datetime, optional
        Start date for test period.
    test_end : str or datetime, optional
        End date for test period.
    drop_columns : list, optional
        Columns to drop before modeling.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = get_train_test_split(
    ...     df,
    ...     train_start="2010-01-01",
    ...     train_end="2011-12-31",
    ...     test_start="2012-01-01",
    ...     test_end="2012-12-31"
    ... )
    """
    logger.info("="*60)
    logger.info("TRAIN-TEST SPLIT (Time-based)")
    logger.info("="*60)
    
    # Split by date
    train_df, test_df = split_by_date(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )
    
    # Prepare X and y
    X_train, y_train = prepare_X_y(train_df, target_column, drop_columns)
    X_test, y_test = prepare_X_y(test_df, target_column, drop_columns)
    
    logger.info("="*60)
    
    return X_train, X_test, y_train, y_test
