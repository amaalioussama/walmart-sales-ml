"""
Feature engineering module.

This module handles feature extraction and transformation,
particularly for time-based features from the Date column.
"""

import pandas as pd
from typing import List

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def extract_date_features(
    df: pd.DataFrame,
    date_column: str = "Date",
    features: List[str] = None
) -> pd.DataFrame:
    """
    Extract time-based features from a date column.
    
    This function extracts useful time-based features such as year,
    month, week of year, day of week, etc. These features help the
    model capture temporal patterns in the data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a datetime column.
    date_column : str, optional
        Name of the date column to extract features from.
        Default is "Date".
    features : List[str], optional
        List of features to extract. Available options:
        - 'year': Year (e.g., 2010, 2011)
        - 'month': Month (1-12)
        - 'week_of_year': Week number in the year (1-52)
        - 'day_of_week': Day of week (0=Monday, 6=Sunday)
        - 'day_of_month': Day of the month (1-31)
        - 'quarter': Quarter of the year (1-4)
        - 'is_month_start': Boolean, True if first day of month
        - 'is_month_end': Boolean, True if last day of month
        If None, defaults to ['year', 'month', 'week_of_year'].
    
    Returns
    -------
    pd.DataFrame
        Dataframe with additional date features.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Date': pd.date_range('2010-01-01', periods=10, freq='W'),
    ...     'Sales': [100, 200, 150, 300, 250, 280, 320, 290, 310, 340]
    ... })
    >>> df_features = extract_date_features(df, features=['year', 'month'])
    >>> print(df_features.columns)
    Index(['Date', 'Sales', 'year', 'month'], dtype='object')
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Default features
    if features is None:
        features = ['year', 'month', 'week_of_year']
    
    # Ensure date column exists and is datetime
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        logger.info(f"Converting '{date_column}' to datetime")
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract requested features
    logger.info(f"Extracting date features: {features}")
    
    feature_map = {
        'year': lambda dt: dt.year,
        'month': lambda dt: dt.month,
        'week_of_year': lambda dt: dt.isocalendar().week,
        'day_of_week': lambda dt: dt.dayofweek,
        'day_of_month': lambda dt: dt.day,
        'quarter': lambda dt: dt.quarter,
        'is_month_start': lambda dt: dt.is_month_start.astype(int),
        'is_month_end': lambda dt: dt.is_month_end.astype(int)
    }
    
    for feature in features:
        if feature in feature_map:
            df[feature] = df[date_column].apply(feature_map[feature])
            logger.info(f"  Created feature: {feature}")
        else:
            logger.warning(f"  Unknown feature: {feature} (skipped)")
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = "Weekly_Sales",
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Add lagged features (previous weeks' sales).
    
    Lagged features allow the model to use past values to predict
    future values, which is common in time-series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by date.
    target_column : str, optional
        Column to create lags for. Default is "Weekly_Sales".
    lags : List[int], optional
        List of lag periods (in weeks). For example, [1, 2, 4]
        creates lag_1 (previous week), lag_2 (2 weeks ago),
        and lag_4 (4 weeks ago). Default is [1, 2, 4].
    
    Returns
    -------
    pd.DataFrame
        Dataframe with additional lag features.
    
    Notes
    -----
    This function will introduce NaN values for the first few rows
    (depending on the maximum lag). These rows should be dropped
    before training.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Weekly_Sales': [100, 110, 120, 130, 140]
    ... })
    >>> df_lags = add_lag_features(df, lags=[1, 2])
    >>> print(df_lags)
       Weekly_Sales  lag_1  lag_2
    0          100    NaN    NaN
    1          110  100.0    NaN
    2          120  110.0  100.0
    3          130  120.0  110.0
    4          140  130.0  120.0
    """
    # Make a copy
    df = df.copy()
    
    # Default lags
    if lags is None:
        lags = [1, 2, 4]
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataframe")
    
    # Create lag features
    logger.info(f"Creating lag features for '{target_column}': {lags}")
    
    for lag in lags:
        feature_name = f"lag_{lag}"
        df[feature_name] = df[target_column].shift(lag)
        logger.info(f"  Created feature: {feature_name}")
    
    # Report NaN values introduced
    max_lag = max(lags)
    logger.info(f"Note: First {max_lag} rows contain NaN values due to lagging")
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = "Weekly_Sales",
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Add rolling window features (moving averages).
    
    Rolling features capture trends over a period of time,
    which can help the model identify patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by date.
    target_column : str, optional
        Column to create rolling features for. Default is "Weekly_Sales".
    windows : List[int], optional
        List of window sizes (in weeks). For example, [4, 8]
        creates 4-week and 8-week rolling averages.
        Default is [4, 8].
    
    Returns
    -------
    pd.DataFrame
        Dataframe with additional rolling features.
    
    Notes
    -----
    This function will introduce NaN values for the first few rows.
    These rows should be dropped before training.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Weekly_Sales': [100, 110, 120, 130, 140, 150]
    ... })
    >>> df_rolling = add_rolling_features(df, windows=[3])
    >>> print(df_rolling)
       Weekly_Sales  rolling_mean_3
    0          100             NaN
    1          110             NaN
    2          120           110.0
    3          130           120.0
    4          140           130.0
    5          150           140.0
    """
    # Make a copy
    df = df.copy()
    
    # Default windows
    if windows is None:
        windows = [4, 8]
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataframe")
    
    # Create rolling features
    logger.info(f"Creating rolling features for '{target_column}': {windows}")
    
    for window in windows:
        feature_name = f"rolling_mean_{window}"
        df[feature_name] = df[target_column].rolling(window=window).mean()
        logger.info(f"  Created feature: {feature_name}")
    
    return df


def build_features(
    df: pd.DataFrame,
    include_date_features: bool = True,
    include_lag_features: bool = False,
    include_rolling_features: bool = False,
    date_features: List[str] = None,
    lags: List[int] = None,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Build all features for the model.
    
    This is the main feature engineering function that orchestrates
    all feature extraction steps. For the initial version, we focus
    on date features only. Lag and rolling features can be added later.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw data.
    include_date_features : bool, optional
        Whether to extract date features. Default is True.
    include_lag_features : bool, optional
        Whether to add lag features. Default is False.
        Note: Use with caution as it introduces NaN values.
    include_rolling_features : bool, optional
        Whether to add rolling features. Default is False.
        Note: Use with caution as it introduces NaN values.
    date_features : List[str], optional
        List of date features to extract. If None, uses default.
    lags : List[int], optional
        List of lag periods. If None, uses default.
    windows : List[int], optional
        List of rolling windows. If None, uses default.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features.
    
    Examples
    --------
    >>> # Basic usage (date features only)
    >>> df_features = build_features(df)
    
    >>> # With specific date features
    >>> df_features = build_features(
    ...     df,
    ...     date_features=['year', 'month', 'week_of_year', 'quarter']
    ... )
    
    >>> # With lag features (advanced)
    >>> df_features = build_features(
    ...     df,
    ...     include_lag_features=True,
    ...     lags=[1, 2, 4]
    ... )
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Make a copy
    df = df.copy()
    
    # 1. Extract date features
    if include_date_features:
        df = extract_date_features(df, features=date_features)
    
    # 2. Add lag features (optional, for advanced models)
    if include_lag_features:
        df = add_lag_features(df, lags=lags)
    
    # 3. Add rolling features (optional, for advanced models)
    if include_rolling_features:
        df = add_rolling_features(df, windows=windows)
    
    # Drop rows with NaN values if lag/rolling features were added
    if include_lag_features or include_rolling_features:
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        logger.info(f"Dropped {rows_before - rows_after} rows with NaN values")
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df
