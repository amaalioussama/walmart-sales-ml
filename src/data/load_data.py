"""
Data loading module.

This module handles loading the raw Walmart sales data and 
filtering it for analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.utils.logger import get_logger
from src.utils.paths import check_file_exists, get_file_size_mb

# Initialize logger
logger = get_logger(__name__)


def load_raw_data(
    file_path: Union[str, Path],
    store_id: Optional[int] = None,
    store_list: Optional[list] = None
) -> pd.DataFrame:
    """
    Load the raw Walmart sales data from CSV.
    
    This function loads the Walmart dataset and optionally filters
    it for specific store(s). It also performs basic validation.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the raw CSV file.
    store_id : int, optional
        If provided, filters data for this specific store.
        If None, returns data for all stores.
    store_list : list, optional
        If provided, filters data for these specific stores.
        Overrides store_id if both are provided.
    
    Returns
    -------
    pd.DataFrame
        Loaded and optionally filtered dataframe.
    
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist.
    ValueError
        If the required columns are missing.
    
    Examples
    --------
    >>> # Load data for Store 1
    >>> df = load_raw_data("data/raw/walmart_store_sales.csv", store_id=1)
    >>> print(df.shape)
    (143, 8)
    """
    # Check if file exists
    if not check_file_exists(file_path):
        raise FileNotFoundError(
            f"Data file not found at: {file_path}\n"
            f"Please ensure the Walmart dataset is placed in the correct location."
        )
    
    # Log file info
    file_size = get_file_size_mb(file_path)
    logger.info(f"Loading data from {file_path} (Size: {file_size} MB)")
    
    # Load CSV
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # Validate required columns
    required_columns = [
        "Store", "Date", "Weekly_Sales", "Holiday_Flag",
        "Temperature", "Fuel_Price", "CPI", "Unemployment"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Filter for specific store(s) if requested
    if store_list is not None:
        df = df[df["Store"].isin(store_list)].copy()
        logger.info(f"Filtered data for Stores {store_list}: {len(df):,} rows")
        
        if len(df) == 0:
            raise ValueError(f"No data found for Stores {store_list}")
    elif store_id is not None:
        df = df[df["Store"] == store_id].copy()
        logger.info(f"Filtered data for Store {store_id}: {len(df):,} rows")
        
        if len(df) == 0:
            raise ValueError(f"No data found for Store {store_id}")
    else:
        logger.info(f"Loaded data for all stores: {df['Store'].nunique()} stores")
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info("Converted 'Date' column to datetime format")
    
    # Sort by date for time-series consistency
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info("Sorted data by date")
    
    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataframe.
    
    This function displays useful information about the dataset
    including shape, date range, missing values, and basic statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarize.
    
    Examples
    --------
    >>> df = load_raw_data("data/raw/walmart_store_sales.csv", store_id=1)
    >>> get_data_summary(df)
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Date range
    if "Date" in df.columns:
        print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found ✓")
    
    # Target variable statistics
    if "Weekly_Sales" in df.columns:
        print(f"\nWeekly Sales statistics:")
        print(f"  Mean:   ${df['Weekly_Sales'].mean():,.2f}")
        print(f"  Median: ${df['Weekly_Sales'].median():,.2f}")
        print(f"  Min:    ${df['Weekly_Sales'].min():,.2f}")
        print(f"  Max:    ${df['Weekly_Sales'].max():,.2f}")
    
    # Columns
    print(f"\nColumns: {list(df.columns)}")
    
    print("="*60 + "\n")


def check_data_quality(df: pd.DataFrame) -> bool:
    """
    Perform basic data quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    
    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    
    Examples
    --------
    >>> df = load_raw_data("data/raw/walmart_store_sales.csv", store_id=1)
    >>> is_valid = check_data_quality(df)
    """
    logger.info("Running data quality checks...")
    
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"Found {missing} missing values")
    
    # Check for negative sales
    if "Weekly_Sales" in df.columns:
        negative_sales = (df["Weekly_Sales"] < 0).sum()
        if negative_sales > 0:
            issues.append(f"Found {negative_sales} negative sales values")
    
    # Check for duplicate dates (for single store)
    if "Date" in df.columns and "Store" in df.columns:
        if df["Store"].nunique() == 1:
            duplicates = df["Date"].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate dates")
    
    # Report results
    if issues:
        logger.warning("Data quality issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("All data quality checks passed ✓")
        return True
