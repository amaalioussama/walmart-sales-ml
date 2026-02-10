"""
Utility to create sample datasets for testing.

This module helps create small sample datasets that can be
used for testing without requiring the full dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.paths import ensure_dir

# Initialize logger
logger = get_logger(__name__)


def create_sample_from_full(
    input_file: Path,
    output_file: Path,
    n_rows: int = 100,
    store_id: Optional[int] = None
) -> None:
    """
    Create a sample CSV from the full dataset.
    
    Parameters
    ----------
    input_file : Path
        Path to the full dataset CSV.
    output_file : Path
        Path where the sample CSV will be saved.
    n_rows : int, optional
        Number of rows to include in the sample. Default is 100.
    store_id : int, optional
        If provided, sample only from this store. Default is None.
    
    Examples
    --------
    >>> create_sample_from_full(
    ...     Path("data/raw/walmart_store_sales.csv"),
    ...     Path("data/sample/sample_100.csv"),
    ...     n_rows=100,
    ...     store_id=1
    ... )
    """
    logger.info(f"Creating sample dataset from {input_file}")
    
    # Read full dataset
    df = pd.read_csv(input_file)
    logger.info(f"Loaded full dataset: {len(df):,} rows")
    
    # Filter by store if requested
    if store_id is not None:
        df = df[df["Store"] == store_id]
        logger.info(f"Filtered for Store {store_id}: {len(df):,} rows")
    
    # Sample rows
    if len(df) > n_rows:
        df_sample = df.head(n_rows)
        logger.info(f"Sampled {n_rows} rows")
    else:
        df_sample = df
        logger.info(f"Dataset has only {len(df)} rows, using all")
    
    # Save sample
    ensure_dir(output_file.parent)
    df_sample.to_csv(output_file, index=False)
    logger.info(f"Sample saved to {output_file}")


def create_synthetic_sample(
    output_file: Path,
    n_rows: int = 50,
    store_id: int = 1
) -> None:
    """
    Create a synthetic sample dataset for testing.
    
    This is useful when the real dataset is not available yet.
    
    Parameters
    ----------
    output_file : Path
        Path where the sample CSV will be saved.
    n_rows : int, optional
        Number of rows to generate. Default is 50.
    store_id : int, optional
        Store ID to use. Default is 1.
    
    Examples
    --------
    >>> create_synthetic_sample(
    ...     Path("data/sample/synthetic_sample.csv"),
    ...     n_rows=50
    ... )
    """
    import numpy as np
    
    logger.info(f"Creating synthetic sample dataset with {n_rows} rows")
    
    # Generate dates (weekly from 2010-02-05)
    dates = pd.date_range(start="2010-02-05", periods=n_rows, freq="W")
    
    # Generate synthetic data
    np.random.seed(42)
    
    df = pd.DataFrame({
        "Store": [store_id] * n_rows,
        "Date": dates,
        "Weekly_Sales": np.random.uniform(10000, 50000, n_rows),
        "Holiday_Flag": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
        "Temperature": np.random.uniform(30, 90, n_rows),
        "Fuel_Price": np.random.uniform(2.5, 4.5, n_rows),
        "CPI": np.random.uniform(125, 225, n_rows),
        "Unemployment": np.random.uniform(5, 10, n_rows)
    })
    
    # Save sample
    ensure_dir(output_file.parent)
    df.to_csv(output_file, index=False)
    logger.info(f"Synthetic sample saved to {output_file}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Shape: {df.shape}")


if __name__ == "__main__":
    # Example: Create a synthetic sample
    output = Path("data/sample/synthetic_sample.csv")
    create_synthetic_sample(output, n_rows=50)
    print(f"\nSynthetic sample created at: {output}")
