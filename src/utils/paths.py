"""
Path utility functions.

This module provides helper functions for managing file paths
and ensuring directories exist.
"""

from pathlib import Path
from typing import Union


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory : str or Path
        The directory path to check/create.
    
    Returns
    -------
    Path
        The Path object for the directory.
    
    Examples
    --------
    >>> ensure_dir("data/processed")
    Path('data/processed')
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get the size of a file in megabytes.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file.
    
    Returns
    -------
    float
        File size in MB.
    
    Examples
    --------
    >>> get_file_size_mb("data/raw/walmart_store_sales.csv")
    2.5
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file.
    
    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    return Path(file_path).exists()
