"""
Logging utility for the project.

This module provides a centralized logging configuration.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger.
    
    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the module).
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Default is "INFO".
    format_string : str, optional
        Custom format string for log messages.
        Default format includes timestamp, name, level, and message.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting data processing...")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger
