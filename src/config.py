"""
Configuration file for the Walmart Sales ML project.

This module centralizes all project configuration parameters,
making it easy to modify settings without changing the code.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Raw data file
RAW_DATA_FILE = RAW_DATA_DIR / "walmart_store_sales.csv"

# Store to analyze (starting with one store for simplicity)
TARGET_STORE = 1

# Target variable
TARGET_COLUMN = "Weekly_Sales"

# Feature columns
FEATURE_COLUMNS = [
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment"
]

# Date column
DATE_COLUMN = "Date"

# ============================================================================
# TIME-BASED SPLIT CONFIGURATION
# ============================================================================

# Train period: 2010-2011
TRAIN_START_DATE = "2010-02-05"
TRAIN_END_DATE = "2011-12-31"

# Test period: 2012
TEST_START_DATE = "2012-01-01"
TEST_END_DATE = "2012-11-01"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Date features to extract
DATE_FEATURES = ["year", "month", "week_of_year"]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Model parameters
BASELINE_MODEL_PARAMS = {
    # Linear Regression has no hyperparameters
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to compute
METRICS = ["mae", "rmse", "r2"]

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
