"""
Complete example script showing the full ML pipeline.

This script demonstrates how to:
1. Load data
2. Engineer features
3. Split data (time-based)
4. Train a model
5. Evaluate performance

Run this script after placing the dataset at: data/raw/walmart_store_sales.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_FILE,
    TARGET_STORE,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE
)
from src.data.load_data import load_raw_data, get_data_summary, check_data_quality
from src.features.build_features import build_features
from src.data.split_time import get_train_test_split
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.logger import get_logger

from sklearn.linear_model import LinearRegression

# Initialize logger
logger = get_logger(__name__)


def main():
    """
    Main pipeline execution.
    """
    logger.info("="*70)
    logger.info("WALMART SALES PREDICTION - COMPLETE PIPELINE")
    logger.info("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading Data")
    logger.info("="*70)
    
    try:
        df = load_raw_data(RAW_DATA_FILE, store_id=TARGET_STORE)
        get_data_summary(df)
        check_data_quality(df)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("\nPlease download the dataset and place it at:")
        logger.error(f"  {RAW_DATA_FILE}")
        return
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*70)
    
    df_features = build_features(
        df,
        include_date_features=True,
        include_lag_features=False,  # Keep it simple for now
        include_rolling_features=False
    )
    
    # ========================================================================
    # STEP 3: TRAIN/TEST SPLIT (TIME-BASED)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Train/Test Split")
    logger.info("="*70)
    
    X_train, X_test, y_train, y_test = get_train_test_split(
        df_features,
        train_start=TRAIN_START_DATE,
        train_end=TRAIN_END_DATE,
        test_start=TEST_START_DATE,
        test_end=TEST_END_DATE
    )
    
    # ========================================================================
    # STEP 4: TRAIN BASELINE MODEL
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Training Baseline Model (Linear Regression)")
    logger.info("="*70)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Model training complete ✓")
    
    # ========================================================================
    # STEP 5: EVALUATE MODEL
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Model Evaluation")
    logger.info("="*70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Print results
    print_metrics(train_metrics, "Training Set - Linear Regression")
    print_metrics(test_metrics, "Test Set - Linear Regression")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info("Summary:")
    logger.info(f"  - Training samples: {len(X_train):,}")
    logger.info(f"  - Test samples:     {len(X_test):,}")
    logger.info(f"  - Features used:    {X_train.shape[1]}")
    logger.info(f"  - Test R² Score:    {test_metrics['r2']:.4f}")
    logger.info(f"  - Test MAE:         ${test_metrics['mae']:,.2f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
