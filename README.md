# Walmart Sales ML

A machine learning project for predicting weekly sales of Walmart stores using historical data, economic indicators, and holiday information.

## 📋 Project Overview

This is an educational ML project that demonstrates:
- Time-series data handling
- Feature engineering from temporal data
- Machine learning regression models
- Proper train/test splitting for time-series

**Dataset**: Walmart Store Sales (Kaggle)  
**Target**: Predict `Weekly_Sales` for Store #1  
**Time range**: 2010-02-05 to 2012-11-01  
**Train period**: 2010-2011  
**Test period**: 2012

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd walmart-sales-ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the Walmart Store Sales dataset from Kaggle
   - Place the CSV file at: `data/raw/walmart_store_sales.csv`

## 📁 Project Structure

```
walmart-sales-ml/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore patterns
│
├── data/
│   ├── raw/                  # Raw data (not versioned)
│   ├── sample/               # Sample data for testing
│   └── processed/            # Processed data
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_model_tests.ipynb # Model experimentation
│
├── src/
│   ├── config.py            # Project configuration
│   │
│   ├── data/
│   │   ├── load_data.py     # Data loading utilities
│   │   ├── split_time.py    # Time-based train/test split
│   │   └── make_sample.py   # Create sample datasets
│   │
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   │
│   ├── models/
│   │   ├── train_baseline.py   # Baseline model (Linear Regression)
│   │   ├── train_random_forest.py # Random Forest model
│   │   ├── evaluate.py         # Model evaluation
│   │   └── save_load.py        # Model persistence
│   │
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       ├── paths.py         # Path utilities
│       └── logger.py        # Logging configuration
│
├── models/                   # Saved models
├── tests/                    # Unit tests
└── scripts/                  # Standalone scripts
```

## 📊 Dataset

### Columns
- `Store`: Store number
- `Date`: Week date
- `Weekly_Sales`: Sales for the given store (TARGET)
- `Holiday_Flag`: Whether the week contains a holiday (0/1)
- `Temperature`: Average temperature (°F)
- `Fuel_Price`: Fuel price in the region
- `CPI`: Consumer Price Index
- `Unemployment`: Unemployment rate

### Key Characteristics
- **Time-series data**: Requires chronological train/test split
- **Single store focus**: Currently analyzing Store #1 only
- **Weekly granularity**: One observation per week

## 🔧 Usage

### 1. Load and Explore Data

```python
from src.data.load_data import load_raw_data, get_data_summary
from src.config import RAW_DATA_FILE, TARGET_STORE

# Load data for Store 1
df = load_raw_data(RAW_DATA_FILE, store_id=TARGET_STORE)

# Print summary
get_data_summary(df)
```

### 2. Feature Engineering

```python
from src.features.build_features import build_features

# Extract date features (year, month, week_of_year)
df_features = build_features(df, include_date_features=True)
```

### 3. Train/Test Split

```python
from src.data.split_time import get_train_test_split
from src.config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE

# Time-based split (2010-2011 for training, 2012 for testing)
X_train, X_test, y_train, y_test = get_train_test_split(
    df_features,
    train_start=TRAIN_START_DATE,
    train_end=TRAIN_END_DATE,
    test_start=TEST_START_DATE,
    test_end=TEST_END_DATE
)
```

### 4. Train and Evaluate Models

```python
from sklearn.linear_model import LinearRegression
from src.utils.metrics import calculate_metrics, print_metrics

# Train baseline model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
metrics = calculate_metrics(y_test, y_pred)
print_metrics(metrics, "Linear Regression")
```

## 📈 Models

### Baseline Model
- **Algorithm**: Linear Regression
- **Purpose**: Simple baseline to establish minimum performance

### Advanced Model
- **Algorithm**: Random Forest Regressor
- **Purpose**: Capture non-linear relationships and interactions

## 📊 Evaluation Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **R²** (R-squared): Proportion of variance explained (0 to 1)

## 🛠️ Development

### Code Style
- Clear, readable code with docstrings
- Separation of concerns (data/features/models/utils)
- Educational comments for learning purposes

### Testing
```bash
pytest tests/
```

## 📝 Notes

- This is an **educational project** focused on learning ML workflows
- Currently uses **Store #1 only** for simplicity
- **No frontend or API** in the current version (Flask will be added later)
- Data must be obtained separately from Kaggle

## 🔮 Future Enhancements

- [ ] Add more stores
- [ ] Implement lag features (use previous weeks' sales)
- [ ] Implement rolling window features
- [ ] Hyperparameter tuning
- [ ] Cross-validation for time-series
- [ ] Flask API for predictions
- [ ] Model deployment

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

This is a personal learning project, but suggestions are welcome!

---

**Happy Learning! 🎓**
