"""
Configuration file for the churn prediction project.

This file contains default settings and parameters for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "churn_prediction" / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

# Data settings
DEFAULT_DATASET = "Dataset_Cay quyet dinh_HV.xlsx"
DEFAULT_SHEET_NAME = "final_dataset"
TARGET_COLUMN = None  # Will be auto-detected as last column
ID_COLUMN = "id"  # Column to remove if present

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Preprocessing settings
HANDLE_OUTLIERS = True
OUTLIER_METHOD = "iqr"
OUTLIER_THRESHOLD = 1.5
IMBALANCE_THRESHOLD = 1.5
RESAMPLING_METHOD = "smote"

# Feature selection settings
FEATURE_IMPORTANCE_THRESHOLD = 0.95
TOP_N_FEATURES = 15
FEATURE_SELECTION_METHODS = [
    'model_importance_mean',
    'model_importance_median', 
    'cumulative_95'
]

# Hyperparameter tuning settings
PARAM_GRID = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

# Visualization settings
FIGURE_DPI = 300
TREE_MAX_DEPTH = 4
TREE_FIGSIZE = (20, 12)
PLOT_FIGSIZE = (10, 6)
FEATURE_PLOT_FIGSIZE = (12, 8)

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Web app settings
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"

# Model configurations
BASELINE_MODEL_NAME = "baseline_model.pkl"
OPTIMIZED_MODEL_NAME = "optimized_model.pkl"
SELECTED_FEATURES_FILE = "selected_features.txt"

# Class names for visualization
CLASS_NAMES = ['No Churn', 'Churn']

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Configuration loaded and directories created.") 