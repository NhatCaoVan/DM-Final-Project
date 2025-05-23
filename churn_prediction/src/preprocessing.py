"""
Data preprocessing module for the churn prediction model.

This module handles loading, cleaning, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path, sheet_name='final_dataset'):
    """
    Load data from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file
    sheet_name : str, default='final_dataset'
        Name of the sheet to load

    Returns
    -------
    pd.DataFrame
        The loaded dataset
    """
    logger.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def describe_dataset(df, target_col=None):
    """
    Generate descriptive statistics for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to describe
    target_col : str, optional
        The target column name

    Returns
    -------
    dict
        Dictionary containing descriptive statistics
    """
    logger.info("Generating dataset description")
    
    # If target column is not specified, assume it's the last column
    if target_col is None:
        target_col = df.columns[-1]
    
    # Count missing values
    missing_values = df.isnull().sum()
    
    # Get class distribution if target is provided
    class_distribution = None
    class_pct = None
    if target_col in df.columns:
        class_distribution = df[target_col].value_counts().to_dict()
        class_pct = df[target_col].value_counts(normalize=True).to_dict()
    
    description = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': missing_values.to_dict(),
        'numeric_stats': df.describe().to_dict(),
        'target_column': target_col,
        'class_distribution': class_distribution,
        'class_percentages': class_pct
    }
    
    return description


def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detect and remove outliers from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to clean
    columns : list
        List of column names to check for outliers
    method : str, default='iqr'
        Method to use for outlier detection ('iqr' for IQR method)
    threshold : float, default=1.5
        Threshold for the IQR method

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed
    dict
        Dictionary with outlier statistics
    """
    logger.info(f"Checking outliers in {len(columns)} numeric columns")
    df_clean = df.copy()
    outlier_stats = {}
    rows_before = df.shape[0]
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_stats[col] = outliers
            logger.info(f"Column '{col}': {outliers} outliers detected")
            
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    rows_after = df_clean.shape[0]
    removed_rows = rows_before - rows_after
    logger.info(f"Removed {removed_rows} rows containing outliers ({removed_rows/rows_before:.2%} of data)")
    
    return df_clean, {
        'outlier_counts': outlier_stats,
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_removed': removed_rows,
        'percent_removed': removed_rows/rows_before if rows_before > 0 else 0
    }


def prepare_features(df, target_col=None, id_col=None):
    """
    Prepare features and target for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    target_col : str, optional
        Target column name (if None, assumes it's the last column)
    id_col : str, optional
        ID column name to remove (if any)

    Returns
    -------
    tuple
        X (features DataFrame), y (target Series), categorical_cols, numeric_cols
    """
    # Identify target column if not specified
    if target_col is None:
        target_col = df.columns[-1]
    
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remove target column from features
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    elif target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove ID column if specified
    if id_col and id_col in numeric_cols:
        numeric_cols.remove(id_col)
        df = df.drop(id_col, axis=1)
    
    logger.info(f"Feature preparation: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric features")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, categorical_cols, numeric_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data into train and test sets (test size: {test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def create_preprocessor(numeric_cols, categorical_cols):
    """
    Create a column transformer for preprocessing data.

    Parameters
    ----------
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names

    Returns
    -------
    ColumnTransformer
        The preprocessing pipeline
    """
    transformers = []
    
    # Add numeric transformer if there are numeric columns
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    
    # Add categorical transformer if there are categorical columns
    if categorical_cols:
        transformers.append((
            'cat', 
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
            categorical_cols
        ))
    
    return ColumnTransformer(transformers=transformers)


def preprocess_data(X_train, X_test, numeric_cols, categorical_cols):
    """
    Preprocess training and testing data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names

    Returns
    -------
    tuple
        X_train_processed, X_test_processed, preprocessor, feature_names
    """
    logger.info("Preprocessing data")
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = numeric_cols.copy()
    if categorical_cols:
        # Get categorical feature names after one-hot encoding
        try:
            encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not get categorical feature names: {e}")
    
    logger.info(f"Processed features: {len(feature_names)}")
    
    return X_train_processed, X_test_processed, preprocessor, feature_names


def handle_imbalance(X_train, y_train, method='smote', random_state=42, imbalance_threshold=1.5):
    """
    Handle class imbalance in the training data.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    method : str, default='smote'
        Method to use for handling imbalance ('smote' or 'none')
    random_state : int, default=42
        Random seed for reproducibility
    imbalance_threshold : float, default=1.5
        Threshold for considering the data imbalanced

    Returns
    -------
    tuple
        X_train_resampled, y_train_resampled
    """
    # Check if data is imbalanced
    class_counts = pd.Series(y_train).value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count

    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > imbalance_threshold and method == 'smote':
        logger.info("Dataset is imbalanced. Applying SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Original training samples: {len(y_train)}, Resampled: {len(y_train_resampled)}")
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        return X_train_resampled, y_train_resampled
    else:
        if imbalance_ratio <= imbalance_threshold:
            logger.info("Dataset is relatively balanced. No resampling needed.")
        else:
            logger.info(f"Not applying resampling method (method={method}).")
        return X_train, y_train


def preprocess_pipeline(file_path, sheet_name='final_dataset', target_col=None, id_col=None, 
                      test_size=0.2, handle_outliers=True, random_state=42):
    """
    Complete data preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path to the dataset file
    sheet_name : str, default='final_dataset'
        Sheet name for Excel files
    target_col : str, optional
        Target column name
    id_col : str, optional
        ID column to remove
    test_size : float, default=0.2
        Proportion of data to use for testing
    handle_outliers : bool, default=True
        Whether to remove outliers
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing all preprocessing results and artifacts
    """
    # Load data
    df = load_data(file_path, sheet_name)
    
    # Determine target column if not provided
    if target_col is None:
        target_col = df.columns[-1]
    
    # Get dataset description
    description = describe_dataset(df, target_col)
    
    # Handle outliers if requested
    if handle_outliers:
        # Identify numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if id_col and id_col in numeric_cols:
            numeric_cols.remove(id_col)
        
        df_clean, outlier_stats = remove_outliers(df, numeric_cols)
    else:
        df_clean = df.copy()
        outlier_stats = {'handled': False}
    
    # Prepare features
    X, y, categorical_cols, numeric_cols = prepare_features(df_clean, target_col, id_col)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Preprocess data
    X_train_processed, X_test_processed, preprocessor, feature_names = preprocess_data(
        X_train, X_test, numeric_cols, categorical_cols
    )
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_processed, y_train, 
                                                         random_state=random_state)
    
    # Return all results
    return {
        'df_original': df,
        'df_clean': df_clean,
        'description': description,
        'outlier_stats': outlier_stats,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols
    }


if __name__ == "__main__":
    # Example usage
    results = preprocess_pipeline("datasets/Dataset_Cay quyet dinh_HV.xlsx")
    
    # Print some statistics
    print(f"\nDataset shape: {results['df_original'].shape}")
    print(f"After cleaning: {results['df_clean'].shape}")
    print(f"Categorical features: {len(results['categorical_cols'])}")
    print(f"Numeric features: {len(results['numeric_cols'])}")
    print(f"Processed features: {len(results['feature_names'])}")
    print(f"Training samples: {results['X_train'].shape[0]}")
    print(f"Testing samples: {results['X_test'].shape[0]}")
    print(f"Resampled training samples: {results['X_train_resampled'].shape[0]}") 