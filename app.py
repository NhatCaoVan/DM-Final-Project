import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import warnings
import pickle
import os
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from churn_prediction.src.preprocessing import preprocess_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Project paths
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = OUTPUTS_DIR / "models"

# Global variables to store the preprocessor and feature names
_preprocessor = None
_feature_names = None
_numeric_cols = None
_categorical_cols = None

def load_preprocessor():
    """
    Load the preprocessor by running the same preprocessing pipeline used during training.
    This ensures we apply the exact same transformations.
    """
    global _preprocessor, _feature_names, _numeric_cols, _categorical_cols
    
    if _preprocessor is None:
        logger.info("Loading preprocessor from training data...")
        try:
            # Use the same dataset and preprocessing as during training (excludes ID column)
            data = preprocess_pipeline("datasets/Dataset_Cay quyet dinh_HV.xlsx", id_col='id')
            _preprocessor = data['preprocessor']
            _feature_names = data['feature_names']
            _numeric_cols = data['numeric_cols']
            _categorical_cols = data['categorical_cols']
            
            # Log which ID column was excluded
            if data.get('id_col_excluded'):
                logger.info(f"ID column '{data['id_col_excluded']}' excluded from model features")
            
            logger.info(f"Preprocessor loaded successfully. Features: {len(_feature_names)}")
            logger.info(f"Numeric columns: {_numeric_cols}")
            logger.info(f"Categorical columns: {_categorical_cols}")
            logger.info(f"Feature names: {_feature_names}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise
    
    return _preprocessor, _feature_names, _numeric_cols, _categorical_cols

def preprocess_input(input_dict):
    """
    Preprocess the input data using the same preprocessor used during training.
    
    Parameters
    ----------
    input_dict : dict
        Dictionary with raw input features
        
    Returns
    -------
    np.ndarray
        Preprocessed feature array ready for model prediction
    """
    logger.info(f"Raw input values: {input_dict}")
    
    # Load the preprocessor
    preprocessor, feature_names, numeric_cols, categorical_cols = load_preprocessor()
    
    # Create a DataFrame with the input data in the same format as training data
    input_df = pd.DataFrame([input_dict])
    
    # Add missing columns that the model expects but aren't in the UI
    # Note: ID column is deliberately excluded from the model
    if 'month' in numeric_cols and 'month' not in input_df.columns:
        input_df['month'] = 6  # Default to June (middle of year)
    if 'year' in numeric_cols and 'year' not in input_df.columns:
        input_df['year'] = 2024  # Default to current year
    
    # Ensure all required columns are present (except ID which is excluded)
    for col in numeric_cols + categorical_cols:
        if col not in input_df.columns:
            if col in numeric_cols:
                input_df[col] = 0  # Default numeric value
                logger.warning(f"Missing numeric column {col}, using default value 0")
            else:
                # For categorical columns, use a value that exists in the training data
                if col == 'gender':
                    input_df[col] = 'Male'  # Default gender
                elif col == 'district':
                    input_df[col] = 'Hue'  # Default district
                elif col == 'data_package':
                    input_df[col] = 'BIG70'  # Default package
                else:
                    input_df[col] = 'Unknown'  # Generic default
                logger.warning(f"Missing categorical column {col}, using default value {input_df[col].iloc[0]}")
    
    # Select only the columns used during training, in the correct order (excludes ID)
    input_df = input_df[numeric_cols + categorical_cols]
    
    logger.info(f"Input DataFrame shape: {input_df.shape}")
    logger.info(f"Input DataFrame columns: {list(input_df.columns)}")
    logger.info(f"Input DataFrame values: {input_df.iloc[0].to_dict()}")
    
    # Apply the same preprocessing
    try:
        processed_array = preprocessor.transform(input_df)
        logger.info(f"Processed array shape: {processed_array.shape}")
        logger.info(f"Expected features: {len(feature_names)}")
        
        # Log some sample values from the processed array
        if processed_array.shape[1] > 0:
            sample_values = processed_array[0][:min(10, processed_array.shape[1])]
            logger.info(f"First 10 processed values: {sample_values}")
        
        return processed_array
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def make_prediction(input_data):
    """
    Make a prediction using the model, handling all feature adjustments directly.
    
    Parameters
    ----------
    input_data : dict
        Dictionary containing input features
        
    Returns
    -------
    float
        Probability of churn
    """
    try:
        # Preprocess the input data using the same pipeline as training
        X = preprocess_input(input_data)
        
        # Load the model
        model_path = MODELS_DIR / "baseline_model.pkl"
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return 0.5
        
        # Load the model
        model = joblib.load(model_path)
        
        logger.info(f"Input array shape: {X.shape}")
        logger.info(f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features")
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)
                churn_prob = probas[0][1]  # Probability of class 1 (churn)
                logger.info(f"Prediction successful: {churn_prob:.4f}")
                return churn_prob
            except Exception as e:
                logger.error(f"Error in predict_proba: {str(e)}")
                # Fallback to binary prediction
                try:
                    prediction = model.predict(X)[0]
                    logger.info(f"Fallback prediction: {prediction}")
                    return float(prediction)
                except Exception as e2:
                    logger.error(f"Error in predict fallback: {str(e2)}")
                    return 0.5
        else:
            # Use predict directly
            prediction = model.predict(X)[0]
            logger.info(f"Direct prediction: {prediction}")
            return float(prediction)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        logger.error(f"Error in make_prediction: {str(e)}")
        return 0.5  # Return 50% probability as fallback

# Main title
st.title("Customer Churn Prediction")

# Create prediction form
st.header("Customer Data Input")

# Create columns for input form
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("Demographics")
    age = st.slider("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    district = st.selectbox("District", ["Hue", "Phu Vang", "Phu Loc", "Huong Thuy", "Vinh hoi"])

with col2:
    st.subheader("Service Information")
    data_package = st.selectbox("Data Package", 
                               ["BIG70", "BIG90", "D120", "DCH", "DMAX", "DMAX100", 
                                "GM30", "SP50", "THAGA60", "THAGA70", "DINO70", "BM69"])
    data_volume = st.number_input("Data Volume (GB)", 0.0, 100.0, 5.0, step=1.0)
    sms_volume = st.number_input("SMS Volume", 0, 1000, 50, step=10)

with col3:
    st.subheader("Spending Patterns")
    data_spending = st.number_input("Data Spending", 0, 500000, 100000, step=10000, format="%d")
    voice_spending = st.number_input("Voice Spending", 0, 200000, 50000, step=5000, format="%d")
    voice_duration = st.number_input("Voice Duration (min)", 0.0, 1000.0, 100.0, step=10.0)
    sms_spending = st.number_input("SMS Spending", 0, 50000, 5000, step=1000, format="%d")

# Prediction button
predict_button = st.button("Predict Churn Risk", type="primary")

# If prediction button is clicked
if predict_button:
    try:
        with st.spinner("Analyzing customer data..."):
            # Create feature dictionary with raw input features
            input_dict = {
                'age': age,
                'data_volume': data_volume, 
                'data_spending': data_spending,
                'sms_volume': sms_volume, 
                'sms_spending': sms_spending,
                'voice_duration': voice_duration, 
                'voice_spending': voice_spending,
                'gender': gender,
                'district': district,
                'data_package': data_package
            }
            
            # Make prediction (preprocessing is handled inside)
            churn_prob = make_prediction(input_dict)
            
            # Ensure probability is between 0 and 1
            churn_prob = max(0, min(1, churn_prob))
            
            # Display results
            st.header("Prediction Results")
            
            # Create two columns for metrics
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
            
            with res_col2:
                prediction = "Likely to Churn" if churn_prob > 0.5 else "Likely to Stay"
                st.metric("Prediction", prediction)
            
            # Simple progress bar for probability
            st.progress(float(churn_prob))
            
            # Risk level
            if churn_prob > 0.7:
                st.error("High risk of churn")
            elif churn_prob > 0.4:
                st.warning("Medium risk of churn")
            else:
                st.success("Low risk of churn")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        with st.expander("View Error Details"):
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("Customer Churn Prediction | Built with Streamlit & scikit-learn") 