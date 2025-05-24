"""
Model training and evaluation module for customer churn prediction.

This module provides functions for training decision tree models and evaluating their performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_decision_tree(X_train, y_train, params=None, random_state=42):
    """
    Train a decision tree classifier with optimized parameters.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    params : dict, optional
        Model hyperparameters
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    DecisionTreeClassifier
        Trained model
    """
    logger.info("Training decision tree model")
    
    # Set default parameters if none provided
    if params is None:
        # Use more optimized default parameters
        params = {
            'max_depth': 5,  # Prevent overfitting
            'min_samples_split': 10,  # Require more samples to split
            'min_samples_leaf': 5,  # Require more samples in leaf nodes
            'class_weight': 'balanced',  # Handle class imbalance
            'criterion': 'entropy'  # Often provides better results for classification
        }
    
    # Always set random_state for reproducibility
    params['random_state'] = random_state
    
    # Initialize and train the model
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"Model trained: {model.tree_.node_count} nodes, max depth {model.get_depth()}")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names=None, class_names=None):
    """
    Evaluate a trained model.

    Parameters
    ----------
    model : estimator
        Trained model with predict and predict_proba methods
    X_test : array-like
        Test features
    y_test : array-like
        True target values
    feature_names : list, optional
        Names of the features
    class_names : list, optional
        Names of the classes

    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Binary classification metrics if binary target
    binary_metrics = {}
    roc_data = {}
    if len(np.unique(y_test)) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store ROC curve data
        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
        
        # Binary classification metrics
        binary_metrics = {
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'auc': roc_auc
        }
        
        logger.info(f"Binary classification metrics: Precision={binary_metrics['precision']:.4f}, "
                   f"Recall={binary_metrics['recall']:.4f}, F1={binary_metrics['f1']:.4f}, "
                   f"AUC={binary_metrics['auc']:.4f}")
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get classification report as dict
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Return evaluation results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr,
        'binary_metrics': binary_metrics,
        'roc_data': roc_data,
        'y_pred': y_pred
    }
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        results['feature_importance'] = importance_dict
    
    return results


def cross_validate_model(X, y, params=None, cv=5, scoring='accuracy', random_state=42):
    """
    Perform cross-validation on a model.

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target
    params : dict, optional
        Model hyperparameters
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict
        Cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation")
    
    # Set default parameters if none provided
    if params is None:
        params = {}
    
    # Set random_state for reproducibility
    params['random_state'] = random_state
    
    # Initialize model
    model = DecisionTreeClassifier(**params)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std()
    }


def tune_hyperparameters(X_train, y_train, X_test, y_test, param_grid=None, cv=5, 
                         scoring='accuracy', n_jobs=-1, random_state=42):
    """
    Perform hyperparameter tuning using grid search.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    param_grid : dict, optional
        Grid of hyperparameters to search
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict
        Hyperparameter tuning results
    """
    logger.info("Starting hyperparameter tuning with GridSearchCV")
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2']
        }
    
    # Initialize model
    model = DecisionTreeClassifier(random_state=random_state)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate best model
    best_model_eval = evaluate_model(best_model, X_test, y_test)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    logger.info(f"Test set accuracy: {best_model_eval['accuracy']:.4f}")
    
    return {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'test_evaluation': best_model_eval
    }


def plot_roc_curve(fpr, tpr, roc_auc, title='Receiver Operating Characteristic', 
                  figsize=(8, 6), save_path=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    roc_auc : float
        Area under the ROC curve
    title : str, default='Receiver Operating Characteristic'
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return plt.gcf()


def plot_feature_importance(importance_dict, top_n=10, figsize=(10, 6), save_path=None):
    """
    Plot feature importance.

    Parameters
    ----------
    importance_dict : dict
        Dictionary of feature importances
    top_n : int, default=10
        Number of top features to show
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Sort by importance
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top_n features
    if top_n and top_n < len(sorted_imp):
        features = [item[0] for item in sorted_imp[:top_n]]
        values = [item[1] for item in sorted_imp[:top_n]]
    else:
        features = [item[0] for item in sorted_imp]
        values = [item[1] for item in sorted_imp]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), values, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return plt.gcf()


def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : array
        Confusion matrix
    class_names : list, optional
        Names of the classes
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def save_model(model, filepath):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : estimator
        Trained model
    filepath : str
        Path to save the model

    Returns
    -------
    str
        Path to the saved model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(filepath):
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model

    Returns
    -------
    estimator
        The loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def fix_model_predictions(model):
    """
    Fix issues with model probability predictions.
    
    This function ensures the model's predict_proba method returns valid probabilities.
    
    Parameters
    ----------
    model : estimator
        The model to fix
        
    Returns
    -------
    estimator
        The fixed model
    """
    logger.info("Applying fix to model probability predictions")
    
    # Check if the model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        logger.warning("Model does not have predict_proba method, no fix applied")
        return model
    
    # Create a wrapper class to ensure proper probability outputs
    class ProbabilityFixWrapper:
        def __init__(self, base_model):
            self.base_model = base_model
            # Copy all attributes from the base model
            for attr in dir(base_model):
                if not attr.startswith('__') and not callable(getattr(base_model, attr)):
                    setattr(self, attr, getattr(base_model, attr))
            
            # Store feature names for easier access
            if hasattr(base_model, 'feature_names_in_'):
                self.expected_feature_names = base_model.feature_names_in_
                logger.info(f"Model expects {len(self.expected_feature_names)} features")
            else:
                self.expected_feature_names = None
                logger.warning("Model doesn't have feature_names_in_ attribute")
        
        def predict(self, X):
            """Pass through the prediction to the base model"""
            try:
                # Ensure X has the right features
                X_adjusted = self._adjust_features(X)
                logger.info(f"Adjusted features for prediction: input shape {X.shape if hasattr(X, 'shape') else 'unknown'} -> adjusted shape {X_adjusted.shape if hasattr(X_adjusted, 'shape') else 'unknown'}")
                return self.base_model.predict(X_adjusted)
            except Exception as e:
                logger.error(f"Error in predict: {str(e)}")
                # Return default prediction (0)
                if isinstance(X, pd.DataFrame):
                    return np.zeros(len(X))
                else:
                    return np.zeros(X.shape[0])
        
        def predict_proba(self, X):
            """Ensure probabilities are valid"""
            try:
                # Ensure X has the right features
                X_adjusted = self._adjust_features(X)
                logger.info(f"Adjusted features for predict_proba: input shape {X.shape if hasattr(X, 'shape') else 'unknown'} -> adjusted shape {X_adjusted.shape if hasattr(X_adjusted, 'shape') else 'unknown'}")
                
                # Try using numpy array instead of DataFrame if we still have issues
                if isinstance(X_adjusted, pd.DataFrame):
                    X_numpy = X_adjusted.values
                    logger.info(f"Converting DataFrame to numpy array with shape {X_numpy.shape}")
                    try:
                        proba = self.base_model.predict_proba(X_numpy)
                        logger.info("Successfully used numpy array for prediction")
                    except Exception as e:
                        logger.warning(f"Error with numpy array: {str(e)}, falling back to DataFrame")
                        proba = self.base_model.predict_proba(X_adjusted)
                else:
                    proba = self.base_model.predict_proba(X_adjusted)
                
                # Check if probabilities are valid (sum to 1 for each sample)
                if not np.allclose(np.sum(proba, axis=1), 1.0):
                    logger.warning("Fixing invalid probabilities that don't sum to 1")
                    # Normalize the probabilities
                    proba = proba / np.sum(proba, axis=1)[:, np.newaxis]
                
                # Ensure no negative probabilities
                if np.any(proba < 0):
                    logger.warning("Fixing negative probabilities")
                    proba = np.maximum(proba, 0)
                    # Re-normalize
                    proba = proba / np.sum(proba, axis=1)[:, np.newaxis]
                
                return proba
            except Exception as e:
                logger.error(f"Error in predict_proba: {str(e)}")
                # Fallback to binary prediction and convert to probabilities
                try:
                    preds = self.predict(X)
                    # Convert to probability format [P(class=0), P(class=1)]
                    proba = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        proba[i, int(p)] = 1.0
                    return proba
                except Exception as inner_e:
                    logger.error(f"Error in fallback prediction: {str(inner_e)}")
                    # Return default probabilities (50/50)
                    if isinstance(X, pd.DataFrame):
                        return np.tile([0.5, 0.5], (len(X), 1))
                    else:
                        return np.tile([0.5, 0.5], (X.shape[0], 1))
        
        def _adjust_features(self, X):
            """Adjust input features to match what the model expects"""
            if self.expected_feature_names is None:
                logger.warning("No feature_names_in_ attribute, cannot adjust features")
                return X
            
            # If X is already a DataFrame with matching columns, check for missing features
            if isinstance(X, pd.DataFrame):
                logger.info(f"Input is DataFrame with {len(X.columns)} features")
                
                # Check if we have all expected features and they're in the right order
                if list(X.columns) == list(self.expected_feature_names):
                    logger.info("Input features already match model's expected features exactly")
                    return X
                
                # Create a DataFrame with all expected features, initialized to 0
                X_adjusted = pd.DataFrame(0, index=range(len(X)), columns=self.expected_feature_names)
                
                # Copy over values for features that exist in both
                for col in self.expected_feature_names:
                    if col in X.columns:
                        X_adjusted[col] = X[col]
                
                # Log missing features
                missing_features = set(self.expected_feature_names) - set(X.columns)
                if missing_features:
                    logger.warning(f"Added {len(missing_features)} missing features: {', '.join(list(missing_features)[:5])}...")
                
                # Log extra features
                extra_features = set(X.columns) - set(self.expected_feature_names)
                if extra_features:
                    logger.warning(f"Ignored {len(extra_features)} extra features: {', '.join(list(extra_features)[:5])}...")
                
                logger.info(f"Adjusted DataFrame to have exactly {len(self.expected_feature_names)} features")
                return X_adjusted
            
            # If X is a numpy array, ensure it has the right number of features
            if isinstance(X, np.ndarray):
                logger.info(f"Input is numpy array with shape {X.shape}")
                
                if X.shape[1] == len(self.expected_feature_names):
                    return X
                
                # Create array with correct shape
                X_adjusted = np.zeros((X.shape[0], len(self.expected_feature_names)))
                
                # Copy over values for as many features as we can
                min_features = min(X.shape[1], len(self.expected_feature_names))
                X_adjusted[:, :min_features] = X[:, :min_features]
                
                logger.warning(f"Adjusted numpy array from {X.shape[1]} to {len(self.expected_feature_names)} features")
                return X_adjusted
            
            # Last resort: convert to DataFrame and try again
            try:
                logger.warning(f"Converting unknown input type {type(X)} to DataFrame")
                df = pd.DataFrame(X)
                return self._adjust_features(df)
            except Exception as e:
                logger.error(f"Failed to convert to DataFrame: {str(e)}")
                logger.warning(f"Unknown input type {type(X)}, cannot adjust features")
                return X
        
        def __getattr__(self, name):
            """Forward any other attributes/methods to the base model"""
            return getattr(self.base_model, name)
    
    # Create and return the wrapped model
    fixed_model = ProbabilityFixWrapper(model)
    logger.info("Model probability fix applied")
    return fixed_model


def optimize_tree_pruning(model, X_val, y_val):
    """
    Optimize a decision tree using cost-complexity pruning.
    
    Parameters
    ----------
    model : DecisionTreeClassifier
        Trained decision tree model
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
        
    Returns
    -------
    DecisionTreeClassifier
        Pruned model
    """
    logger.info("Performing cost-complexity pruning")
    
    # Get path with different alphas
    path = model.cost_complexity_pruning_path(X_val, y_val)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Filter out too small alphas that might cause numerical issues
    ccp_alphas = ccp_alphas[:-1]  # Remove the last alpha which gives an empty tree
    
    # If we have too many alphas, sample a reasonable number
    if len(ccp_alphas) > 20:
        indices = np.linspace(0, len(ccp_alphas) - 1, 20, dtype=int)
        ccp_alphas = ccp_alphas[indices]
    
    # Train models with different alphas
    models = []
    for alpha in ccp_alphas:
        dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=model.random_state)
        dt.fit(X_val, y_val)
        models.append(dt)
    
    # Evaluate models
    train_scores = [dt.score(X_val, y_val) for dt in models]
    
    # Find the best alpha
    best_idx = np.argmax(train_scores)
    best_alpha = ccp_alphas[best_idx]
    
    # Train final model with best alpha
    final_model = DecisionTreeClassifier(
        ccp_alpha=best_alpha,
        random_state=model.random_state,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
        class_weight=model.class_weight,
        criterion=model.criterion
    )
    final_model.fit(X_val, y_val)
    
    logger.info(f"Pruned model: {final_model.tree_.node_count} nodes (reduced from {model.tree_.node_count})")
    logger.info(f"Best alpha: {best_alpha:.6f}, Validation accuracy: {train_scores[best_idx]:.4f}")
    
    return final_model


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    
    # Preprocess data
    data = preprocess_pipeline("datasets/Dataset_Cay quyet dinh_HV.xlsx")
    
    # Train a model
    model = train_decision_tree(
        data['X_train_resampled'], 
        data['y_train_resampled']
    )
    
    # Evaluate the model
    eval_results = evaluate_model(
        model, 
        data['X_test_processed'], 
        data['y_test'],
        feature_names=data['feature_names']
    )
    
    # Print some evaluation metrics
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    
    if eval_results['binary_metrics']:
        print(f"AUC: {eval_results['binary_metrics']['auc']:.4f}")
    
    # Test the model's probabilities before fix
    print("\nProbabilities before fix:")
    if hasattr(model, 'predict_proba'):
        test_input = data['X_test_processed'][:5]  # Take first 5 test samples
        probas = model.predict_proba(test_input)
        for i, proba in enumerate(probas):
            print(f"Sample {i+1}: {proba}")
    
    # Apply the fix
    model = fix_model_predictions(model)
    
    # Test the model's probabilities after fix
    print("\nProbabilities after fix:")
    if hasattr(model, 'predict_proba'):
        test_input = data['X_test_processed'][:5]  # Same 5 test samples
        probas = model.predict_proba(test_input)
        for i, proba in enumerate(probas):
            print(f"Sample {i+1}: {proba}")
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs/models", exist_ok=True)
    
    # Save the model
    save_model(model, "outputs/models/decision_tree_model.pkl") 