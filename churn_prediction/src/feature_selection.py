"""
Feature selection module for the churn prediction model.

This module provides functions for analyzing and selecting features for the model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_feature_importance(model, X, y, feature_names, method='model', n_repeats=10, random_state=42):
    """
    Get feature importance using different methods.

    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ attribute
    X : array-like
        Features data
    y : array-like
        Target data
    feature_names : list
        List of feature names
    method : str, default='model'
        Method to use ('model', 'permutation', or 'both')
    n_repeats : int, default=10
        Number of repeats for permutation importance
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with feature importance results
    """
    logger.info(f"Calculating feature importance using {method} method")
    results = {}
    
    # Model-based importance (e.g., impurity-based for trees)
    if method in ['model', 'both'] and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        results['model_importance'] = importance_dict
        
        # Sort by importance
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        results['sorted_model_importance'] = sorted_importance
        
        # Get cumulative importance
        cumulative = []
        cum_sum = 0
        for name, imp in sorted_importance:
            cum_sum += imp
            cumulative.append((name, imp, cum_sum))
        results['cumulative_importance'] = cumulative
        
        logger.info(f"Top 5 features by model importance: {sorted_importance[:5]}")
    
    # Permutation importance
    if method in ['permutation', 'both']:
        start_time = time.time()
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state
        )
        
        # Create a dictionary of mean permutation importance
        perm_importance_dict = dict(zip(
            feature_names, 
            perm_importance.importances_mean
        ))
        
        # Sort by importance
        sorted_perm_importance = sorted(
            perm_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results['permutation_importance'] = perm_importance_dict
        results['sorted_permutation_importance'] = sorted_perm_importance
        results['permutation_importance_std'] = dict(zip(
            feature_names, 
            perm_importance.importances_std
        ))
        
        logger.info(f"Permutation importance calculated in {time.time() - start_time:.2f} seconds")
        logger.info(f"Top 5 features by permutation importance: {sorted_perm_importance[:5]}")
    
    return results


def plot_feature_importance(importance_dict, top_n=15, figsize=(12, 8), save_path=None):
    """
    Plot feature importance.

    Parameters
    ----------
    importance_dict : dict
        Dictionary of feature importances
    top_n : int, default=15
        Number of top features to show
    figsize : tuple, default=(12, 8)
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


def plot_cumulative_importance(importance_dict, threshold=0.95, figsize=(10, 6), save_path=None):
    """
    Plot cumulative feature importance.

    Parameters
    ----------
    importance_dict : dict
        Dictionary of feature importances
    threshold : float, default=0.95
        Threshold for cumulative importance
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    tuple
        Figure object and number of features needed to reach threshold
    """
    # Sort by importance
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative importance
    cum_importance = []
    cum_sum = 0
    for i, (feature, importance) in enumerate(sorted_imp):
        cum_sum += importance
        cum_importance.append(cum_sum)
    
    # Find how many features account for threshold of importance
    features_for_threshold = next((i+1 for i, x in enumerate(cum_importance) if x >= threshold), len(cum_importance))
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(sorted_imp) + 1), cum_importance, 'b-')
    plt.hlines(y=threshold, xmin=1, xmax=len(sorted_imp), color='r', linestyles='dashed')
    
    # Highlight threshold point
    if features_for_threshold < len(sorted_imp):
        plt.plot(features_for_threshold, cum_importance[features_for_threshold - 1], 'ro')
        plt.annotate(f'{features_for_threshold} features', 
                    xy=(features_for_threshold, cum_importance[features_for_threshold - 1]), 
                    xytext=(features_for_threshold + 1, cum_importance[features_for_threshold - 1] - 0.05),
                    arrowprops=dict(arrowstyle='->'))
    
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title(f'Cumulative Feature Importance (Threshold: {threshold:.0%})')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cumulative importance plot saved to {save_path}")
    
    logger.info(f"{features_for_threshold} features account for {threshold:.0%} of importance")
    
    return plt.gcf(), features_for_threshold


def analyze_feature_correlation(X, feature_names=None, threshold=0.8, figsize=(12, 10), save_path=None):
    """
    Analyze and visualize feature correlations.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    feature_names : list, optional
        Names of the features
    threshold : float, default=0.8
        Threshold for high correlation
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    dict
        Dictionary with correlation results
    """
    logger.info("Analyzing feature correlation")
    
    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    # Compute correlation matrix
    correlation_matrix = X.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=figsize)
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix plot saved to {save_path}")
    
    # Find highly correlated features
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    correlation_matrix.columns[i], 
                    correlation_matrix.columns[j], 
                    correlation_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (threshold: {threshold})")
        for pair in high_corr_pairs[:5]:  # Log only the first 5 pairs
            logger.info(f"  {pair[0]} and {pair[1]}: {pair[2]:.3f}")
    else:
        logger.info(f"No highly correlated features found (threshold: {threshold})")
    
    return {
        'correlation_matrix': correlation_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'figure': plt.gcf()
    }


def select_features_from_model(model, X, feature_names, threshold='mean'):
    """
    Select features from a trained model using SelectFromModel.

    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ attribute
    X : array-like
        Feature data
    feature_names : list
        Names of the features
    threshold : str or float, default='mean'
        Threshold for feature selection

    Returns
    -------
    tuple
        Selected feature names, mask, and transformed data
    """
    logger.info(f"Selecting features from model with threshold: {threshold}")
    
    # Initialize selector
    selector = SelectFromModel(model, threshold=threshold)
    
    # Fit and transform
    X_selected = selector.transform(X)
    
    # Get feature mask and names
    feature_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_mask[i]]
    
    logger.info(f"Selected {len(selected_features)}/{len(feature_names)} features using SelectFromModel")
    
    return selected_features, feature_mask, X_selected


def select_k_best_features(X, y, feature_names, k=10, score_func=f_classif):
    """
    Select top k features using univariate statistical tests.

    Parameters
    ----------
    X : array-like
        Feature data
    y : array-like
        Target data
    feature_names : list
        Names of the features
    k : int, default=10
        Number of top features to select
    score_func : callable, default=f_classif
        Function for scoring features

    Returns
    -------
    tuple
        Selected feature names, mask, and SelectKBest object
    """
    logger.info(f"Selecting {k} best features using {score_func.__name__}")
    
    # Initialize selector
    selector = SelectKBest(score_func=score_func, k=k)
    
    # Fit and transform
    selector.fit(X, y)
    
    # Get feature mask and names
    feature_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_mask[i]]
    
    # Get scores
    scores = selector.scores_
    p_values = selector.pvalues_ if hasattr(selector, 'pvalues_') else None
    
    logger.info(f"Selected {len(selected_features)}/{len(feature_names)} features using SelectKBest")
    
    return selected_features, feature_mask, selector


def select_features_by_cumulative_importance(importance_dict, threshold=0.95):
    """
    Select features based on cumulative importance up to a threshold.

    Parameters
    ----------
    importance_dict : dict
        Dictionary of feature importances
    threshold : float, default=0.95
        Cumulative importance threshold

    Returns
    -------
    list
        Selected feature names
    """
    logger.info(f"Selecting features with cumulative importance threshold: {threshold}")
    
    # Sort by importance
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative importance
    cum_sum = 0
    selected_features = []
    
    for feature, importance in sorted_imp:
        cum_sum += importance
        selected_features.append(feature)
        
        if cum_sum >= threshold:
            break
    
    logger.info(f"Selected {len(selected_features)}/{len(importance_dict)} features (cumulative importance: {cum_sum:.4f})")
    
    return selected_features


def save_selected_features(feature_names, file_path, method_name="Feature Selection"):
    """
    Save selected feature names to a file.

    Parameters
    ----------
    feature_names : list
        List of selected feature names
    file_path : str
        Path to save the file
    method_name : str, default="Feature Selection"
        Name of the feature selection method

    Returns
    -------
    str
        Path where the file was saved
    """
    try:
        with open(file_path, 'w') as f:
            f.write(f"Selected features using {method_name}:\n")
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        logger.info(f"Selected features saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving selected features: {e}")
        raise


def compare_feature_selection_methods(X, y, feature_names, base_model=None, eval_model=None, 
                                     methods=None, random_state=42):
    """
    Compare different feature selection methods.

    Parameters
    ----------
    X : array-like
        Feature data
    y : array-like
        Target data
    feature_names : list
        Names of the features
    base_model : estimator, optional
        Base model for feature selection methods
    eval_model : estimator, optional
        Model to evaluate selected feature sets
    methods : list, optional
        List of methods to compare
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict
        Results for different feature selection methods
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    logger.info("Comparing feature selection methods")
    
    # Default models if not provided
    if base_model is None:
        base_model = DecisionTreeClassifier(random_state=random_state)
        base_model.fit(X, y)
    
    if eval_model is None:
        eval_model = DecisionTreeClassifier(random_state=random_state)
    
    # Default methods if not provided
    if methods is None:
        methods = [
            'model_importance_mean',
            'model_importance_median',
            'cumulative_95'
        ]
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Evaluate full feature set as baseline
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Baseline accuracy with all {len(feature_names)} features: {baseline_accuracy:.4f}")
    
    # Get feature importance
    importance_results = get_feature_importance(base_model, X, y, feature_names, method='both')
    
    # Results dictionary
    results = {
        'baseline': {
            'features': feature_names,
            'n_features': len(feature_names),
            'accuracy': baseline_accuracy
        }
    }
    
    # Apply different feature selection methods
    for method in methods:
        selected_features = None
        
        if method == 'model_importance_mean':
            selected_features, mask, _ = select_features_from_model(
                base_model, X, feature_names, threshold='mean'
            )
        
        elif method == 'model_importance_median':
            selected_features, mask, _ = select_features_from_model(
                base_model, X, feature_names, threshold='median'
            )
        
        elif method == 'cumulative_95':
            # Use cumulative importance if available
            if 'model_importance' in importance_results:
                selected_features = select_features_by_cumulative_importance(
                    importance_results['model_importance'], threshold=0.95
                )
            else:
                logger.warning(f"Skipping {method} - model importance not available")
                continue
        
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            continue
        
        if selected_features:
            # Get indices of selected features
            feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
            
            # Extract selected features
            X_train_selected = X_train[:, feature_indices]
            X_test_selected = X_test[:, feature_indices]
            
            # Train and evaluate model with selected features
            eval_model.fit(X_train_selected, y_train)
            y_pred = eval_model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[method] = {
                'features': selected_features,
                'n_features': len(selected_features),
                'accuracy': accuracy,
                'feature_reduction': 1 - len(selected_features) / len(feature_names)
            }
            
            logger.info(f"Method: {method}, Features: {len(selected_features)}, "
                       f"Accuracy: {accuracy:.4f}, Reduction: {results[method]['feature_reduction']:.1%}")
    
    return results


def plot_feature_selection_comparison(results, figsize=(12, 6), save_path=None):
    """
    Plot comparison of feature selection methods.

    Parameters
    ----------
    results : dict
        Results from compare_feature_selection_methods
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Extract method names, accuracies, and feature counts
    methods = list(results.keys())
    accuracies = [results[method]['accuracy'] for method in methods]
    feature_counts = [results[method]['n_features'] for method in methods]
    
    # Calculate feature reduction percentages
    baseline_features = results['baseline']['n_features']
    reduction_pct = [(baseline_features - count) / baseline_features * 100 for count in feature_counts]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot accuracy bars
    bars = ax1.bar(methods, accuracies, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Feature Selection Method')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)  # Add some padding
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', color='blue', fontweight='bold')
    
    # Create second y-axis for feature counts
    ax2 = ax1.twinx()
    
    # Plot feature counts as line
    ax2.plot(methods, feature_counts, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Number of Features', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add feature counts as text
    for i, count in enumerate(feature_counts):
        ax2.text(i, count + 0.5, f'{count} ({reduction_pct[i]:.0f}%↓)', 
                color='red', ha='center', va='bottom')
    
    plt.title('Comparison of Feature Selection Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature selection comparison saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    from model import train_decision_tree, evaluate_model
    import pandas as pd
    
    # Preprocess data
    data = preprocess_pipeline("datasets/Dataset_Cay quyet dinh_HV.xlsx")
    
    # Train a model
    model = train_decision_tree(
        data['X_train_resampled'], 
        data['y_train_resampled']
    )
    
    # Get feature importance
    importance_results = get_feature_importance(
        model, 
        data['X_test_processed'], 
        data['y_test'],
        data['feature_names'],
        method='both'
    )
    
    # Plot top features
    plot_feature_importance(
        importance_results['model_importance'], 
        top_n=15, 
        save_path="top_features_importance.png"
    )
    
    # Plot cumulative importance
    fig, n_features = plot_cumulative_importance(
        importance_results['model_importance'],
        save_path="cumulative_importance.png"
    )
    
    # Select features by cumulative importance
    selected_features = select_features_by_cumulative_importance(
        importance_results['model_importance']
    )
    
    # Save selected features
    save_selected_features(
        selected_features, 
        "selected_features.txt", 
        method_name="Top 95% Important Features"
    )
    
    # Compare different feature selection methods
    comparison_results = compare_feature_selection_methods(
        data['X_train_processed'], 
        data['y_train'], 
        data['feature_names'],
        base_model=model
    )
    
    # Plot comparison
    plot_feature_selection_comparison(
        comparison_results,
        save_path="feature_selection_comparison.png"
    )
    
    # Save comparison results to CSV
    pd.DataFrame([
        {
            'Method': method,
            'Features': results['n_features'],
            'Accuracy': results['accuracy'],
            'Reduction': results.get('feature_reduction', 0)
        }
        for method, results in comparison_results.items()
    ]).to_csv("outputs/results/feature_selection_results.csv", index=False) 