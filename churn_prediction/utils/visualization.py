"""
Visualization utilities for the churn prediction model.

This module provides functions for visualizing decision trees and model results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, export_graphviz
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_decision_tree(model, feature_names=None, class_names=None, max_depth=None, 
                          figsize=(20, 12), save_path=None, dpi=300):
    """
    Visualize a decision tree model with enhanced styling.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Trained decision tree model
    feature_names : list, optional
        Names of the features
    class_names : list, optional
        Names of the classes
    max_depth : int, optional
        Maximum depth to visualize
    figsize : tuple, default=(20, 12)
        Figure size
    save_path : str, optional
        Path to save the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    logger.info(f"Visualizing decision tree (max_depth={max_depth})")
    
    # Create figure with dark background for better contrast
    plt.figure(figsize=figsize)
    plt.style.use('dark_background')
    
    # If class names not provided, create generic ones
    if class_names is None:
        class_names = [str(c) for c in model.classes_]
    
    # Plot the tree with enhanced styling
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=max_depth,
        precision=2,
        proportion=True,
        impurity=True
    )
    
    plt.title(f'Decision Tree (Max Depth = {max_depth if max_depth else "Full"})', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='#0E1117')
        logger.info(f"Decision tree visualization saved to {save_path}")
    
    return plt.gcf()


def export_tree_as_dot(model, feature_names=None, class_names=None, max_depth=None, 
                      output_file='decision_tree.dot'):
    """
    Export a decision tree as a DOT file for visualization with Graphviz.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Trained decision tree model
    feature_names : list, optional
        Names of the features
    class_names : list, optional
        Names of the classes
    max_depth : int, optional
        Maximum depth to include
    output_file : str, default='decision_tree.dot'
        Path to save the DOT file

    Returns
    -------
    str
        Path to the saved DOT file
    """
    logger.info(f"Exporting decision tree to {output_file}")
    
    # If class names not provided, create generic ones
    if class_names is None:
        class_names = [str(c) for c in model.classes_]
    
    # Export as DOT file
    export_graphviz(
        model,
        out_file=output_file,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )
    
    logger.info(f"Decision tree exported to {output_file}")
    
    return output_file


def render_dot_file(dot_file, output_format='png', output_file=None):
    """
    Render a DOT file to various formats using Graphviz.

    Parameters
    ----------
    dot_file : str
        Path to the DOT file
    output_format : str, default='png'
        Output format ('png', 'pdf', 'svg')
    output_file : str, optional
        Path for the output file (without extension)

    Returns
    -------
    str or None
        Path to the rendered file if successful, None otherwise
    """
    try:
        import graphviz
        
        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.splitext(dot_file)[0]
        
        # Load and render the DOT file
        graph = graphviz.Source.from_file(dot_file)
        rendered_file = graph.render(output_file, format=output_format)
        
        logger.info(f"DOT file rendered as {rendered_file}")
        return rendered_file
    
    except ImportError:
        logger.warning("Graphviz Python package not installed. Could not render DOT file.")
        logger.warning("To install: pip install graphviz")
        logger.warning("Note: You also need to install the Graphviz software: https://graphviz.org/download/")
        return None
    
    except Exception as e:
        logger.error(f"Error rendering DOT file: {e}")
        return None


def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6), normalize=False, 
                         save_path=None, cmap='Blues', dpi=300):
    """
    Plot a confusion matrix with enhanced styling.

    Parameters
    ----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of the classes
    figsize : tuple, default=(8, 6)
        Figure size
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    save_path : str, optional
        Path to save the visualization
    cmap : str, default='Blues'
        Colormap for the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure with dark background
    plt.figure(figsize=figsize)
    plt.style.use('dark_background')
    
    # Plot confusion matrix with improved styling
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        xticklabels=class_names, 
        yticklabels=class_names,
        annot_kws={"size": 12},
        linewidths=1,
        linecolor='gray',
        cbar_kws={"shrink": 0.8}
    )
    
    plt.xlabel('Predicted', fontsize=12, labelpad=10)
    plt.ylabel('True', fontsize=12, labelpad=10)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='#0E1117')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    return plt.gcf()


def create_interactive_confusion_matrix(cm, class_names=None, normalize=False, 
                                      title='Confusion Matrix'):
    """
    Create an interactive confusion matrix using Plotly for Streamlit.

    Parameters
    ----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of the classes
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    title : str, default='Confusion Matrix'
        Title for the plot

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object
    """
    import plotly.figure_factory as ff
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create the heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        annotation_text=[[fmt % val for val in row] for row in cm],
        colorscale='Blues',
        showscale=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted', side='bottom'),
        yaxis=dict(title='True', autorange='reversed'),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Add custom hover text
    for i in range(len(fig.data)):
        for j, txt in enumerate(fig.data[i].text):
            fig.data[i].text = None
            fig.data[i].hovertemplate = 'True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>'
    
    return fig


def plot_roc_curve(fpr, tpr, roc_auc, figsize=(8, 6), save_path=None, dpi=300):
    """
    Plot ROC curve.

    Parameters
    ----------
    fpr : array-like
        False positive rate
    tpr : array-like
        True positive rate
    roc_auc : float
        Area under ROC curve
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, optional
        Path to save the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return plt.gcf()


def plot_feature_importance(importance_dict, top_n=10, figsize=(10, 6), save_path=None, dpi=300):
    """
    Plot feature importance with enhanced styling.

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
    dpi : int, default=300
        Resolution for the saved image

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
    
    # Create figure with dark background
    plt.figure(figsize=figsize)
    plt.style.use('dark_background')
    
    # Create horizontal bar chart with gradient colors
    bars = plt.barh(range(len(features)), values, align='center', 
                   color=plt.cm.viridis(np.linspace(0, 0.8, len(features))))
    
    # Add value labels to the right of each bar
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9, color='white')
    
    # Customize plot
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12, labelpad=10)
    plt.ylabel('Feature', fontsize=12, labelpad=10)
    plt.title('Feature Importance', fontsize=14, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='#0E1117')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return plt.gcf()


def create_interactive_feature_importance(importance_dict, top_n=10, title='Feature Importance'):
    """
    Create an interactive feature importance plot using Plotly for Streamlit.

    Parameters
    ----------
    importance_dict : dict
        Dictionary of feature importances
    top_n : int, default=10
        Number of top features to show
    title : str, default='Feature Importance'
        Title for the plot

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object
    """
    import plotly.graph_objects as go
    
    # Sort by importance
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top_n features
    if top_n and top_n < len(sorted_imp):
        features = [item[0] for item in sorted_imp[:top_n]]
        values = [item[1] for item in sorted_imp[:top_n]]
    else:
        features = [item[0] for item in sorted_imp]
        values = [item[1] for item in sorted_imp]
    
    # Create color scale based on importance values
    colors = [f'rgba(59, 130, 246, {0.4 + 0.6 * val / max(values)})' for val in values]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(59, 130, 246, 1.0)', width=1)
        ),
        text=[f'{val:.4f}' for val in values],
        textposition='outside',
        hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Importance'),
        yaxis=dict(title='Feature', autorange='reversed'),
        height=400 + 15 * min(len(features), 20),  # Dynamic height based on feature count
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_interactive_roc_curve(fpr, tpr, roc_auc, title='Receiver Operating Characteristic'):
    """
    Create an interactive ROC curve using Plotly for Streamlit.

    Parameters
    ----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    roc_auc : float
        Area under ROC curve
    title : str, default='Receiver Operating Characteristic'
        Title for the plot

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object
    """
    import plotly.graph_objects as go
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2),
        hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>'
    ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='navy', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='False Positive Rate', range=[0, 1]),
        yaxis=dict(title='True Positive Rate', range=[0, 1.05]),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_feature_correlation_heatmap(X, feature_names=None, threshold=0.7):
    """
    Create an interactive correlation heatmap using Plotly for Streamlit.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature data
    feature_names : list, optional
        Names of the features
    threshold : float, default=0.7
        Threshold for highlighting high correlations

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object
    """
    import plotly.graph_objects as go
    
    # Convert to DataFrame if it's not already
    if not isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X)
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',  # Red-Blue diverging colorscale
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        hovertemplate='%{y} & %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Add annotations for high correlations
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            if abs(value) >= threshold and i != j:
                annotations.append(dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.columns[i],
                    text='!',
                    showarrow=False,
                    font=dict(color='white', size=12)
                ))
    
    fig.update_layout(annotations=annotations)
    
    return fig


def plot_model_comparison(model_names, accuracies, figsize=(10, 6), save_path=None, dpi=300):
    """
    Plot model comparison.

    Parameters
    ----------
    model_names : list
        Names of the models
    accuracies : list
        Accuracy scores for each model
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, accuracies, color='skyblue')
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(top=max(accuracies) * 1.1)  # Add some space for text
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return plt.gcf()


def plot_precision_recall_curve(precision, recall, average_precision, figsize=(8, 6),
                               save_path=None, dpi=300):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    precision : array-like
        Precision values
    recall : array-like
        Recall values
    average_precision : float
        Average precision score
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, optional
        Path to save the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    return plt.gcf()


def plot_learning_curve(train_sizes, train_scores, test_scores, figsize=(10, 6),
                       title='Learning Curve', save_path=None, dpi=300):
    """
    Plot a learning curve.

    Parameters
    ----------
    train_sizes : array-like
        Training set sizes
    train_scores : array-like
        Training scores
    test_scores : array-like
        Test scores
    figsize : tuple, default=(10, 6)
        Figure size
    title : str, default='Learning Curve'
        Plot title
    save_path : str, optional
        Path to save the visualization
    dpi : int, default=300
        Resolution for the saved image

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Learning curve saved to {save_path}")
    
    return plt.gcf()


if __name__ == "__main__":
    # Example usage
    import joblib
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    # Load a model
    try:
        # Try to load optimized model first, then baseline model
        model_paths = [
            'outputs/models/optimized_model.pkl',
            'outputs/models/baseline_model.pkl',
            'models/decision_tree_model.pkl'  # fallback for old path
        ]
        
        model = None
        for model_path in model_paths:
            try:
                model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
                break
            except FileNotFoundError:
                continue
        
        if model is None:
            raise FileNotFoundError("No model files found")
        
        # Example of visualizing the decision tree
        visualize_decision_tree(
            model,
            max_depth=3,
            save_path='decision_tree_depth3.png'
        )
        
        # Example of exporting as DOT file
        dot_file = export_tree_as_dot(
            model,
            max_depth=3
        )
        
        # Try to render the DOT file if graphviz is available
        render_dot_file(dot_file, output_format='png')
    
    except FileNotFoundError:
        logger.warning("Model file not found. Example visualization skipped.")
        
    # Example of creating a confusion matrix plot
    example_cm = np.array([[85, 5], [10, 50]])
    plot_confusion_matrix(
        example_cm,
        class_names=['No Churn', 'Churn'],
        save_path='example_confusion_matrix.png'
    ) 