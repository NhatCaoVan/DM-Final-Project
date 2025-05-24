"""
Visualization utilities for the churn prediction model.

This module provides functions for visualizing decision trees and model results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_graphviz
import logging

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