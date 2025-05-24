#!/usr/bin/env python3
"""
Main script for training and evaluating the customer churn prediction model.

This script provides a complete pipeline for:
1. Data preprocessing
2. Baseline model training
3. Model evaluation and visualization
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from churn_prediction.src.preprocessing import preprocess_pipeline
from churn_prediction.src.model import (
    train_decision_tree, evaluate_model, 
    save_model, plot_roc_curve, plot_confusion_matrix
)
from churn_prediction.utils.visualization import visualize_decision_tree, export_tree_as_dot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the churn prediction pipeline."""
    parser = argparse.ArgumentParser(description='Train and evaluate churn prediction model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the dataset file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--visualize-tree', action='store_true',
                       help='Create decision tree visualizations')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum depth for tree visualization')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    logger.info("Starting churn prediction pipeline")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Data preprocessing
    logger.info("Step 1: Data preprocessing")
    try:
        data = preprocess_pipeline(
            args.data, 
            random_state=args.random_state
        )
        logger.info("Data preprocessing completed successfully")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Original shape: {data['df_original'].shape}")
        print(f"After cleaning: {data['df_clean'].shape}")
        print(f"Training samples: {data['X_train'].shape[0]}")
        print(f"Test samples: {data['X_test'].shape[0]}")
        print(f"Features after preprocessing: {len(data['feature_names'])}")
        print(f"Resampled training samples: {data['X_train_resampled'].shape[0]}")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return 1
    
    # Step 2: Train baseline model
    logger.info("Step 2: Training baseline model")
    try:
        baseline_model = train_decision_tree(
            data['X_train_resampled'],
            data['y_train_resampled'],
            random_state=args.random_state
        )
        
        # Evaluate baseline model
        baseline_eval = evaluate_model(
            baseline_model,
            data['X_test_processed'],
            data['y_test'],
            feature_names=data['feature_names']
        )
        
        print(f"\nBaseline Model Performance:")
        print(f"Accuracy: {baseline_eval['accuracy']:.4f}")
        if baseline_eval['binary_metrics']:
            print(f"AUC: {baseline_eval['binary_metrics']['auc']:.4f}")
            print(f"Precision: {baseline_eval['binary_metrics']['precision']:.4f}")
            print(f"Recall: {baseline_eval['binary_metrics']['recall']:.4f}")
            print(f"F1-Score: {baseline_eval['binary_metrics']['f1']:.4f}")
        
        # Save baseline model
        baseline_model_path = output_dir / 'models' / 'baseline_model.pkl'
        save_model(baseline_model, str(baseline_model_path))
        
        # Plot ROC curve for baseline
        if baseline_eval['roc_data']:
            roc_path = output_dir / 'plots' / 'baseline_roc_curve.png'
            plot_roc_curve(
                baseline_eval['roc_data']['fpr'],
                baseline_eval['roc_data']['tpr'],
                baseline_eval['roc_data']['auc'],
                title='Baseline Model ROC Curve',
                save_path=str(roc_path)
            )
        
        # Plot confusion matrix
        cm_path = output_dir / 'plots' / 'baseline_confusion_matrix.png'
        plot_confusion_matrix(
            baseline_eval['confusion_matrix'],
            class_names=['No Churn', 'Churn'],
            save_path=str(cm_path)
        )
        
    except Exception as e:
        logger.error(f"Error in baseline model training: {e}")
        return 1
    
    # Step 3: Tree visualization (if requested)
    if args.visualize_tree:
        logger.info("Step 3: Decision tree visualization")
        try:
            # Visualize the baseline model
            tree_viz_path = output_dir / 'plots' / f'decision_tree_depth_{args.max_depth}.png'
            visualize_decision_tree(
                baseline_model,
                feature_names=data['feature_names'],
                class_names=['No Churn', 'Churn'],
                max_depth=args.max_depth,
                save_path=str(tree_viz_path)
            )
            
            # Export as DOT file
            dot_path = output_dir / 'results' / f'decision_tree_depth_{args.max_depth}.dot'
            export_tree_as_dot(
                baseline_model,
                feature_names=data['feature_names'],
                class_names=['No Churn', 'Churn'],
                max_depth=args.max_depth,
                output_file=str(dot_path)
            )
            
        except Exception as e:
            logger.error(f"Error in tree visualization: {e}")
            return 1
    
    logger.info("Pipeline completed successfully!")
    print(f"\nAll outputs saved to: {output_dir}")
    print("Pipeline completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 