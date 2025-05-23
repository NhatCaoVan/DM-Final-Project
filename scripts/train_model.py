#!/usr/bin/env python3
"""
Main script for training and evaluating the customer churn prediction model.

This script provides a complete pipeline for:
1. Data preprocessing
2. Baseline model training
3. Optimized model training (with hyperparameter tuning and feature selection)
4. Model evaluation and visualization
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
    train_decision_tree, evaluate_model, tune_hyperparameters, 
    save_model, plot_roc_curve, plot_confusion_matrix
)
from churn_prediction.src.feature_selection import (
    get_feature_importance, plot_feature_importance, 
    plot_cumulative_importance, select_features_by_cumulative_importance,
    save_selected_features, compare_feature_selection_methods,
    plot_feature_selection_comparison
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
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning for optimized model')
    parser.add_argument('--feature-selection', action='store_true',
                       help='Perform feature selection analysis')
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
    
    # Step 3: Feature importance analysis
    logger.info("Step 3: Feature importance analysis")
    try:
        importance_results = get_feature_importance(
            baseline_model,
            data['X_test_processed'],
            data['y_test'],
            data['feature_names'],
            method='both'
        )
        
        # Plot feature importance
        importance_path = output_dir / 'plots' / 'feature_importance.png'
        plot_feature_importance(
            importance_results['model_importance'],
            top_n=15,
            save_path=str(importance_path)
        )
        
        # Plot cumulative importance
        cumulative_path = output_dir / 'plots' / 'cumulative_importance.png'
        fig, n_features_95 = plot_cumulative_importance(
            importance_results['model_importance'],
            save_path=str(cumulative_path)
        )
        
        print(f"\nFeature Importance Analysis:")
        print(f"Top 5 most important features:")
        for i, (feature, importance) in enumerate(importance_results['sorted_model_importance'][:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        print(f"Features needed for 95% importance: {n_features_95}")
        
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        return 1
    
    # Step 4: Optimized model (combines hyperparameter tuning and feature selection)
    optimized_model = baseline_model  # Default to baseline if optimization not requested
    
    if args.tune_hyperparams or args.feature_selection:
        logger.info("Step 4: Training optimized model")
        
        # Start with the original features and data
        X_train_opt = data['X_train_resampled']
        X_test_opt = data['X_test_processed']
        feature_names_opt = data['feature_names']
        
        # Feature selection (if requested)
        if args.feature_selection:
            try:
                logger.info("Performing feature selection...")
                
                # Select features by cumulative importance
                selected_features = select_features_by_cumulative_importance(
                    importance_results['model_importance'],
                    threshold=0.95
                )
                
                # Save selected features
                features_path = output_dir / 'results' / 'selected_features.txt'
                save_selected_features(
                    selected_features,
                    str(features_path),
                    method_name="Top 95% Important Features"
                )
                logger.info(f"Selected features saved to {features_path}")
                
                # Compare feature selection methods
                comparison_results = compare_feature_selection_methods(
                    data['X_train_processed'],
                    data['y_train'],
                    data['feature_names'],
                    base_model=baseline_model,
                    random_state=args.random_state
                )
                
                # Save comparison results to CSV
                csv_path = output_dir / 'results' / 'feature_selection_results.csv'
                pd.DataFrame([
                    {
                        'Method': method,
                        'Features': results['n_features'],
                        'Accuracy': results['accuracy'],
                        'Reduction': results.get('feature_reduction', 0)
                    }
                    for method, results in comparison_results.items()
                ]).to_csv(str(csv_path), index=False)
                logger.info(f"Feature selection results saved to {csv_path}")
                
                # Plot comparison
                comparison_path = output_dir / 'plots' / 'feature_selection_comparison.png'
                plot_feature_selection_comparison(
                    comparison_results,
                    save_path=str(comparison_path)
                )
                
                # Update training data with selected features
                feature_indices = [i for i, name in enumerate(data['feature_names']) 
                                 if name in selected_features]
                X_train_opt = data['X_train_resampled'][:, feature_indices]
                X_test_opt = data['X_test_processed'][:, feature_indices]
                feature_names_opt = selected_features
                
                print(f"\nFeature Selection Results:")
                print(f"Selected features: {len(selected_features)}/{len(data['feature_names'])}")
                print(f"Feature reduction: {(1 - len(selected_features)/len(data['feature_names'])):.1%}")
                
            except Exception as e:
                logger.error(f"Error in feature selection: {e}")
                return 1
        
        # Hyperparameter tuning (if requested)
        if args.tune_hyperparams:
            try:
                logger.info("Performing hyperparameter tuning...")
                
                tuning_results = tune_hyperparameters(
                    X_train_opt,
                    data['y_train_resampled'],
                    X_test_opt,
                    data['y_test'],
                    random_state=args.random_state
                )
                
                optimized_model = tuning_results['best_model']
                
                print(f"\nHyperparameter Tuning Results:")
                print(f"Best parameters: {tuning_results['best_params']}")
                print(f"Best CV score: {tuning_results['best_score']:.4f}")
                print(f"Test accuracy: {tuning_results['test_evaluation']['accuracy']:.4f}")
                
                # Plot ROC curve for optimized model
                if tuning_results['test_evaluation']['roc_data']:
                    opt_roc_path = output_dir / 'plots' / 'optimized_roc_curve.png'
                    plot_roc_curve(
                        tuning_results['test_evaluation']['roc_data']['fpr'],
                        tuning_results['test_evaluation']['roc_data']['tpr'],
                        tuning_results['test_evaluation']['roc_data']['auc'],
                        title='Optimized Model ROC Curve',
                        save_path=str(opt_roc_path)
                    )
                
            except Exception as e:
                logger.error(f"Error in hyperparameter tuning: {e}")
                return 1
        else:
            # If only feature selection, train simple model with selected features
            optimized_model = train_decision_tree(
                X_train_opt,
                data['y_train_resampled'],
                random_state=args.random_state
            )
            
            # Evaluate optimized model
            opt_eval = evaluate_model(
                optimized_model,
                X_test_opt,
                data['y_test'],
                feature_names=feature_names_opt
            )
            
            print(f"Optimized model accuracy: {opt_eval['accuracy']:.4f}")
        
        # Save optimized model
        optimized_model_path = output_dir / 'models' / 'optimized_model.pkl'
        save_model(optimized_model, str(optimized_model_path))
    
    # Step 5: Tree visualization (if requested)
    if args.visualize_tree:
        logger.info("Step 5: Decision tree visualization")
        try:
            # Use the appropriate feature names based on whether feature selection was performed
            viz_feature_names = feature_names_opt if args.feature_selection else data['feature_names']
            
            # Visualize the optimized model (or baseline if no optimization)
            tree_viz_path = output_dir / 'plots' / f'decision_tree_depth_{args.max_depth}.png'
            visualize_decision_tree(
                optimized_model,
                feature_names=viz_feature_names,
                class_names=['No Churn', 'Churn'],
                max_depth=args.max_depth,
                save_path=str(tree_viz_path)
            )
            
            # Export as DOT file
            dot_path = output_dir / 'results' / f'decision_tree_depth_{args.max_depth}.dot'
            export_tree_as_dot(
                optimized_model,
                feature_names=viz_feature_names,
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