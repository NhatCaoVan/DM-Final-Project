#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script performs complete evaluation of the customer churn prediction model including:
- Basic performance metrics (accuracy, precision, recall, F1)
- Cross-validation analysis
- ROC curve and AUC analysis
- Confusion matrix analysis
- Feature importance analysis
- Decision tree visualization and analysis
- Business impact assessment
- Report generation

Usage:
    python evaluate_model.py [options]

Options:
    --model-path: Path to the model file (default: outputs/models/baseline_model.pkl)
    --data-path: Path to the dataset (default: datasets/Dataset_Cay quyet dinh_HV.xlsx)
    --output-dir: Output directory for reports (default: outputs/evaluation)
    --max-tree-depth: Maximum depth for decision tree visualization (default: None - show full tree)
    --limit-viz-depth: Limit visualization depth to max-tree-depth (default: show full tree)
    --generate-reports: Generate markdown reports (default: True)
    --visualize-tree: Generate decision tree visualization (default: True)

Examples:
    # Basic evaluation - shows FULL tree depth (recommended)
    python evaluate_model.py
    
    # Show full tree but also limit one visualization to depth 4
    python evaluate_model.py --max-tree-depth 4 --limit-viz-depth
    
    # Force visualization to show only depth 2 (for overview)
    python evaluate_model.py --max-tree-depth 2 --limit-viz-depth
    
    # Skip tree visualization entirely
    python evaluate_model.py --no-tree-viz
    
    Note: By default, the script shows the FULL tree depth of your trained model.
    Use --limit-viz-depth only if you want to artificially limit the visualization depth.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import joblib
import json
from datetime import datetime
import logging

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

# Set matplotlib backend for non-interactive use
try:
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available, plots will be skipped")
    MATPLOTLIB_AVAILABLE = False


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path, data_path, output_dir="outputs/evaluation", max_tree_depth=None, limit_viz_depth=False):
        """
        Initialize the model evaluator.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model
        data_path : str
            Path to the dataset
        output_dir : str
            Directory to save evaluation outputs
        max_tree_depth : int or None
            Maximum depth for decision tree visualization. If None, shows full tree depth.
            If limit_viz_depth=False, this parameter is ignored.
        limit_viz_depth : bool
            Whether to limit the visualization depth. If False, shows the full model tree.
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_tree_depth = max_tree_depth
        self.limit_viz_depth = limit_viz_depth
        
        # Initialize storage for results
        self.model = None
        self.data = None
        self.results = {}
        
    def load_data_and_model(self):
        """Load the dataset and trained model."""
        logger.info("Loading data and model...")
        
        # Load preprocessed data
        self.data = preprocess_pipeline(str(self.data_path), id_col='id')
        
        # Load trained model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        logger.info(f"✅ Data loaded: {self.data['X_test'].shape[0]} test samples")
        logger.info(f"✅ Model loaded: {type(self.model).__name__}")
        logger.info(f"✅ Features: {len(self.data['feature_names'])}")
        
    def calculate_basic_metrics(self):
        """Calculate basic performance metrics."""
        logger.info("Calculating basic performance metrics...")
        
        # Make predictions
        y_pred = self.model.predict(self.data['X_test_processed'])
        y_pred_proba = self.model.predict_proba(self.data['X_test_processed'])[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.data['y_test'], y_pred)
        precision = precision_score(self.data['y_test'], y_pred)
        recall = recall_score(self.data['y_test'], y_pred)
        f1 = f1_score(self.data['y_test'], y_pred)
        
        # ROC analysis
        fpr, tpr, roc_thresholds = roc_curve(self.data['y_test'], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-recall analysis
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            self.data['y_test'], y_pred_proba
        )
        avg_precision = average_precision_score(self.data['y_test'], y_pred_proba)
        
        self.results['basic_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {
                'precision': precision_curve, 
                'recall': recall_curve, 
                'thresholds': pr_thresholds
            }
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
    def analyze_confusion_matrix(self):
        """Analyze confusion matrix and related metrics."""
        logger.info("Analyzing confusion matrix...")
        
        y_pred = self.results['basic_metrics']['predictions']
        cm = confusion_matrix(self.data['y_test'], y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        
        self.results['confusion_matrix'] = {
            'matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        logger.info(f"Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
    def perform_cross_validation(self):
        """Perform cross-validation analysis."""
        logger.info("Performing cross-validation...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = cross_validate(
            self.model, 
            self.data['X_train_processed'], 
            self.data['y_train'], 
            cv=cv, 
            scoring=cv_metrics,
            return_train_score=True
        )
        
        # Process results
        cv_summary = {}
        for metric in cv_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            cv_summary[metric] = {
                'test_mean': float(test_scores.mean()),
                'test_std': float(test_scores.std()),
                'train_mean': float(train_scores.mean()),
                'train_std': float(train_scores.std()),
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist()
            }
        
        self.results['cross_validation'] = cv_summary
        
        logger.info("Cross-validation completed")
        
    def analyze_feature_importance(self):
        """Analyze feature importance."""
        logger.info("Analyzing feature importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.data['feature_names']
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Calculate percentages
            importance_df['importance_pct'] = (
                importance_df['importance'] / importance_df['importance'].sum() * 100
            )
            
            # Feature utilization summary
            zero_importance = (importance_df['importance'] == 0).sum()
            used_features = len(importance_df) - zero_importance
            
            self.results['feature_importance'] = {
                'importance_data': importance_df.to_dict('records'),
                'total_features': len(importance_df),
                'used_features': int(used_features),
                'unused_features': int(zero_importance),
                'top_5_cumulative_importance': float(
                    importance_df.head(5)['importance_pct'].sum()
                )
            }
            
            logger.info(f"Feature analysis: {used_features}/{len(importance_df)} features used")
        else:
            logger.warning("Model does not have feature importance information")
            self.results['feature_importance'] = None

    def analyze_decision_tree(self):
        """Analyze decision tree structure and create visualizations."""
        logger.info("Analyzing decision tree structure...")
        
        if not isinstance(self.model, DecisionTreeClassifier):
            logger.warning("Model is not a DecisionTreeClassifier, skipping tree analysis")
            self.results['decision_tree'] = None
            return
        
        tree = self.model.tree_
        feature_names = self.data['feature_names']
        class_names = ['No Churn', 'Churn']
        
        # Basic tree statistics
        tree_stats = {
            'max_depth': int(self.model.get_depth()),
            'n_nodes': int(tree.node_count),
            'n_leaves': int(tree.n_leaves),
            'n_features_used': int(np.sum(self.model.feature_importances_ > 0)),
            'total_features': len(feature_names),
            'configured_max_depth': self.max_tree_depth,
            'depth_limiting_enabled': self.limit_viz_depth
        }
        
        # Analyze tree structure
        tree_analysis = self.analyze_tree_structure(tree, feature_names, class_names)
        
        # Export tree rules to text
        tree_rules = self.export_tree_rules(feature_names, class_names)
        
        # Create tree visualization
        tree_visualizations = None
        if MATPLOTLIB_AVAILABLE:
            tree_visualizations = self.visualize_decision_tree(feature_names, class_names)
        
        self.results['decision_tree'] = {
            'statistics': tree_stats,
            'analysis': tree_analysis,
            'rules': tree_rules,
            'visualizations': tree_visualizations
        }
        
        viz_info = ""
        if tree_visualizations:
            actual_depth = tree_visualizations.get('actual_model_depth', tree_stats['max_depth'])
            viz_depth = tree_visualizations.get('visualization_depth', actual_depth)
            depth_limited = tree_visualizations.get('depth_limited', False)
            
            if depth_limited:
                viz_info = f"(showing depth {viz_depth} of {actual_depth})"
            else:
                viz_info = f"(full tree depth {actual_depth})"
        
        logger.info(f"Tree analysis: {tree_stats['n_nodes']} nodes, "
                   f"{tree_stats['n_leaves']} leaves, model depth {tree_stats['max_depth']} {viz_info}")

    def analyze_tree_structure(self, tree, feature_names, class_names):
        """Analyze the decision tree structure in detail."""
        
        def recurse(node_id, depth=0, parent_info="Root"):
            # Get node information
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            value = tree.value[node_id][0]
            n_samples = tree.n_node_samples[node_id]
            
            # Calculate class probabilities
            total_samples = value.sum()
            class_probs = value / total_samples if total_samples > 0 else [0, 0]
            predicted_class = np.argmax(value)
            
            node_info = {
                'node_id': int(node_id),
                'depth': depth,
                'parent_info': parent_info,
                'n_samples': int(n_samples),
                'class_distribution': value.tolist(),
                'class_probabilities': class_probs.tolist(),
                'predicted_class': class_names[predicted_class],
                'is_leaf': left_child == right_child  # Both are -1 for leaves
            }
            
            if not node_info['is_leaf']:
                # Internal node
                feature_name = feature_names[feature]
                node_info.update({
                    'feature': feature_name,
                    'feature_index': int(feature),
                    'threshold': float(threshold),
                    'split_rule': f"{feature_name} <= {threshold:.3f}"
                })
                
                # Recursively analyze children
                left_info = recurse(left_child, depth + 1, f"{feature_name} <= {threshold:.3f}")
                right_info = recurse(right_child, depth + 1, f"{feature_name} > {threshold:.3f}")
                
                node_info['left_child'] = left_info
                node_info['right_child'] = right_info
            
            return node_info
        
        # Start analysis from root
        tree_structure = recurse(0)
        
        # Calculate additional statistics
        def count_nodes_by_depth(node, depth_counts=None):
            if depth_counts is None:
                depth_counts = {}
            
            depth = node['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            if not node['is_leaf']:
                count_nodes_by_depth(node['left_child'], depth_counts)
                count_nodes_by_depth(node['right_child'], depth_counts)
            
            return depth_counts
        
        def find_important_paths(node, path=[], important_paths=[]):
            current_path = path + [node]
            
            if node['is_leaf']:
                # Calculate path importance (samples in leaf / total samples)
                path_importance = node['n_samples'] / tree.n_node_samples[0]
                if path_importance > 0.05:  # Paths with >5% of samples
                    important_paths.append({
                        'path': [n['split_rule'] if 'split_rule' in n else n['predicted_class'] 
                                for n in current_path],
                        'samples': node['n_samples'],
                        'importance': path_importance,
                        'prediction': node['predicted_class'],
                        'confidence': max(node['class_probabilities'])
                    })
            else:
                find_important_paths(node['left_child'], current_path, important_paths)
                find_important_paths(node['right_child'], current_path, important_paths)
            
            return important_paths
        
        depth_distribution = count_nodes_by_depth(tree_structure)
        important_paths = find_important_paths(tree_structure)
        
        return {
            'tree_structure': tree_structure,
            'depth_distribution': depth_distribution,
            'important_paths': important_paths,
            'max_depth': max(depth_distribution.keys()) if depth_distribution else 0,
            'total_nodes': sum(depth_distribution.values()) if depth_distribution else 0
        }

    def export_tree_rules(self, feature_names, class_names):
        """Export decision tree rules in human-readable format."""
        
        # Use sklearn's export_text for basic rules
        basic_rules = export_text(
            self.model, 
            feature_names=feature_names,
            class_names=class_names,
            show_weights=True
        )
        
        # Create detailed rules explanation
        detailed_rules = self.create_detailed_rules_explanation()
        
        # Save rules to file
        rules_file = self.output_dir / "decision_tree_rules.txt"
        with open(rules_file, 'w', encoding='utf-8') as f:
            f.write("DECISION TREE RULES ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. BASIC TREE STRUCTURE\n")
            f.write("-" * 30 + "\n")
            f.write(basic_rules)
            f.write("\n\n")
            
            f.write("2. DETAILED RULES EXPLANATION\n")
            f.write("-" * 30 + "\n")
            f.write(detailed_rules)
            
        logger.info(f"Tree rules saved to: {rules_file}")
        
        return {
            'basic_rules': basic_rules,
            'detailed_rules': detailed_rules,
            'rules_file': str(rules_file)
        }

    def create_detailed_rules_explanation(self):
        """Create detailed explanation of tree rules."""
        
        if 'decision_tree' not in self.results or not self.results['decision_tree']:
            return "Decision tree analysis not available."
        
        tree_analysis = self.results['decision_tree']['analysis']
        important_paths = tree_analysis['important_paths']
        
        explanation = []
        explanation.append("DECISION TREE INTERPRETATION")
        explanation.append("=" * 40)
        explanation.append("")
        
        # Tree overview
        stats = self.results['decision_tree']['statistics']
        explanation.append(f"Tree Overview:")
        explanation.append(f"- Maximum Depth: {stats['max_depth']} levels")
        explanation.append(f"- Total Nodes: {stats['n_nodes']}")
        explanation.append(f"- Leaf Nodes: {stats['n_leaves']}")
        explanation.append(f"- Features Used: {stats['n_features_used']}/{stats['total_features']}")
        explanation.append("")
        
        # Important decision paths
        explanation.append("IMPORTANT DECISION PATHS:")
        explanation.append("-" * 30)
        
        for i, path in enumerate(sorted(important_paths, key=lambda x: x['importance'], reverse=True), 1):
            explanation.append(f"\nPath {i}: {path['prediction']} "
                             f"({path['samples']} samples, {path['importance']:.1%} of data)")
            explanation.append(f"Confidence: {path['confidence']:.1%}")
            explanation.append("Rules:")
            for rule in path['path'][:-1]:  # Exclude the final prediction
                explanation.append(f"  -> {rule}")
            explanation.append(f"  => Prediction: {path['path'][-1]}")
        
        # Feature usage analysis
        if self.results['feature_importance']:
            explanation.append("\n\nFEATURE USAGE IN TREE:")
            explanation.append("-" * 30)
            
            important_features = [f for f in self.results['feature_importance']['importance_data'] 
                                if f['importance'] > 0][:10]
            
            for feature in important_features:
                explanation.append(f"- {feature['feature']}: "
                                 f"{feature['importance']:.4f} importance "
                                 f"({feature['importance_pct']:.1f}%)")
        
        # Business insights
        explanation.append("\n\nBUSINESS INSIGHTS:")
        explanation.append("-" * 30)
        explanation.append(self.generate_business_insights())
        
        return "\n".join(explanation)

    def generate_business_insights(self):
        """Generate business insights from the decision tree."""
        
        insights = []
        
        if 'decision_tree' not in self.results or not self.results['decision_tree']:
            return "Decision tree analysis not available for business insights."
        
        important_paths = self.results['decision_tree']['analysis']['important_paths']
        
        # Analyze key decision factors
        churn_paths = [p for p in important_paths if p['prediction'] == 'Churn']
        no_churn_paths = [p for p in important_paths if p['prediction'] == 'No Churn']
        
        insights.append("Key Factors Leading to Churn:")
        for path in churn_paths[:3]:  # Top 3 churn paths
            key_rules = [rule for rule in path['path'][:-1] if any(
                keyword in rule.lower() for keyword in ['age', 'gender', 'package', 'spending']
            )]
            if key_rules:
                insights.append(f"- {path['importance']:.1%} of customers churn when: {', '.join(key_rules[:2])}")
        
        insights.append("\nKey Factors for Retention:")
        for path in no_churn_paths[:3]:  # Top 3 retention paths
            key_rules = [rule for rule in path['path'][:-1] if any(
                keyword in rule.lower() for keyword in ['age', 'gender', 'package', 'spending']
            )]
            if key_rules:
                insights.append(f"- {path['importance']:.1%} of customers stay when: {', '.join(key_rules[:2])}")
        
        # Risk segments
        insights.append(f"\nHigh-Risk Segments:")
        high_risk_paths = [p for p in churn_paths if p['confidence'] > 0.7]
        for path in high_risk_paths:
            insights.append(f"- {path['importance']:.1%} of customers (very high risk)")
        
        return "\n".join(insights)

    def visualize_decision_tree(self, feature_names, class_names):
        """Create and save decision tree visualization with configurable depth."""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping tree visualization")
            return None
        
        # Get the actual model depth
        actual_model_depth = self.model.get_depth()
        
        # Determine visualization depth
        if self.limit_viz_depth and self.max_tree_depth is not None:
            viz_depth = min(self.max_tree_depth, actual_model_depth)
            depth_limited = True
        else:
            viz_depth = None  # Show full tree
            depth_limited = False
        
        # Create main tree visualization
        fig, ax = plt.subplots(1, 1, figsize=(24, 16))
        
        plot_tree(
            self.model,
            max_depth=viz_depth,  # None means show full depth
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            impurity=True,
            proportion=True,
            rounded=True,
            precision=3,
            fontsize=max(6, 12 - (viz_depth or actual_model_depth)),  # Adjust font size based on depth
            ax=ax
        )
        
        # Create appropriate title
        if depth_limited:
            title = f"Customer Churn Prediction - Decision Tree (Showing Depth 1-{viz_depth} of {actual_model_depth})"
        else:
            title = f"Customer Churn Prediction - Decision Tree (Full Tree - Depth {actual_model_depth})"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Save the main plot
        if depth_limited:
            plot_path = self.output_dir / f"decision_tree_limited_depth_{viz_depth}.png"
        else:
            plot_path = self.output_dir / f"decision_tree_full_depth_{actual_model_depth}.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Create additional visualizations for comparison
        visualization_paths = [plot_path]
        depths_generated = [viz_depth or actual_model_depth]
        
        # Always create a simplified overview (depth 2) if the tree is deeper
        if actual_model_depth > 2:
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            plot_tree(
                self.model,
                max_depth=2,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                impurity=False,
                proportion=True,
                rounded=True,
                precision=2,
                fontsize=12,
                ax=ax
            )
            
            ax.set_title(f"Customer Churn Prediction - Overview (Depth 1-2 of {actual_model_depth})", 
                        fontsize=16, fontweight='bold', pad=20)
            
            overview_plot_path = self.output_dir / "decision_tree_overview_depth_2.png"
            plt.savefig(overview_plot_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            visualization_paths.append(overview_plot_path)
            depths_generated.append(2)
        
        # Create a mid-level view if the tree is very deep (depth 4)
        if actual_model_depth > 4:
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            
            plot_tree(
                self.model,
                max_depth=4,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                impurity=True,
                proportion=True,
                rounded=True,
                precision=2,
                fontsize=10,
                ax=ax
            )
            
            ax.set_title(f"Customer Churn Prediction - Mid-Level View (Depth 1-4 of {actual_model_depth})", 
                        fontsize=16, fontweight='bold', pad=20)
            
            mid_plot_path = self.output_dir / "decision_tree_mid_depth_4.png"
            plt.savefig(mid_plot_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            visualization_paths.append(mid_plot_path)
            depths_generated.append(4)
        
        logger.info(f"Tree visualizations saved to: {', '.join(str(p) for p in visualization_paths)}")
        
        return {
            'main_visualization': str(plot_path),
            'all_visualizations': [str(p) for p in visualization_paths],
            'depths_generated': depths_generated,
            'actual_model_depth': actual_model_depth,
            'visualization_depth': viz_depth or actual_model_depth,
            'depth_limited': depth_limited
        }

    def calculate_business_impact(self):
        """Calculate business impact metrics."""
        logger.info("Calculating business impact...")
        
        cm = self.results['confusion_matrix']
        tp = cm['true_positives']
        tn = cm['true_negatives']
        fp = cm['false_positives']
        fn = cm['false_negatives']
        
        total_customers = len(self.data['y_test'])
        actual_churners = tp + fn
        predicted_churners = tp + fp
        
        # Business metrics
        business_metrics = {
            'total_customers': total_customers,
            'actual_churners': actual_churners,
            'predicted_churners': predicted_churners,
            'correctly_identified': tp,
            'missed_churners': fn,
            'false_alarms': fp,
            'churn_rate': actual_churners / total_customers,
            'identification_rate': tp / actual_churners if actual_churners > 0 else 0,
            'false_alarm_rate': fp / predicted_churners if predicted_churners > 0 else 0
        }
        
        # Financial impact estimation (with hypothetical values)
        cost_per_lost_customer = 500  # $500
        cost_per_retention_campaign = 50  # $50
        
        value_saved = tp * cost_per_lost_customer
        campaign_cost = predicted_churners * cost_per_retention_campaign
        lost_value = fn * cost_per_lost_customer
        net_value = value_saved - campaign_cost - lost_value
        
        business_metrics['financial_impact'] = {
            'cost_per_lost_customer': cost_per_lost_customer,
            'cost_per_retention_campaign': cost_per_retention_campaign,
            'value_saved': value_saved,
            'campaign_cost': campaign_cost,
            'lost_value': lost_value,
            'net_value': net_value,
            'roi': (net_value / campaign_cost) if campaign_cost > 0 else 0
        }
        
        self.results['business_impact'] = business_metrics
        
        logger.info(f"Business impact: Net value ${net_value:,.0f}")
        
    def generate_performance_rating(self):
        """Generate overall performance rating."""
        logger.info("Generating performance rating...")
        
        auc = self.results['basic_metrics']['roc_auc']
        accuracy = self.results['basic_metrics']['accuracy']
        
        # AUC rating
        if auc >= 0.9:
            auc_rating = "Excellent"
        elif auc >= 0.8:
            auc_rating = "Good"
        elif auc >= 0.7:
            auc_rating = "Fair"
        else:
            auc_rating = "Poor"
            
        # Accuracy rating
        if accuracy >= 0.9:
            acc_rating = "Excellent"
        elif accuracy >= 0.8:
            acc_rating = "Good"
        elif accuracy >= 0.7:
            acc_rating = "Fair"
        else:
            acc_rating = "Poor"
            
        # Overall grade calculation
        auc_score = min(5, max(1, int((auc - 0.5) * 10)))
        acc_score = min(5, max(1, int(accuracy * 5)))
        overall_score = (auc_score + acc_score) / 2
        
        if overall_score >= 4.5:
            overall_grade = "A"
        elif overall_score >= 3.5:
            overall_grade = "B+"
        elif overall_score >= 2.5:
            overall_grade = "B"
        elif overall_score >= 1.5:
            overall_grade = "C"
        else:
            overall_grade = "D"
            
        self.results['performance_rating'] = {
            'auc_rating': auc_rating,
            'accuracy_rating': acc_rating,
            'overall_grade': overall_grade,
            'overall_score': overall_score,
            'auc_score': auc_score,
            'accuracy_score': acc_score
        }
        
        logger.info(f"Performance rating: {overall_grade} ({auc_rating} AUC, {acc_rating} Accuracy)")
        
    def save_results(self):
        """Save evaluation results to JSON file."""
        logger.info("Saving evaluation results...")
        
        def convert_for_json(obj):
            """Convert objects to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert all results to JSON-serializable format
        json_results = convert_for_json(self.results)
                
        # Add metadata
        json_results['metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'data_path': str(self.data_path),
            'model_type': type(self.model).__name__,
            'dataset_shape': self.data['df_original'].shape,
            'test_samples': len(self.data['y_test']),
            'training_samples': len(self.data['y_train'])
        }
        
        # Save to file
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Results saved to: {results_file}")
        
    def generate_summary_report(self):
        """Generate a summary markdown report."""
        logger.info("Generating summary report...")
        
        metrics = self.results['basic_metrics']
        cm = self.results['confusion_matrix']
        business = self.results['business_impact']
        rating = self.results['performance_rating']
        
        # Add decision tree information if available
        tree_info = ""
        if 'decision_tree' in self.results and self.results['decision_tree']:
            tree_stats = self.results['decision_tree']['statistics']
            visualizations = self.results['decision_tree']['visualizations']
            
            viz_info = ""
            if visualizations:
                viz_files = visualizations.get('all_visualizations', [])
                depths = [d for d in visualizations.get('depths_generated', []) if d is not None]
                viz_info = f"- **Visualizations Generated**: {len(viz_files)} files at depths {depths}"
            
            tree_info = f"""

## Decision Tree Analysis
- **Actual Tree Depth**: {tree_stats['max_depth']} levels
- **Configured Max Depth**: {tree_stats['configured_max_depth']} levels
- **Total Nodes**: {tree_stats['n_nodes']}
- **Leaf Nodes**: {tree_stats['n_leaves']}
- **Features Used**: {tree_stats['n_features_used']}/{tree_stats['total_features']}
{viz_info}
- **Decision Rules**: Exported to text file
"""
        
        report = f"""# Model Evaluation Summary

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Quick Performance Overview

| Metric | Value | Rating |
|--------|--------|--------|
| **Accuracy** | {metrics['accuracy']:.2%} | {rating['accuracy_rating']} |
| **ROC AUC** | {metrics['roc_auc']:.3f} | {rating['auc_rating']} |
| **Precision** | {metrics['precision']:.2%} | - |
| **Recall** | {metrics['recall']:.2%} | - |
| **F1-Score** | {metrics['f1_score']:.3f} | - |

**Overall Grade: {rating['overall_grade']}**

## Confusion Matrix
```
                 Predicted
                No    Yes
Actual    No   {cm['true_negatives']:4d} {cm['false_positives']:4d}
          Yes   {cm['false_negatives']:4d} {cm['true_positives']:4d}
```
{tree_info}

## Business Impact
- **Total Customers Evaluated**: {business['total_customers']:,}
- **Actual Churners**: {business['actual_churners']} ({business['churn_rate']:.1%})
- **Successfully Identified**: {business['correctly_identified']} ({business['identification_rate']:.1%} of actual)
- **Missed Churners**: {business['missed_churners']}
- **False Alarms**: {business['false_alarms']}

### Financial Impact
- **Net Value**: ${business['financial_impact']['net_value']:,.0f}
- **ROI**: {business['financial_impact']['roi']:.1%}

## Model Assessment

### Strengths
- {"[+] High Recall" if metrics['recall'] > 0.75 else "[~] Moderate Recall"} ({metrics['recall']:.1%})
- {"[+] Good AUC" if metrics['roc_auc'] > 0.8 else "[~] Fair AUC"} ({metrics['roc_auc']:.3f})
- {"[+] Good Accuracy" if metrics['accuracy'] > 0.8 else "[~] Moderate Accuracy"} ({metrics['accuracy']:.1%})

### Areas for Improvement
- {"[-] Low Precision" if metrics['precision'] < 0.6 else "[+] Good Precision"} ({metrics['precision']:.1%})
- False Positive Rate: {cm['false_positives']/(cm['false_positives']+cm['true_negatives']):.1%}

## Files Generated
- evaluation_results.json - Complete evaluation data
- decision_tree_visualization.png - Tree visualization
- decision_tree_simplified.png - Simplified tree view
- decision_tree_rules.txt - Detailed tree rules and analysis

## Deployment Recommendation
"""
        
        if rating['overall_score'] >= 3.5:
            report += "[+] **APPROVED** for production deployment"
        elif rating['overall_score'] >= 2.5:
            report += "[~] **CONDITIONAL** approval - monitor closely"
        else:
            report += "[-] **NOT RECOMMENDED** for production - needs improvement"
            
        report += f"""

---
*Evaluation completed using {len(self.data['y_test'])} test samples*
"""
        
        # Save report with UTF-8 encoding
        report_file = self.output_dir / "evaluation_summary.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"Summary report saved to: {report_file}")
        
    def run_complete_evaluation(self, generate_reports=True, visualize_tree=True):
        """Run the complete evaluation pipeline."""
        print("🔍 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        try:
            # Step 1: Load data and model
            self.load_data_and_model()
            
            # Step 2: Calculate basic metrics
            self.calculate_basic_metrics()
            
            # Step 3: Analyze confusion matrix
            self.analyze_confusion_matrix()
            
            # Step 4: Perform cross-validation
            self.perform_cross_validation()
            
            # Step 5: Analyze feature importance
            self.analyze_feature_importance()
            
            # Step 6: Analyze decision tree (if applicable)
            if visualize_tree:
                self.analyze_decision_tree()
            
            # Step 7: Calculate business impact
            self.calculate_business_impact()
            
            # Step 8: Generate performance rating
            self.generate_performance_rating()
            
            # Step 9: Save results
            self.save_results()
            
            # Step 10: Generate reports
            if generate_reports:
                self.generate_summary_report()
            
            # Print summary
            self.print_summary()
            
            print("\n" + "=" * 60)
            print("✅ COMPREHENSIVE EVALUATION COMPLETED")
            print("=" * 60)
            print(f"📁 Results saved to: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def print_summary(self):
        """Print evaluation summary to console."""
        metrics = self.results['basic_metrics']
        rating = self.results['performance_rating']
        business = self.results['business_impact']
        
        print(f"\n📊 EVALUATION SUMMARY")
        print("-" * 40)
        print(f"Overall Grade: {rating['overall_grade']}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"Precision: {metrics['precision']:.1%}")
        print(f"Recall: {metrics['recall']:.1%}")
        print(f"Net Business Value: ${business['financial_impact']['net_value']:,.0f}")
        
        # Add tree information if available
        if 'decision_tree' in self.results and self.results['decision_tree']:
            tree_stats = self.results['decision_tree']['statistics']
            visualizations = self.results['decision_tree']['visualizations']
            print(f"Model Tree Depth: {tree_stats['max_depth']}")
            
            if visualizations:
                actual_depth = visualizations.get('actual_model_depth', tree_stats['max_depth'])
                viz_depth = visualizations.get('visualization_depth', actual_depth)
                depth_limited = visualizations.get('depth_limited', False)
                
                if depth_limited:
                    print(f"Visualization Depth: {viz_depth} (limited from {actual_depth})")
                else:
                    print(f"Visualization Depth: {viz_depth} (full tree)")
                    
                viz_count = len(visualizations.get('all_visualizations', []))
                print(f"Visualizations Generated: {viz_count}")
            else:
                print(f"Depth Limiting: {'Enabled' if tree_stats['depth_limiting_enabled'] else 'Disabled'}")
            
            print(f"Tree Nodes: {tree_stats['n_nodes']}")
            print(f"Features Used: {tree_stats['n_features_used']}/{tree_stats['total_features']}")


def main():
    """Main function to run evaluation from command line."""
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model-path', default='outputs/models/baseline_model.pkl',
                       help='Path to the model file')
    parser.add_argument('--data-path', default='datasets/Dataset_Cay quyet dinh_HV.xlsx',
                       help='Path to the dataset')
    parser.add_argument('--output-dir', default='outputs/evaluation',
                       help='Output directory for reports')
    parser.add_argument('--max-tree-depth', type=int, default=None,
                       help='Maximum depth for decision tree visualization (default: None - show full tree)')
    parser.add_argument('--limit-viz-depth', action='store_true',
                       help='Limit visualization depth to max-tree-depth (default: show full tree)')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip generating markdown reports')
    parser.add_argument('--no-tree-viz', action='store_true',
                       help='Skip decision tree visualization')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_tree_depth=args.max_tree_depth,
        limit_viz_depth=args.limit_viz_depth
    )
    
    results = evaluator.run_complete_evaluation(
        generate_reports=not args.no_reports,
        visualize_tree=not args.no_tree_viz
    )
    
    return results


if __name__ == "__main__":
    main() 