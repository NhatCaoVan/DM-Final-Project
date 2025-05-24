#!/usr/bin/env python3
"""
Unified Deployment Analysis Script

This script combines all analysis tools into one comprehensive pipeline:
1. Feature importance analysis and preprocessing reports
2. Complete model evaluation with decision tree visualization
3. Detailed tree results analysis and business insights

Usage:
    python deploy_analysis.py [options]

Options:
    --model-path: Path to the model file (default: outputs/models/baseline_model.pkl)
    --data-path: Path to the dataset (default: datasets/Dataset_Cay quyet dinh_HV.xlsx)
    --output-dir: Output directory for reports (default: outputs/deployment)
    --skip-viz: Skip decision tree visualization (faster execution)
    --quick: Run quick analysis (skip detailed reports)
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all required modules
from churn_prediction.src.analysis import run_complete_analysis
from evaluate_model import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentAnalyzer:
    """Unified deployment analysis class."""
    
    def __init__(self, model_path, data_path, output_dir="outputs/deployment"):
        """Initialize the deployment analyzer."""
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for all results
        self.analysis_results = {}
        self.evaluation_results = {}
        self.combined_results = {}
        
    def run_feature_analysis(self):
        """Run complete feature importance and preprocessing analysis."""
        print("📊 STEP 1: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        try:
            # Create analysis subdirectory
            analysis_dir = self.output_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Run the complete analysis
            self.analysis_results = run_complete_analysis(
                model_path=str(self.model_path),
                data_path=str(self.data_path),
                output_dir=str(analysis_dir)
            )
            
            print("✅ Feature analysis completed")
            print(f"   📄 Generated {len(self.analysis_results['output_files'])} analysis files")
            
        except Exception as e:
            print(f"❌ Error in feature analysis: {e}")
            raise
    
    def run_model_evaluation(self, skip_viz=False):
        """Run comprehensive model evaluation."""
        print("\n📈 STEP 2: MODEL EVALUATION")
        print("=" * 50)
        
        try:
            # Create evaluation subdirectory
            eval_dir = self.output_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            
            # Run the evaluation
            evaluator = ModelEvaluator(
                model_path=str(self.model_path),
                data_path=str(self.data_path),
                output_dir=str(eval_dir)
            )
            
            self.evaluation_results = evaluator.run_complete_evaluation(
                generate_reports=True,
                visualize_tree=not skip_viz
            )
            
            print("✅ Model evaluation completed")
            
        except Exception as e:
            print(f"❌ Error in model evaluation: {e}")
            raise
    
    def analyze_tree_results(self):
        """Analyze decision tree results in detail."""
        print("\n🌳 STEP 3: DECISION TREE ANALYSIS")
        print("=" * 50)
        
        try:
            if 'decision_tree' not in self.evaluation_results or not self.evaluation_results['decision_tree']:
                print("⚠️  No decision tree data available for detailed analysis")
                return
            
            tree_data = self.evaluation_results['decision_tree']
            feature_data = self.evaluation_results.get('feature_importance')
            
            # Analyze tree structure
            self.analyze_tree_structure_detailed(tree_data)
            
            # Analyze feature importance in business context
            self.analyze_feature_importance_detailed(feature_data)
            
            # Generate business insights
            self.generate_business_insights_detailed()
            
            print("✅ Decision tree analysis completed")
            
        except Exception as e:
            print(f"❌ Error in tree analysis: {e}")
            raise
    
    def analyze_tree_structure_detailed(self, tree_data):
        """Detailed tree structure analysis."""
        stats = tree_data['statistics']
        analysis = tree_data['analysis']
        
        print("🔍 Tree Structure Analysis:")
        print(f"   • Depth: {stats['max_depth']} levels")
        print(f"   • Nodes: {stats['n_nodes']} total, {stats['n_leaves']} leaves")
        print(f"   • Efficiency: {stats['n_leaves']/stats['n_nodes']*100:.1f}% leaf ratio")
        print(f"   • Feature Usage: {stats['n_features_used']}/{stats['total_features']} ({stats['n_features_used']/stats['total_features']*100:.1f}%)")
        
        # Important paths summary
        important_paths = analysis['important_paths']
        churn_paths = [p for p in important_paths if p['prediction'] == 'Churn']
        retention_paths = [p for p in important_paths if p['prediction'] == 'No Churn']
        
        print(f"   • Decision Paths: {len(important_paths)} significant paths")
        print(f"   • Churn Paths: {len(churn_paths)} paths leading to churn")
        print(f"   • Retention Paths: {len(retention_paths)} paths leading to retention")
    
    def analyze_feature_importance_detailed(self, feature_data):
        """Detailed feature importance analysis."""
        if not feature_data:
            print("⚠️  No feature importance data available")
            return
        
        features = feature_data['importance_data']
        used_features = [f for f in features if f['importance'] > 0]
        
        print(f"\n🎯 Feature Importance Analysis:")
        print(f"   • Total Features: {feature_data['total_features']}")
        print(f"   • Used Features: {feature_data['used_features']} ({feature_data['used_features']/feature_data['total_features']*100:.1f}%)")
        print(f"   • Top 5 Contribution: {feature_data['top_5_cumulative_importance']:.1f}%")
        
        # Top features
        print(f"   • Top 3 Features:")
        for i, feature in enumerate(used_features[:3], 1):
            print(f"     {i}. {feature['feature']}: {feature['importance_pct']:.1f}%")
    
    def generate_business_insights_detailed(self):
        """Generate detailed business insights."""
        metrics = self.evaluation_results['basic_metrics']
        business = self.evaluation_results['business_impact']
        
        print(f"\n💼 Business Insights:")
        print(f"   • Accuracy: {metrics['accuracy']*100:.1f}% (identifies {metrics['recall']*100:.1f}% of churners)")
        print(f"   • Precision: {metrics['precision']*100:.1f}% (of flagged customers, {metrics['precision']*100:.1f}% actually churn)")
        print(f"   • Business Value: ${business['financial_impact']['net_value']:,} net positive")
        print(f"   • ROI: {business['financial_impact']['roi']*100:.1f}% return on investment")
        print(f"   • Campaign Target: {business['predicted_churners']} customers for retention")
    
    def generate_deployment_summary(self):
        """Generate comprehensive deployment summary."""
        print("\n📋 STEP 4: DEPLOYMENT SUMMARY")
        print("=" * 50)
        
        try:
            # Combine all results
            self.combined_results = {
                'deployment_info': {
                    'analysis_date': datetime.now().isoformat(),
                    'model_path': str(self.model_path),
                    'data_path': str(self.data_path),
                    'output_directory': str(self.output_dir)
                },
                'feature_analysis': self.analysis_results,
                'model_evaluation': self.evaluation_results,
                'summary_metrics': self.extract_key_metrics()
            }
            
            # Save combined results
            summary_file = self.output_dir / "deployment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.combined_results, f, indent=2, default=str)
            
            # Generate markdown report
            self.generate_deployment_report()
            
            print("✅ Deployment summary generated")
            print(f"   📄 JSON Summary: {summary_file}")
            print(f"   📄 Report: {self.output_dir / 'DEPLOYMENT_REPORT.md'}")
            
        except Exception as e:
            print(f"❌ Error generating deployment summary: {e}")
            raise
    
    def extract_key_metrics(self):
        """Extract key metrics for deployment decision."""
        metrics = self.evaluation_results.get('basic_metrics', {})
        business = self.evaluation_results.get('business_impact', {})
        rating = self.evaluation_results.get('performance_rating', {})
        tree_stats = self.evaluation_results.get('decision_tree', {}).get('statistics', {})
        
        return {
            'model_performance': {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'overall_grade': rating.get('overall_grade', 'Unknown')
            },
            'business_impact': {
                'net_value': business.get('financial_impact', {}).get('net_value', 0),
                'roi_percentage': business.get('financial_impact', {}).get('roi', 0) * 100,
                'customers_at_risk': business.get('actual_churners', 0),
                'customers_to_target': business.get('predicted_churners', 0)
            },
            'model_complexity': {
                'tree_depth': tree_stats.get('max_depth', 0),
                'total_nodes': tree_stats.get('n_nodes', 0),
                'features_used': tree_stats.get('n_features_used', 0),
                'total_features': tree_stats.get('total_features', 0)
            },
            'deployment_readiness': self.assess_deployment_readiness()
        }
    
    def assess_deployment_readiness(self):
        """Assess if model is ready for deployment."""
        metrics = self.evaluation_results.get('basic_metrics', {})
        rating = self.evaluation_results.get('performance_rating', {})
        
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('roc_auc', 0)
        grade = rating.get('overall_grade', 'D')
        
        # Deployment criteria
        ready = (
            accuracy >= 0.75 and  # At least 75% accuracy
            auc >= 0.8 and       # At least 0.8 AUC
            grade in ['A', 'B+', 'B']  # Good overall grade
        )
        
        if ready:
            status = "APPROVED"
            recommendation = "Model is ready for production deployment"
        elif accuracy >= 0.7 and auc >= 0.75:
            status = "CONDITIONAL"
            recommendation = "Model can be deployed with close monitoring"
        else:
            status = "NOT READY"
            recommendation = "Model needs improvement before deployment"
        
        return {
            'status': status,
            'recommendation': recommendation,
            'criteria_met': {
                'accuracy_threshold': accuracy >= 0.75,
                'auc_threshold': auc >= 0.8,
                'grade_acceptable': grade in ['A', 'B+', 'B']
            }
        }
    
    def generate_deployment_report(self):
        """Generate markdown deployment report."""
        metrics = self.combined_results['summary_metrics']
        
        report = f"""# Model Deployment Analysis Report

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

### Deployment Status: **{metrics['deployment_readiness']['status']}**
**Recommendation**: {metrics['deployment_readiness']['recommendation']}

## Model Performance

| Metric | Value | Status |
|--------|--------|--------|
| **Overall Grade** | {metrics['model_performance']['overall_grade']} | {"✅" if metrics['model_performance']['overall_grade'] in ['A', 'B+', 'B'] else "⚠️"} |
| **Accuracy** | {metrics['model_performance']['accuracy']:.1%} | {"✅" if metrics['model_performance']['accuracy'] >= 0.75 else "⚠️"} |
| **ROC AUC** | {metrics['model_performance']['roc_auc']:.3f} | {"✅" if metrics['model_performance']['roc_auc'] >= 0.8 else "⚠️"} |
| **Precision** | {metrics['model_performance']['precision']:.1%} | - |
| **Recall** | {metrics['model_performance']['recall']:.1%} | - |
| **F1-Score** | {metrics['model_performance']['f1_score']:.3f} | - |

## Business Impact

- **Net Value**: ${metrics['business_impact']['net_value']:,.0f}
- **ROI**: {metrics['business_impact']['roi_percentage']:.1f}%
- **Customers at Risk**: {metrics['business_impact']['customers_at_risk']:,}
- **Target for Campaigns**: {metrics['business_impact']['customers_to_target']:,}

## Model Complexity

- **Decision Tree Depth**: {metrics['model_complexity']['tree_depth']} levels
- **Total Nodes**: {metrics['model_complexity']['total_nodes']}
- **Features Used**: {metrics['model_complexity']['features_used']}/{metrics['model_complexity']['total_features']} ({metrics['model_complexity']['features_used']/max(metrics['model_complexity']['total_features'],1)*100:.1f}%)

## Deployment Checklist

### ✅ Criteria Assessment
- {"✅" if metrics['deployment_readiness']['criteria_met']['accuracy_threshold'] else "❌"} Accuracy ≥ 75%
- {"✅" if metrics['deployment_readiness']['criteria_met']['auc_threshold'] else "❌"} ROC AUC ≥ 0.8
- {"✅" if metrics['deployment_readiness']['criteria_met']['grade_acceptable'] else "❌"} Overall Grade A/B+/B

### 📋 Deployment Recommendations

1. **Model Monitoring**: Set up automated performance tracking
2. **Campaign Implementation**: Target {metrics['business_impact']['customers_to_target']:,} high-risk customers
3. **Business Integration**: Integrate with CRM and marketing automation
4. **Performance Review**: Schedule monthly model performance reviews

## Generated Files

### Analysis Files
- `analysis/feature_importance.png` - Feature importance visualization
- `analysis/feature_importance.csv` - Feature importance data
- `analysis/PREPROCESSING_REPORT.md` - Detailed preprocessing analysis

### Evaluation Files
- `evaluation/decision_tree_visualization.png` - Full tree visualization
- `evaluation/decision_tree_simplified.png` - Simplified tree view
- `evaluation/decision_tree_rules.txt` - Tree rules and analysis
- `evaluation/evaluation_results.json` - Complete evaluation data
- `evaluation/evaluation_summary.md` - Evaluation summary

### Deployment Files
- `deployment_summary.json` - Complete deployment analysis
- `DEPLOYMENT_REPORT.md` - This report

## Next Steps

1. **If APPROVED**: Proceed with production deployment
2. **If CONDITIONAL**: Deploy with enhanced monitoring
3. **If NOT READY**: Address performance issues before deployment

---
*This report contains all necessary information for deployment decision-making.*
"""

        # Save the report
        report_file = self.output_dir / "DEPLOYMENT_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def print_final_summary(self):
        """Print final deployment summary."""
        metrics = self.combined_results['summary_metrics']
        
        print("\n🎯 DEPLOYMENT ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Deployment status
        status = metrics['deployment_readiness']['status']
        print(f"📊 Deployment Status: {status}")
        print(f"💡 Recommendation: {metrics['deployment_readiness']['recommendation']}")
        
        # Key metrics
        print(f"\n📈 Key Performance Metrics:")
        print(f"   • Overall Grade: {metrics['model_performance']['overall_grade']}")
        print(f"   • Accuracy: {metrics['model_performance']['accuracy']:.1%}")
        print(f"   • ROC AUC: {metrics['model_performance']['roc_auc']:.3f}")
        print(f"   • Business Value: ${metrics['business_impact']['net_value']:,}")
        print(f"   • ROI: {metrics['business_impact']['roi_percentage']:.1f}%")
        
        # Model efficiency
        print(f"\n🔧 Model Efficiency:")
        print(f"   • Tree Depth: {metrics['model_complexity']['tree_depth']} levels")
        print(f"   • Features Used: {metrics['model_complexity']['features_used']}/{metrics['model_complexity']['total_features']}")
        print(f"   • Target Customers: {metrics['business_impact']['customers_to_target']:,}")
        
        # Files generated
        print(f"\n📁 Output Directory: {self.output_dir.absolute()}")
        total_files = len(list(self.output_dir.rglob('*'))) - len(list(self.output_dir.rglob('*/')))
        print(f"📄 Total Files Generated: {total_files}")
        
        print("\n" + "=" * 60)
        if status == "APPROVED":
            print("🟢 MODEL APPROVED FOR DEPLOYMENT")
        elif status == "CONDITIONAL": 
            print("🟡 MODEL CONDITIONALLY APPROVED - MONITOR CLOSELY")
        else:
            print("🔴 MODEL NOT READY FOR DEPLOYMENT")
        print("=" * 60)
    
    def run_complete_deployment_analysis(self, skip_viz=False, quick=False):
        """Run the complete deployment analysis pipeline."""
        print("🚀 UNIFIED DEPLOYMENT ANALYSIS")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Output: {self.output_dir.absolute()}")
        print("=" * 60)
        
        try:
            # Step 1: Feature Analysis
            self.run_feature_analysis()
            
            # Step 2: Model Evaluation  
            self.run_model_evaluation(skip_viz=skip_viz)
            
            # Step 3: Tree Analysis (unless quick mode)
            if not quick:
                self.analyze_tree_results()
            
            # Step 4: Generate Deployment Summary
            self.generate_deployment_summary()
            
            # Final Summary
            self.print_final_summary()
            
            return self.combined_results
            
        except Exception as e:
            print(f"\n❌ DEPLOYMENT ANALYSIS FAILED: {e}")
            raise


def main():
    """Main function for deployment analysis."""
    parser = argparse.ArgumentParser(description='Unified Deployment Analysis')
    parser.add_argument('--model-path', default='outputs/models/baseline_model.pkl',
                       help='Path to the model file')
    parser.add_argument('--data-path', default='datasets/Dataset_Cay quyet dinh_HV.xlsx',
                       help='Path to the dataset')
    parser.add_argument('--output-dir', default='outputs/deployment',
                       help='Output directory for all reports')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip decision tree visualization (faster execution)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (skip detailed reports)')
    
    args = parser.parse_args()
    
    # Create analyzer and run complete analysis
    analyzer = DeploymentAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    results = analyzer.run_complete_deployment_analysis(
        skip_viz=args.skip_viz,
        quick=args.quick
    )
    
    return results


if __name__ == "__main__":
    main() 