# Model Evaluation Summary

*Generated on: 2025-05-25 00:49:30*

## Quick Performance Overview

| Metric | Value | Rating |
|--------|--------|--------|
| **Accuracy** | 80.46% | Good |
| **ROC AUC** | 0.867 | Good |
| **Precision** | 51.25% | - |
| **Recall** | 80.34% | - |
| **F1-Score** | 0.626 | - |

**Overall Grade: B+**

## Confusion Matrix
```
                 Predicted
                No    Yes
Actual    No    561  136
          Yes     35  143
```


## Decision Tree Analysis
- **Actual Tree Depth**: 5 levels
- **Configured Max Depth**: None levels
- **Total Nodes**: 33
- **Leaf Nodes**: 17
- **Features Used**: 9/36
- **Visualizations Generated**: 3 files at depths [5, 2, 4]
- **Decision Rules**: Exported to text file


## Business Impact
- **Total Customers Evaluated**: 875
- **Actual Churners**: 178 (20.3%)
- **Successfully Identified**: 143 (80.3% of actual)
- **Missed Churners**: 35
- **False Alarms**: 136

### Financial Impact
- **Net Value**: $40,050
- **ROI**: 287.1%

## Model Assessment

### Strengths
- [+] High Recall (80.3%)
- [+] Good AUC (0.867)
- [+] Good Accuracy (80.5%)

### Areas for Improvement
- [-] Low Precision (51.3%)
- False Positive Rate: 19.5%

## Files Generated
- evaluation_results.json - Complete evaluation data
- decision_tree_visualization.png - Tree visualization
- decision_tree_simplified.png - Simplified tree view
- decision_tree_rules.txt - Detailed tree rules and analysis

## Deployment Recommendation
[+] **APPROVED** for production deployment

---
*Evaluation completed using 875 test samples*
