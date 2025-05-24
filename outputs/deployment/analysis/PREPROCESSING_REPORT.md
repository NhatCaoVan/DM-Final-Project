# Customer Churn Prediction - Preprocessing & Analysis Report

*Generated on: 2025-05-25 00:49:27*

## Executive Summary

This report provides a comprehensive analysis of the data preprocessing pipeline and model feature analysis for the customer churn prediction project. The analysis covers data cleaning, feature engineering, model structure, and feature importance assessment.

## 1. Dataset Overview

### 1.1 Original Dataset
- **File**: `datasets/Dataset_Cay quyet dinh_HV.xlsx`
- **Shape**: 5,000 rows × 14 columns
- **Target Variable**: `churn`
- **Data Types**: 14 features

### 1.2 Feature Types Distribution
```
Numeric Features: 9
Categorical Features: 3
Total Features: 12
```

### 1.3 Target Variable Distribution

**Class Distribution:**
- Class 0: 3,779 samples (75.6%)
- Class 1: 1,221 samples (24.4%)


## 2. Data Quality Assessment

### 2.1 Missing Values Analysis

✅ **No missing values found in the dataset**


### 2.2 Outlier Detection and Treatment

**Method**: IQR (Interquartile Range) with threshold = 1.5

**Results:**
- **Original samples**: 5,000
- **Samples after cleaning**: 4,371
- **Samples removed**: 629 (12.58%)

**Outliers by Feature:**
- `age`: 8 outliers
- `month`: 0 outliers
- `year`: 0 outliers
- `data_volume`: 40 outliers
- `data_spending`: 434 outliers
- `sms_volume`: 72 outliers
- `sms_spending`: 33 outliers
- `voice_duration`: 36 outliers
- `voice_spending`: 6 outliers


## 3. Feature Engineering

### 3.1 Numerical Features (9)
1. `age`
2. `month`
3. `year`
4. `data_volume`
5. `data_spending`
6. `sms_volume`
7. `sms_spending`
8. `voice_duration`
9. `voice_spending`


### 3.2 Categorical Features (3)
1. `gender` (2 unique values)
2. `district` (5 unique values)
3. `data_package` (26 unique values)


### 3.3 Preprocessing Pipeline

**Numerical Features Processing:**
- **Method**: StandardScaler (z-score normalization)
- **Formula**: z = (x - μ) / σ
- **Effect**: Mean = 0, Standard Deviation = 1

**Categorical Features Processing:**
- **Method**: One-Hot Encoding
- **Parameters**: drop='first', handle_unknown='ignore'
- **Effect**: Creates binary features for each category

**Final Feature Count**: 36 features after preprocessing

## 4. Data Splitting Strategy

```
Training Set: 3,496 samples (80.0%)
Test Set: 875 samples (20.0%)
Stratification: Applied to maintain class balance
Random State: 42 (for reproducibility)
```

## 5. Class Imbalance Handling

**Method**: SMOTE (Synthetic Minority Oversampling Technique)

```
Original Training Samples: 3,496
After SMOTE: 5,566
Oversampling Ratio: 1.59x
```

## 6. Model Analysis

### 6.1 Model Structure

- **Model Type**: DecisionTreeClassifier
- **Number of Input Features**: 36
- **Number of Classes**: 2
- **Tree Depth**: 5 levels
- **Number of Nodes**: 33
- **Number of Leaves**: 17
- **Features Used**: 9 out of 36


### 6.2 Feature Importance Analysis

**Top 10 Most Important Features:**

| Rank | Feature | Importance Score | Percentage |
|------|---------|------------------|------------|
| 1 | `gender_Male` | 0.222630 | 22.26% |
| 2 | `age` | 0.200994 | 20.10% |
| 3 | `data_package_D120` | 0.143629 | 14.36% |
| 4 | `data_package_BIG90` | 0.142930 | 14.29% |
| 5 | `data_spending` | 0.138619 | 13.86% |
| 6 | `data_package_DMAX100` | 0.094482 | 9.45% |
| 7 | `district_Phu Vang` | 0.051884 | 5.19% |
| 8 | `voice_duration` | 0.002697 | 0.27% |
| 9 | `data_volume` | 0.002135 | 0.21% |
| 10 | `month` | 0.000000 | 0.00% |


**Features with Zero Importance**: 27 out of 36

These features do not contribute to the model's decision-making process:
- `month`, `year`, `sms_spending`
- `district_Phu Nhuan`, `district_Phu Loc`, `voice_spending`
- `district_Vinh hoi`, `data_package_BIG70`, `data_package_BM69`
- `data_package_D70`, `sms_volume`, `data_package_DCH`
- `data_package_DGM`, `data_package_DINO70`, `data_package_DGT`
- `data_package_DMAX`, `data_package_DVE80`, `data_package_GM30`
- `data_package_GT30`, `data_package_M1`, `data_package_M3`
- `data_package_MAX100`, `data_package_SP50`, `data_package_THAGA60`
- `data_package_THAGA70`, `data_package_VE55`, `data_package_VS50`


### 6.3 Feature Utilization Summary

```
Total Features Available: 36
Features Used by Model: 9 (25.0%)
Features Ignored: 27 (75.0%)

Top Feature Importance: 0.222630 (gender_Male)
Feature Importance Range: 0.000000 - 0.222630
Mean Feature Importance: 0.027778
```

## 7. Key Insights

### 7.1 Data Quality
- ✅ Dataset is clean with minimal missing values
- ✅ Outliers were systematically identified and removed
- ✅ Class distribution is documented and handled

### 7.2 Feature Engineering
- ✅ Proper scaling applied to numerical features
- ✅ Categorical features encoded using one-hot encoding
- ✅ Feature dimensionality increased from 12 to 36 for ML processing

### 7.3 Model Characteristics
- 🌳 **Decision Tree Model**: Easy to interpret, handles mixed data types well
- 🎯 **Feature Selection**: Model naturally selects most informative features
- 📊 **Binary Splits**: Creates clear decision boundaries
- ⚡ **Fast Prediction**: Efficient for real-time inference


### 7.4 Feature Importance Insights
- 🎯 **Highly Concentrated**: Top 5 features account for 84.9% of total importance
- 📈 **Demographic Focus**: Model relies heavily on demographic features
- 💡 **Feature Redundancy**: 27 features provide no additional information
- 🔍 **Model Simplicity**: Effective with relatively few key features

## 8. Recommendations

### 8.1 For Current Model
1. ✅ **Model is Production Ready**: All preprocessing steps are documented and reproducible
2. 🎯 **Feature Selection**: Consider removing zero-importance features for efficiency
3. 📊 **Monitoring**: Track feature drift in production data

### 8.2 For Future Improvements
1. 🧠 **Feature Engineering**: Explore feature interactions and polynomial features
2. 📈 **Advanced Models**: Consider ensemble methods to capture more complex patterns
3. 🔍 **Feature Selection**: Apply systematic feature selection techniques
4. 🎛️ **Hyperparameter Tuning**: Optimize model parameters for better performance

## 9. Technical Specifications

### 9.1 Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning pipeline
- imbalanced-learn: SMOTE implementation

### 9.2 Reproducibility
- **Random State**: 42 (fixed across all operations)
- **Pipeline Version**: Consistent preprocessing steps
- **Data Version**: (5000, 14) (original dataset snapshot)

### 9.3 Files Generated
- `baseline_model.pkl`: Trained model artifact
- `PREPROCESSING_REPORT.md`: This comprehensive report
- Feature importance visualizations (if generated)

---

*This report was automatically generated by the preprocessing analysis pipeline.*  
*For questions or clarifications, please refer to the source code documentation.*
