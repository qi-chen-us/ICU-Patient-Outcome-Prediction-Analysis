
# Predicting ICU Patient Outcomes: Analysis and Insights

## Overview

ICU patients are critically ill and require close monitoring. Identification of high-risk patients is vital for timely intervention, potentially preventing life-threatening events while optimizing hospital resource use.

In this project, I explored publicly available ICU data from Kaggle, developed machine learning models to predict patient outcomes (survived vs. died), and extracted clinical insights to inform care decisions.

## Objectives
- Evaluate missingness and apply data imputation strategies
- Assess predictive signal in raw clinical data
- Engineer temporal features to capture patient health dynamics
- Compare baseline models (raw data) with enhanced models using both raw and engineered features
- Evaluate model performance under class imbalance using relevant metrics

## Dataset Summary
- **Source**: [Kaggle - ICU Patient Outcome Prediction](https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu)
- **Size**: 3,600 patients with 199 features
- **Target**: `In-Hospital_death` (1 = Died, 0 = Survived)
- **Unique Identifier**: `recordid` (one per patient record)
- **Class Distribution**: 14% died, 86% survived

## Data Preprocessing
- Split dataset: 70% training, 30% testing, ensuring sufficient representation of positive cases in both sets
- Extreme but interpretable outliers retained; extreme uninterpretable outliers removed
- Missing data occurred at both patient and feature levels. Missingness was treated as a potentially informative indicator, while ensuring sufficient clinical data for reliable modeling
- Missing values were imputed using training data (median/mode) or patient-specific historical values

## Feature Engineering
- Temporal features (`_delta`, `_ratio`, `_range`, `_HighLow_ratio`) were created to capture trends during ICU stay, reflecting patient recovery or deterioration dynamics
- Additional indicators such as `BMI` and `BUN_Creatinine_Ratio` were derived to represent patients’ health status over time

## Modeling Approaches
- **Baseline Model**: Logistic Regression with L1 regularization trained on raw clinical data to support interpretability and linear multivariable relationships.
- **Enhanced Models**: Logistic Regression and XGBoost trained on raw and engineered features. XGBoost captures nonlinear relationships through an ensemble decision-tree approach.


## Performance Evaluation
- **Primary Metric**: Precision-Recall AUC (PR-AUC) due to class imbalance, balancing false negatives (missed high-risk patients) and false positives (overburdening hospital resources)

| Model                          | Features         | Threshold | Recall          | Precision      | FPR            | Accuracy        | PR AUC | ROC-AUC | Brier Score |
|:-------------------------------|:----------------|:---------|:----------------|:---------------|:---------------|:----------------|:-------|:--------|:------------|
| Logistic Regression (baseline) | Raw              | 0.50     | 0.75 (113/150)  | 0.38 (113/299) | 0.20 (186/911) | 0.79 (838/1061) | 0.52   | 0.85    | 0.15        |
| Logistic Regression            | Raw + Engineered | 0.14     | 0.83 (124/150)  | 0.36 (124/346) | 0.22 (222/911) | 0.77 (813/1061) | 0.54   | 0.87    | 0.088       |
| XGBoost                        | Raw + Engineered | 0.12     | 0.83 (124/150)  | 0.38 (124/323) | 0.24 (199/911) | 0.79 (836/1061) | 0.54   | 0.87    | 0.088       |

- **Thresholds** were selected to prioritize recall for high-risk patients, while limiting false positives that could strain hospital resources.


## Key Insights

### Clinical Insights
- Features such as `GCS`, `BUN`, and `Age` consistently influenced outcomes across models, reinforcing that ICU mortality is driven by multivariate interactions rather than a single measurement
- While `GCS_first` was insignificant in univariate statistical tests, its derivative `GCS_delta` (calculated from the first and last measurements) emerged as important in model predictions, highlighting that temporal changes provide clinically relevant information beyond single-point measurements
- SHAP analysis of misclassified patients indicated that ICU data may not always capture the most recent patient condition, which can change rapidly with medical interventions, making some predictions challenging

### Model Insights
- Engineered temporal features improved model performance across both Logistic Regression and XGBoost, indicating that capturing patient dynamics over time provides meaningful predictive information
- XGBoost achieved slightly higher precision and accuracy while maintaining the same recall as Logistic Regression, showing its strength in modeling nonlinear relationships
- Brier Scores (0.088) for models with engineered features indicate better calibration compared to the baseline, aligning predicted probabilities with observed outcomes. Calibration curves were more volatile at higher probabilities, likely due to the limited number of positive cases and inherent prediction uncertainty


## Takeaways
- Engineered temporal features capture patients’ dynamic health trajectories, improving predictive performance and model interpretability


## Limitations and Future Work
- Dataset limited to a single source, small sample size, and lacks detailed temporal and treatment information, likely due to patient privacy considerations.
- Future work includes external validation, assessing the generalizability of temporal features, and evaluating whether additional clinical temporal features improve prediction and interpretability

## Additional Resources
- Jupyter Notebook with preprocessing, feature engineering, modeling, evaluation, and visualizations: [View Notebook](icu_patient_outcome.ipynb)
- Key results and figures, including PR/ROC curves, confusion matrix, classification report, feature importance plots, and SHAP summary plots: [View Results](results)