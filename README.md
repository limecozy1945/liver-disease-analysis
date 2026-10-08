# Explainable AI System for Liver Disease Risk Prediction

An end-to-end machine learning pipeline that predicts liver disease risk from patient lifestyle and clinical data, classifies disease category, and explains every prediction using SHAP (SHapley Additive exPlanations).

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Methodology](#methodology)
4. [Explainability](#explainability)
5. [Ablation Study](#ablation-study)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Limitations](#limitations)
10. [Disclaimer](#disclaimer)
11. [Author](#author)

---

## Overview

Early identification of liver disease is important, yet black-box models are difficult to trust in healthcare settings. This project addresses both concerns by combining accurate gradient-boosted classifiers with transparent, feature-level explanations.

The system provides three capabilities:

- **Binary screening:** estimates the probability that a patient has liver disease.
- **Multi-class categorization:** distinguishes between six disease categories for a more granular view of patient condition.
- **Explainable output:** produces both global feature importance and patient-level explanations for each prediction, along with a categorical risk label.

## Key Results

| Task | Metric | Result |
| --- | --- | --- |
| Binary classification (disease vs. no disease) | Accuracy | **96.7%** |
| Binary classification | ROC-AUC | **0.987** |
| Multi-class classification (6 categories) | Accuracy | **92.1%** |
| Multi-class classification | Macro F1-score | **0.88** |

Evaluation was performed on a held-out test set of 1,500 patient records. Per-class multi-class performance ranges from an F1-score of 0.74 (class 4) to 0.96 (classes 2 and 5).

## Methodology

### Data

The model is trained and evaluated on the Indian liver disease dataset, provided as separate training and testing files. Each record contains demographic, lifestyle, symptom, comorbidity, and laboratory features.

| Feature group | Examples |
| --- | --- |
| Demographics and body metrics | Age, Gender, Occupation, BMI, Obesity Class |
| Lifestyle | Diet Quality, Physical Activity, Sleep Hours, Smoking Status, Alcohol Consumption |
| Symptoms | Fatigue, Jaundice, Abdominal Pain, Itching, Ascites, Dark Urine, Weight Loss |
| Comorbidities | Diabetes, Hypertension, Genetic History |
| Clinical biomarkers | ALT, AST, Bilirubin, Albumin, Platelets, Alkaline Phosphatase |

The target variable is `Liver_Disease_Type`. Class `0` denotes absence of disease; the binary label is derived as `Liver_Disease_Type != 0`.

### Preprocessing

- The `Patient_ID` column is removed, as it carries no predictive information.
- Categorical features are label-encoded. Encoders are fitted on the training set only and applied to the test set to prevent data leakage.

### Feature Engineering

Three clinically motivated features are derived:

| Feature | Definition | Rationale |
| --- | --- | --- |
| `AST_ALT_ratio` | AST / ALT | Established indicator used in liver injury assessment |
| `Bilirubin_Albumin` | Bilirubin / Albumin | Combines excretory and synthetic liver function |
| `BMI_Alcohol` | BMI × Alcohol Consumption | Captures the combined effect of metabolic and alcohol-related risk |

### Models

Both models use XGBoost with identical core hyperparameters:

| Parameter | Value |
| --- | --- |
| `n_estimators` | 800 |
| `max_depth` | 10 |
| `learning_rate` | 0.03 |

The multi-class model additionally uses the `multi:softprob` objective with `mlogloss` as the evaluation metric.

### Risk Stratification

The predicted probability from the binary model is converted into an interpretable risk category:

| Predicted probability | Risk category |
| --- | --- |
| Greater than 0.85 | High Risk |
| Greater than 0.60 | Moderate Risk |
| 0.60 or below | Low Risk |

## Explainability

Interpretability is a central design goal. SHAP's `TreeExplainer` is used to provide:

- **Global explanations:** SHAP summary plots for each of the six classes, showing which features drive predictions overall. Bilirubin, liver enzymes, and alcohol consumption emerge as dominant predictors.
- **Dependence plots:** visualizations of how individual features (for example, Bilirubin and ALT) influence model output and how they interact with other variables.
- **Local explanations:** SHAP force plots for individual patients, plus a plain-language summary function that lists the five most influential features and whether each increases or reduces predicted risk.

**Example output** for a sample patient (prediction: low risk, probability 0.10):

```
Albumin reduces predicted risk
Sym_Jaundice reduces predicted risk
AST reduces predicted risk
Bilirubin reduces predicted risk
Alk_Phosphatase increases predicted risk
```

## Ablation Study

To understand the contribution of each feature family, three binary models were trained on different feature subsets:

| Feature set | Accuracy |
| --- | --- |
| Lifestyle only | 86.6% |
| Clinical only | 92.5% |
| Lifestyle + clinical (all features) | **96.9%** |

Clinical biomarkers are individually more informative than lifestyle features, but the two groups are complementary: combining them yields a substantial improvement over either alone. Ablation models use 600 estimators with XGBoost's default depth and learning rate, so the combined-model figure differs slightly from the main binary model (96.7%).

## Repository Structure

```
.
├── data/
│   ├── Training_indian_liver_disease_dataset.csv
│   └── Testing_indian_liver_disease_dataset.csv
├── liver_explainable_ai.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
```

## Usage

1. Place the training and testing CSV files in the `data/` directory.
2. Launch the notebook:

   ```bash
   jupyter notebook liver_explainable_ai.ipynb
   ```

3. Run all cells to preprocess the data, train both models, evaluate performance, and generate SHAP visualizations.

### Predicting for a New Patient

Define a patient profile using the same feature names and encodings as the training data. The notebook applies the identical feature engineering, then returns the prediction, probability, risk category, and explanation:

```python
patient = {
    'Age': 45, 'Gender': 1, 'Occupation': 2, 'BMI': 28, 'Obesity_Class': 1,
    'Diet_Quality': 2, 'Physical_Activity': 1, 'Sleep_Hours': 6,
    'Smoking_Status': 1, 'Alcohol_Consumption': 3,
    'Sym_Fatigue': 1, 'Sym_Jaundice': 0, 'Sym_Abdominal_Pain': 1,
    'Sym_Itching': 0, 'Sym_Ascites': 0, 'Sym_Dark_Urine': 1,
    'Sym_Weight_Loss': 0, 'Comorb_Diabetes': 1, 'Comorb_Hypertension': 0,
    'Comorb_Genetic_History': 0, 'ALT': 65, 'AST': 70, 'Bilirubin': 1.8,
    'Albumin': 3.2, 'Platelets': 210, 'Alk_Phosphatase': 180
}
```

## Limitations

- Results are based on a single train/test split. Cross-validation and hyperparameter tuning were not performed, so reported figures may vary under different splits.
- The models have been validated on one dataset only. Performance on data from other populations or clinical settings is unknown.
- Categorical features are label-encoded, which imposes an arbitrary ordering; this is generally tolerated by tree-based models but is not ideal for all variables.
- SHAP explanations describe the model's behavior, not causal relationships in patients.

## Disclaimer

This project is intended for research and educational purposes only. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.


