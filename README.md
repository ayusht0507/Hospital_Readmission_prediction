```markdown
# Hospital Readmission Prediction

### Machine Learning Classification for 30-Day Hospital Readmission Risk

A machine learning project that predicts whether a diabetic patient is likely to be **readmitted to the hospital within 30 days** using demographic, medical, admission, medication, diagnosis, and previous-visit information.

The project implements an end-to-end machine learning workflow including **data cleaning, feature engineering, preprocessing, L2-regularized Logistic Regression, XGBoost, cross-validation, class-imbalance handling, threshold tuning, and model evaluation**.

> **Educational Disclaimer:** This project is developed for educational and machine learning demonstration purposes. It is not intended for clinical diagnosis, treatment, or real-world medical decision-making.

---

## 📌 Overview

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The original `readmitted` variable is converted into a binary classification target:

| Target | Meaning |
|---|---|
| `1` | Readmitted within 30 days |
| `0` | Not readmitted within 30 days |

The notebook follows this workflow:

```text
Dataset Upload
      ↓
Data Loading
      ↓
Data Cleaning
      ↓
Target Creation
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Train / Validation / Test Split
      ↓
Preprocessing
      ↓
Logistic Regression
      ↓
XGBoost
      ↓
Threshold Tuning
      ↓
Final Evaluation
      ↓
Visualization
      ↓
Sample Prediction
```

---

## Problem Statement

Build a machine learning classification system that predicts whether a diabetic patient will be **readmitted within 30 days** based on available patient, admission, diagnosis, medication, and previous-visit information.

The project focuses on:

- Binary classification
- Class imbalance
- Model comparison
- Threshold tuning
- Multiple evaluation metrics
- Data leakage prevention

---

## Project Objectives

- Load and understand hospital patient data
- Clean missing and inconsistent values
- Convert the original readmission variable into a binary target
- Perform feature engineering
- Prevent data leakage
- Build a preprocessing pipeline
- Train L2-regularized Logistic Regression
- Tune Logistic Regression using cross-validation
- Train XGBoost as a comparison model
- Handle class imbalance
- Tune the classification threshold
- Evaluate model performance
- Visualize model performance
- Generate a sample prediction

---

## 📊 Dataset

### Diabetes 130-US Hospitals for Years 1999–2008

The dataset contains patient records from multiple U.S. hospitals and includes information related to:

- Patient demographics
- Hospital admission
- Medical specialty
- Length of hospital stay
- Laboratory procedures
- Medical procedures
- Medication usage
- Previous hospital visits
- Diagnoses
- Diabetes-related information
- Readmission status

The dataset is **not included in this GitHub repository**.

The notebook provides a file-upload step where the dataset ZIP file can be uploaded and extracted automatically.

---

## Target Variable

The original `readmitted` column contains multiple readmission categories.

For this project, it is converted into:

```text
1 → Readmitted within 30 days
0 → Not readmitted within 30 days
```

---

## Features Used

The final model uses:

```text
age
race
gender
admission_type_id
admission_source_id
medical_specialty
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
number_diagnoses
max_glu_serum
A1Cresult
metformin
insulin
diabetesMed
diag_1_group
diag_2_group
diag_3_group
total_prior_visits
medications_per_day
```

---

## Feature Engineering

### Diagnosis Grouping

The diagnosis variables are grouped into broader categories:

```text
diag_1_group
diag_2_group
diag_3_group
```

### Total Prior Visits

Previous outpatient, emergency, and inpatient visits are combined:

```text
total_prior_visits =
number_outpatient +
number_emergency +
number_inpatient
```

### Medications per Day

Medication usage is normalized by the length of hospital stay:

```text
medications_per_day =
num_medications / time_in_hospital
```

---

## Data Leakage Prevention

`discharge_disposition_id` is excluded from the final feature set because it is associated with the patient's discharge process and can contain information determined later during the hospital stay.

The preprocessing steps are also placed inside machine learning pipelines so that preprocessing is performed as part of the model workflow.

---

## Train / Validation / Test Split

The dataset is divided into:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratified splitting is used to preserve the target-class distribution.

```text
random_state = 42
```

---

## Data Preprocessing

### Numerical Features

- Median imputation
- StandardScaler

### Categorical Features

- Most-frequent imputation
- One-hot encoding
- `handle_unknown="ignore"`

The preprocessing steps are implemented using `Pipeline` and `ColumnTransformer`.

---

## Models Used

### 1. Logistic Regression

The project uses **Logistic Regression with L2 regularization**.

Hyperparameters are tuned using `GridSearchCV` with 5-fold cross-validation.

The search includes:

```text
C = [0.01, 0.1, 1, 10, 100]
class_weight = [None, "balanced"]
```

The optimization metric is:

```text
ROC-AUC
```

### 2. XGBoost

XGBoost is used as a comparison model.

Configuration:

```text
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

Class imbalance is handled using `scale_pos_weight`.

---

## Class Imbalance

The readmission target is imbalanced.

Therefore, accuracy alone is not sufficient for evaluating the models.

The project evaluates:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Class imbalance is addressed using model configuration, including `class_weight` for Logistic Regression and `scale_pos_weight` for XGBoost.

---

## Threshold Tuning

The notebook evaluates multiple probability thresholds instead of relying only on the default `0.50` threshold.

Thresholds from:

```text
0.20 to 0.70
```

are evaluated, and the threshold producing the highest validation **F1-score** is selected.

This helps analyze the trade-off between precision and recall.

---

## Evaluation Metrics

### Accuracy

Percentage of total predictions that are correct.

### Precision

Among predicted positive cases, the proportion that are actually positive.

### Recall

Among actual positive cases, the proportion correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different thresholds.

---

## Final Model Results

The final models are evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.94% | 17.65% | 41.34% | 24.74% | 0.6330 |
| XGBoost | 71.08% | 17.92% | 44.45% | 25.54% | 0.6423 |

Because the target is imbalanced, these metrics should be considered together rather than relying only on accuracy.

---

## 📊 Visualizations

The notebook includes:

### Confusion Matrix

Displays:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC Curve

Shows the relationship between:

- False Positive Rate
- True Positive Rate

and displays the ROC-AUC value.

### Precision-Recall Curve

Shows the relationship between:

- Precision
- Recall

across different classification thresholds.

---

## 🔮 Sample Prediction

The notebook performs a prediction using a sample from the test set.

It displays:

```text
Probability
Threshold
Prediction
```

Example:

```text
Probability: 0.xx
Threshold: 0.xx
Prediction: Readmitted <30 days
```

---

# ▶️ How to Run

## Google Colab

Google Colab is the recommended way to run this notebook.

### 1. Open the Notebook

Open:

```text
Hospital_readmission_prediction.ipynb
```

in Google Colab.

You can upload the notebook using:

```text
Google Colab
→ File
→ Upload notebook
```

### 2. Download the Dataset

Download the:

```text
Diabetes 130-US Hospitals for Years 1999–2008
```

dataset and keep it in ZIP format.

### 3. Upload the Dataset

Run the dataset upload cell in the notebook.

When prompted, select the downloaded ZIP file.

The notebook automatically extracts the dataset.

### 4. Run the Notebook

Run all cells from top to bottom:

```text
Runtime → Run all
```

or:

```text
Ctrl + F9
```

### 5. View the Results

The notebook will generate:

- Logistic Regression results
- XGBoost results
- Threshold analysis
- Final test metrics
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Sample prediction

---

## 💻 Running Locally

### Install Required Libraries

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Hospital_readmission_prediction.ipynb
```

Then upload the dataset ZIP when prompted.

---

## 📁 Repository Structure

```text
Hospital-Readmission-Prediction/
│
├── Hospital_readmission_prediction.ipynb
│
└── README.md
```

The dataset is not included in the repository and must be uploaded separately when running the notebook.

---

## ⚠️ Limitations

- The dataset contains diabetic patient records.
- The data represents historical hospital records from 1999–2008.
- The model has not been clinically validated.
- Model performance depends on the available patient information.
- The relatively low precision indicates that a considerable number of positive predictions are false positives.
- The model is an educational machine learning demonstration and should not be used for clinical decision-making.

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting **30-day hospital readmission**.

The notebook covers:

```text
Data Cleaning
→ Feature Engineering
→ Data Preprocessing
→ Logistic Regression
→ Cross-Validation
→ XGBoost
→ Threshold Tuning
→ Model Evaluation
→ Visualization
→ Sample Prediction
```

The project demonstrates the importance of using **Precision, Recall, F1-Score, and ROC-AUC** alongside accuracy when evaluating an imbalanced classification problem.

> ⚕️ **This project is for educational purposes only and is not intended for clinical diagnosis, treatment, or medical decision-making.**
```
