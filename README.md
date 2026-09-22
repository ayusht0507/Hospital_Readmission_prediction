# Hospital Readmission Prediction

A machine learning project for predicting whether a diabetic patient will be readmitted to the hospital within 30 days.

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset and compares **L2-regularized Logistic Regression** with **XGBoost**.

> **Note:** This project is developed for educational purposes. It is not intended for clinical diagnosis, treatment, or real-world medical decision-making.

---

## Overview

Hospital readmission can be treated as a binary classification problem. In this project, patient information such as demographics, admission details, diagnoses, medications, and previous hospital visits is used to predict whether a patient will be readmitted within 30 days.

The original `readmitted` column contains three categories:

| Value | Meaning |
|---|---|
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days |
| `NO` | Not readmitted |

For this project, these values are converted into a binary target:

```text
1 → Readmitted within 30 days
0 → Not readmitted within 30 days
```

### Project Workflow

```text
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
Model Evaluation
    ↓
Visualization
    ↓
Sample Prediction
```

---

## Problem Statement

Build a machine learning classification model that predicts whether a diabetic patient will be **readmitted within 30 days** using information available from the patient's hospital record.

The project focuses on:

- Binary classification
- Feature engineering
- Data preprocessing
- Class imbalance
- Model comparison
- Cross-validation
- Threshold tuning
- Evaluation using multiple metrics
- Avoiding data leakage

---

## Project Objectives

The main objectives of this project are:

- Understand and clean the hospital dataset
- Convert the original readmission variable into a binary target
- Select relevant features
- Create additional features from existing patient information
- Prevent data leakage during preprocessing and model training
- Build a preprocessing pipeline
- Train L2-regularized Logistic Regression
- Tune Logistic Regression using cross-validation
- Train XGBoost as a second model
- Handle class imbalance
- Compare model performance
- Tune the classification threshold
- Evaluate the final models on unseen test data
- Visualize model performance
- Generate a sample prediction

---

## Dataset

### Diabetes 130-US Hospitals for Years 1999–2008

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The dataset contains records of diabetic patients from multiple U.S. hospitals.

The available information includes:

- Patient demographics
- Hospital admission information
- Medical specialty
- Length of hospital stay
- Laboratory procedures
- Medical procedures
- Medication information
- Previous hospital visits
- Diagnoses
- Diabetes-related information
- Readmission status

The dataset is **not included in this repository**.

The notebook contains a file-upload step that allows the dataset ZIP file to be uploaded and extracted before running the analysis.

---

## Target Variable

The original `readmitted` column contains three possible values:

```text
<30
>30
NO
```

For this project, the target is converted to:

```text
1 → Readmitted within 30 days
0 → Not readmitted within 30 days
```

The `>30` and `NO` categories are treated as the negative class.

---

## Features Used

The final model uses the following features:

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

Some additional features were created from the original dataset.

### Diagnosis Grouping

The original diagnosis variables are converted into broader diagnosis groups:

```text
diag_1_group
diag_2_group
diag_3_group
```

This reduces the number of individual diagnosis categories and makes them easier to use as model features.

### Total Prior Visits

The previous outpatient, emergency, and inpatient visits are combined into one feature:

```text
total_prior_visits =
    number_outpatient +
    number_emergency +
    number_inpatient
```

This gives the model a single feature representing the patient's previous hospital utilization.

### Medications per Day

Medication count is normalized using the patient's length of stay:

```text
medications_per_day =
    num_medications / time_in_hospital
```

This provides an approximate measure of medication usage relative to the duration of the hospital stay.

---

## Data Leakage Prevention

`discharge_disposition_id` is excluded from the final feature set.

The reason is that discharge disposition is associated with information available during the discharge process and could introduce information that would not be appropriate for the intended prediction setup.

The preprocessing operations are also placed inside machine learning pipelines. This ensures that transformations such as imputation, scaling, and encoding are fitted as part of the training process rather than using information from the test set.

---

## Train / Validation / Test Split

The dataset is divided into three parts:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

A stratified split is used so that the distribution of the target classes remains similar across the different datasets.

The random seed used in the project is:

```python
random_state = 42
```

---

## Data Preprocessing

Different preprocessing steps are applied to numerical and categorical features.

### Numerical Features

The numerical preprocessing includes:

- Median imputation
- Standard scaling using `StandardScaler`

### Categorical Features

The categorical preprocessing includes:

- Most-frequent-value imputation
- One-hot encoding
- `handle_unknown="ignore"`

The preprocessing is implemented using:

```python
Pipeline
ColumnTransformer
```

This keeps the preprocessing and model steps together and helps avoid data leakage.

---

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as one of the main classification models.

The model uses **L2 regularization**.

Hyperparameters are tuned using `GridSearchCV` with **5-fold cross-validation**.

The parameter search includes:

```text
C = [0.01, 0.1, 1, 10, 100]

class_weight = [None, "balanced"]
```

ROC-AUC is used as the optimization metric during hyperparameter tuning.

### 2. XGBoost

XGBoost is used as the second model for comparison.

The main configuration used in the project is:

```text
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

Class imbalance is handled using `scale_pos_weight`.

The purpose of using XGBoost alongside Logistic Regression is to compare a regularized linear model with a tree-based boosting model.

---

## Handling Class Imbalance

The target variable is imbalanced, with fewer positive cases than negative cases.

Because of this, accuracy alone does not provide a complete picture of model performance.

The project therefore evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Logistic Regression

During hyperparameter tuning, both of the following options are tested:

```text
class_weight = None
class_weight = "balanced"
```

### XGBoost

For XGBoost, class imbalance is handled using:

```text
scale_pos_weight
```

---

## Threshold Tuning

The default classification threshold is `0.50`, but this threshold is not always the most useful choice for an imbalanced classification problem.

In this project, multiple thresholds between:

```text
0.20 → 0.70
```

are tested.

The threshold that produces the highest **validation F1-score** is selected for the final prediction.

Changing the threshold affects the balance between precision and recall, so threshold tuning is included as part of the model evaluation.

---

## Evaluation Metrics

The following metrics are used to evaluate the models.

### Accuracy

The percentage of predictions that are correct out of all predictions.

### Precision

Precision shows how many of the patients predicted as positive were actually positive.

### Recall

Recall shows how many of the actual positive cases were correctly identified.

### F1-Score

F1-score combines precision and recall into a single metric.

### ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

Because the dataset is imbalanced, the metrics are considered together rather than relying only on accuracy.

---

## Final Results

The final models are evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.94% | 17.65% | 41.34% | 24.74% | 0.6330 |
| XGBoost | 71.08% | 17.92% | 44.45% | 25.54% | 0.6423 |

These results show that accuracy alone is not enough to evaluate the models because the target classes are imbalanced.

The relatively low precision also means that a significant number of positive predictions are false positives.

---

## Visualizations

The notebook generates several visualizations to understand model performance.

### Confusion Matrix

The confusion matrix shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC Curve

The ROC curve shows the relationship between:

- True Positive Rate
- False Positive Rate

The ROC-AUC value is also displayed.

### Precision-Recall Curve

The Precision-Recall curve shows how precision and recall change at different classification thresholds.

### Threshold Analysis

The notebook also evaluates different probability thresholds to show how the model's precision, recall, and F1-score change.

---

## Sample Prediction

The notebook includes a sample prediction using a patient record from the test set.

The prediction displays:

```text
Probability
Threshold
Prediction
```

For example:

```text
Probability: 0.xx
Threshold: 0.xx
Prediction: Readmitted <30 days
```

The actual values are generated when the notebook is executed.

---

## Technologies and Packages

The project is written in **Python** and uses the following libraries:

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and data manipulation |
| `scikit-learn` | Preprocessing, Logistic Regression, cross-validation and evaluation |
| `xgboost` | XGBoost classification model |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `jupyter` | Running the notebook locally |

---

## How to Run

There are two ways to run this project:

1. **Google Colab**
2. **Locally using Jupyter Notebook**

### Option 1 — Google Colab

Google Colab is the simplest option because you don't need to configure a local Python environment.

#### Step 1 — Open the Notebook

Open:

```text
Hospital_readmission_prediction.ipynb
```

in Google Colab.

#### Step 2 — Download the Dataset

Download the:

```text
Diabetes 130-US Hospitals for Years 1999–2008
```

dataset.

Keep the dataset in ZIP format if you are using the upload cell provided in the notebook.

#### Step 3 — Upload the Dataset

Run the dataset upload cell.

When the file selector appears, select the downloaded ZIP file.

The notebook will extract the dataset before continuing.

#### Step 4 — Run the Notebook

Run the cells from top to bottom.

You can use:

```text
Runtime → Run all
```

After execution, the notebook will generate the model results, evaluation metrics, visualizations, and sample prediction.

---

### Option 2 — Run Locally

#### 1. Install Python

Make sure Python 3 is installed on your system.

You can check it using:

```bash
python --version
```

or:

```bash
python3 --version
```

#### 2. Clone the Repository

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project folder:

```bash
cd Hospital-Readmission-Prediction
```

#### 3. Install the Required Packages

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
```

If your system uses `pip3`, use:

```bash
pip3 install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
```

#### 4. Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

A browser window should open.

Open:

```text
Hospital_readmission_prediction.ipynb
```

#### 5. Upload the Dataset

Download the Diabetes 130-US Hospitals dataset and keep it in ZIP format.

Run the dataset upload cell and select the ZIP file when prompted.

#### 6. Run the Notebook

Run the cells from top to bottom.

You can either run each cell individually or use:

```text
Run → Run All Cells
```

The notebook will then perform the data preparation, model training, evaluation, visualization, and sample prediction.

---

## Project Structure

```text
Hospital-Readmission-Prediction/
│
├── Hospital_readmission_prediction.ipynb
└── README.md
```

The dataset is not included in the repository and needs to be uploaded separately when running the notebook.

---

## Limitations

There are several limitations to this project:

- The dataset contains historical hospital records from 1999–2008.
- The model has not been clinically validated.
- The dataset represents a specific population and historical healthcare setting.
- Model performance depends on the information available in the dataset.
- The positive class is relatively difficult to predict.
- The relatively low precision means that many positive predictions are false positives.
- The model should not be used for clinical decision-making.

---

## Conclusion

This project was built to explore a complete machine learning classification workflow using a real-world healthcare dataset.

The main parts of the project include:

```text
Data Cleaning
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Logistic Regression
    ↓
Cross-Validation
    ↓
XGBoost
    ↓
Class Imbalance Handling
    ↓
Threshold Tuning
    ↓
Model Evaluation
    ↓
Visualization
    ↓
Sample Prediction
```

The project also shows why multiple evaluation metrics are important when working with an imbalanced classification problem.

Accuracy alone does not tell the complete story, so precision, recall, F1-score, and ROC-AUC are also considered.

---

## Disclaimer

> **This project is for educational and machine learning practice purposes only. It is not intended for clinical diagnosis, treatment, or real-world medical decision-making.**
