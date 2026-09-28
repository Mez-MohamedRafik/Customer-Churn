# 🏦 Bank Customer Churn Prediction & Retention Optimization

An end-to-end Machine Learning and Decision Optimization pipeline designed to identify at-risk bank customers and maximize retention outcomes. This project bridges **Operations Research (Applied Decision Theory)** and **Machine Learning Engineering** by combining SQL data wrangling, modular Scikit-Learn/XGBoost pipelines, and cost-sensitive threshold optimization.

---

## 📐 Core Engineering Frameworks

This project integrates two primary technical disciplines:

### 1. Operations Research (OR) & Decision Theory
* **Asymmetric Cost/Loss Function Optimization:** In churn management, the cost of a False Negative (losing a customer worth high Lifetime Value) heavily outweighs the cost of a False Positive (sending an automated retention email or offer to a retained customer).
* **Threshold Selection & Constraint Optimization:** Shifted classification decision boundaries ($t = 0.35$) on class-weighted probability outputs to maximize customer recall (~73–85%), treating customer retention as a constrained decision-making problem under uncertainty.
* **Business Metric Alignment:** Evaluated predictions through expected financial impact rather than raw predictive accuracy.

### 2. Machine Learning (ML) Engineering
* **Data Engineering & Relational Sanitization:** Conducted initial schema validation, deduplication, and feature integrity checks in **MySQL Workbench** before exporting `CleanedChurnModeling.csv`.
* **Leak-Proof Production Pipelines:** Utilized Scikit-Learn `Pipeline` and `ColumnTransformer` to enforce strict isolation between training and testing data splits during scaling (`StandardScaler`) and encoding (`OneHotEncoder`).
* **Class Imbalance Handling:** Implemented cost-sensitive gradient boosting using XGBoost (`scale_pos_weight = 3.9`) and Random Forest (`class_weight='balanced'`).

---

## 🛠️ Tech Stack

* **Data Wrangling & SQL:** MySQL Workbench
* **Language & Runtime:** Python 3.x, Google Colab
* **Data Manipulation:** Pandas, NumPy
* **ML Pipelines & Preprocessing:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
* **Predictive Models:** XGBoost (`XGBClassifier`), Random Forest (`RandomForestClassifier`), Logistic Regression
* **Evaluation & Optimization:** ROC-AUC, Precision-Recall Trade-offs, Confusion Matrices, Probability Threshold Tuning

---

## 📊 Model Evaluation & Trade-off Analysis

| Model | Primary Focus | ROC-AUC | Accuracy | Churn Precision | Churn Recall | Decision Impact |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Linear Baseline | 0.7687 | 71.13% | 38.52% | 69.32% | High False Positive rate |
| **Random Forest** | High Precision | 0.8545 | 85.78% | **76.31%** | 43.94% | Misses >56% of churners (High FN cost) |
| **XGBoost (Tuned, $t=0.35$)** | **Retention Optimization** | **0.8370** | **79.77%** | **50.26%** | **>72.77%** | **Minimizes customer loss (Optimal OR strategy)** |

---

## ⚙️ Workflow Architecture

1. **Relational Preprocessing (MySQL):** Deduplication, missing value handling, and distribution validation.
2. **Feature Engineering & Transformation:** Standardized numerical distributions and encoded categorical features via a unified `ColumnTransformer`.
3. **Supervised Learning & Probability Estimation:** Trained `XGBClassifier` with `scale_pos_weight` adjustment to address target class imbalance ($~80/20$ split).
4. **Decision Optimization:** Evaluated prediction probabilities across variable decision thresholds ($0.30 - 0.60$) to minimize expected churn loss.

---

---

## 🚀 Run in Google Colab

Click the badge below to open and execute the complete interactive notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mez-MohamedRafik/Customer-Churn/blob/main/notebooks/Customer_churn_RFC_pred.ipynb)

# Install dependencies
pip install -r requirements.txt
