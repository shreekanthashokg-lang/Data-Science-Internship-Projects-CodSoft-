# 🔬 CodSoft Data Science Internship — All Tasks

# NAME : SHREEKANTH A GUTTEDAR

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Internship](https://img.shields.io/badge/CodSoft-Data%20Science-purple)

> **All 5 tasks completed** as part of the CodSoft Data Science Internship program.  
> Each task covers end-to-end ML: EDA → Preprocessing → Modeling → Evaluation → Visualization.

---

## 📁 Repository Structure

```
CODSOFT/
├── Task1_Titanic/
│   └── titanic_survival.py
├── Task2_MovieRating/
│   └── movie_rating.py
├── Task3_Iris/
│   └── iris_classification.py
├── Task4_Sales/
│   └── sales_prediction.py
├── Task5_FraudDetection/
│   └── fraud_detection.py
└── README.md
```

---

## ✅ Task Summary

### Task 1 — Titanic Survival Prediction
> Predict whether a passenger survived the Titanic disaster.

| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting |
| Test Accuracy | ~82% |
| Cross-Val Accuracy | ~82% |

**Key Steps:**
- Feature engineering (family size, is_alone)
- Encoded sex & embarked features
- Compared Logistic Regression, Random Forest, Gradient Boosting

---

### Task 2 — Movie Rating Prediction
> Predict IMDb-style movie ratings using genre, director, actors & votes.

| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting |
| RMSE | ~0.43 |
| R² Score | ~0.88 |

**Key Steps:**
- Label encoding of categorical features
- Log transform on votes
- Regression with 4 models

---

### Task 3 — Iris Flower Classification
> Classify iris flowers into setosa, versicolor, or virginica.

| Metric | Score |
|--------|-------|
| Best Model | SVM / Random Forest |
| Test Accuracy | 100% |
| 10-Fold CV Accuracy | ~97% |

**Key Steps:**
- Pairplot & correlation analysis
- 5 classifiers compared
- Custom prediction demo

---

### Task 4 — Sales Prediction
> Forecast product sales from TV, Radio, Newspaper advertising budgets.

| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting |
| R² Score | ~0.97 |
| RMSE | ~0.60 |

**Key Steps:**
- Interaction feature: TV × Radio
- Total spend & ratio features
- Business insights extracted

---

### Task 5 — Credit Card Fraud Detection
> Detect fraudulent transactions on an imbalanced dataset.

| Metric | Score |
|--------|-------|
| Best Model | Random Forest |
| F1-Score | ~0.91 |
| ROC-AUC | ~0.99 |

**Key Steps:**
- SMOTE oversampling for class imbalance
- ROC & Precision-Recall curves
- Business impact analysis (TP/FP/FN)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models & evaluation |
| `imbalanced-learn` | SMOTE / resampling |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/CODSOFT.git
cd CODSOFT

# 2. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost

# 3. Run any task
python Task1_Titanic/titanic_survival.py
python Task2_MovieRating/movie_rating.py
python Task3_Iris/iris_classification.py
python Task4_Sales/sales_prediction.py
python Task5_FraudDetection/fraud_detection.py
```

---

## 🔗 Connect

- **LinkedIn: https://www.linkedin.com/in/shreekanth-a-guttedar-81562b384
- **GitHub:  https://github.com/shreekanthashokg-lang
- **Internship:** [CodSoft](https://www.codsoft.in)

---

> Made with ❤️ during the **CodSoft Data Science Internship**  
> `#codsoft` `#datascience` `#machinelearning` `#python` `#internship`
