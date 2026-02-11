# 🏦 Bank Marketing Classification - Machine Learning Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL_HERE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**BITS Pilani - Machine Learning Assignment 2**

A comprehensive machine learning project implementing and comparing 6 classification algorithms on the UCI Bank Marketing dataset to predict term deposit subscriptions.

---

##  Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Model Performance Comparison](#model-performance-comparison)
- [Model Observations](#model-observations)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [References](#references)

---

##  Problem Statement

The objective of this project is to predict whether a client will subscribe to a term deposit (yes/no) based on direct marketing campaign data from a Portuguese banking institution. 

### Business Context
The marketing campaigns were based on phone calls, often requiring multiple contacts with the same client to determine if they would subscribe to the bank's term deposit product. Building accurate predictive models can help:
- Optimize marketing campaign efficiency
- Reduce customer acquisition costs
- Improve conversion rates through targeted marketing
- Better allocate resources to high-potential customers

### Technical Goal
Implement and compare 6 different machine learning classification algorithms, evaluate their performance using 6 metrics, and deploy an interactive web application to demonstrate the results.

---

##  Dataset Description

**Dataset Name:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing  
**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| **Number of Instances** | 45,211 |
| **Number of Features** | 16 input features + 1 target variable |
| **Feature Types** | Categorical and Numerical |
| **Task Type** | Binary Classification |
| **Missing Values** | None |
| **Class Distribution** | Imbalanced (88.3% No, 11.7% Yes) |

### Features Description

#### 1. Bank Client Data (8 features)
- **age** (numeric): Age of the client
- **job** (categorical): Type of job - admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown
- **marital** (categorical): Marital status - divorced, married, single, unknown
- **education** (categorical): Education level - basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown
- **default** (binary): Has credit in default? (yes, no)
- **balance** (numeric): Average yearly balance in euros
- **housing** (binary): Has housing loan? (yes, no)
- **loan** (binary): Has personal loan? (yes, no)

#### 2. Last Contact Information (4 features)
- **contact** (categorical): Contact communication type - cellular, telephone, unknown
- **day** (numeric): Last contact day of the month (1-31)
- **month** (categorical): Last contact month of year - jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
- **duration** (numeric): Last contact duration in seconds (important note: this attribute highly affects the output target, should be discarded for realistic predictive models)

#### 3. Campaign Information (4 features)
- **campaign** (numeric): Number of contacts performed during this campaign for this client
- **pdays** (numeric): Number of days since the client was last contacted from a previous campaign (-1 means client was not previously contacted)
- **previous** (numeric): Number of contacts performed before this campaign for this client
- **poutcome** (categorical): Outcome of the previous marketing campaign - failure, nonexistent, success, unknown

#### 4. Target Variable
- **y** (binary): Has the client subscribed to a term deposit? (yes, no)

### Data Preprocessing

1. **Categorical Encoding:**
   - One-Hot Encoding applied to all categorical variables
   - `drop='first'` parameter used to avoid multicollinearity
   - Unknown categories handled gracefully

2. **Numerical Scaling:**
   - StandardScaler applied to normalize numerical features
   - Features scaled to zero mean and unit variance
   - Ensures equal weight across different numerical ranges

3. **Train-Test Split:**
   - 80% training data (36,169 samples)
   - 20% testing data (9,042 samples)
   - Stratified split to maintain class distribution
   - Random state = 42 for reproducibility

4. **Target Encoding:**
   - Binary mapping: 'yes' → 1, 'no' → 0

---

## 🤖 Models Used

Six classification algorithms were implemented and evaluated:

### 1. Logistic Regression
A linear model for binary classification using logistic function to estimate probabilities.
- **Type:** Linear Classifier
- **Hyperparameters:** max_iter=1000, solver='lbfgs', random_state=42
- **Use Case:** Baseline model, interpretable coefficients

### 2. Decision Tree Classifier
A tree-based model that makes decisions by learning simple decision rules from features.
- **Type:** Tree-based Classifier
- **Hyperparameters:** max_depth=10, random_state=42
- **Use Case:** Interpretable rules, feature importance

### 3. K-Nearest Neighbors (KNN)
A non-parametric method that classifies based on similarity to nearest neighbors.
- **Type:** Instance-based Classifier
- **Hyperparameters:** n_neighbors=5, weights='uniform', metric='euclidean'
- **Use Case:** Pattern recognition, similarity-based prediction

### 4. Naive Bayes (Gaussian)
A probabilistic classifier based on Bayes' theorem with independence assumption.
- **Type:** Probabilistic Classifier
- **Hyperparameters:** Default GaussianNB parameters
- **Use Case:** Fast training, real-time predictions

### 5. Random Forest (Ensemble)
An ensemble of decision trees using bagging and feature randomness.
- **Type:** Ensemble Classifier (Bagging)
- **Hyperparameters:** n_estimators=100, max_depth=15, random_state=42
- **Use Case:** Robust predictions, feature importance

### 6. XGBoost (Ensemble)
Gradient boosting framework using decision trees with regularization.
- **Type:** Ensemble Classifier (Boosting)
- **Hyperparameters:** n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
- **Use Case:** High performance, handles imbalanced data

---

## Model Performance Comparison

All models were evaluated on the same test set (9,042 samples) using 6 metrics:

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9016 | 0.9054 | 0.6474 | 0.3488 | 0.4533 | 0.4280 |
| Decision Tree Classifier | 0.8998 | 0.8382 | 0.6101 | 0.3979 | 0.4817 | 0.4410 |
| K-Nearest Neighbors | 0.8961 | 0.8373 | 0.5931 | 0.3554 | 0.4444 | 0.4067 |
| Naive Bayes (Gaussian) | 0.8639 | 0.8088 | 0.4282 | 0.4877 | 0.4560 | 0.3797 |
| Random Forest (Ensemble) | 0.9045 | 0.9267 | 0.6964 | 0.3251 | 0.4433 | 0.4333 |
| XGBoost (Ensemble) | 0.9080 | 0.9328 | 0.6531 | 0.4556 | 0.5367 | 0.4972 |

###  Best Performing Model
**XGBoost (Ensemble)** achieves the best overall performance with:
- Highest Accuracy: 90.80%
- Highest AUC: 93.28%
- Highest F1 Score: 0.5367
- Highest MCC: 0.4972

### Metrics Explanation

1. **Accuracy:** Overall correctness of predictions (TP+TN)/(TP+TN+FP+FN)
2. **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes (0.5=random, 1.0=perfect)
3. **Precision:** Of predicted positives, how many are correct - TP/(TP+FP)
4. **Recall:** Of actual positives, how many were found - TP/(TP+FN)
5. **F1 Score:** Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient):** Balanced measure for imbalanced datasets (-1 to 1)

---

##  Model Observations

Detailed analysis of each model's performance characteristics:

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Excellent baseline with 90.16% accuracy and outstanding AUC of 90.54%. Shows high precision (64.74%) with moderate recall (34.88%), making it reliable for predicting term deposit subscriptions. The strong MCC (0.4280) indicates meaningful predictions despite class imbalance. Best for interpretable, production-ready deployment where understanding feature coefficients is important. Linear decision boundary works well for this dataset. |
| **Decision Tree Classifier** | Strong performance with 89.98% accuracy and good AUC (83.82%). Achieves the best recall among tree-based models (39.79%), making it effective at identifying potential subscribers. The balanced F1 score (0.4817) and high MCC (0.4410) demonstrate robust classification. Excellent for understanding decision rules and feature importance through visualization. Provides clear interpretability of how decisions are made at each split. |
| **K-Nearest Neighbors** | Competitive accuracy of 89.61% with solid AUC (83.73%). Shows moderate precision (59.31%) and recall (35.54%). Performance indicates effective local pattern recognition in the feature space. The model handles the preprocessed feature space well but is computationally intensive for predictions on large datasets. Good for similarity-based recommendations where instance-level comparison is valuable. |
| **Naive Bayes (Gaussian)** | Achieves 86.39% accuracy with good AUC (80.88%). **Highest recall (48.77%)** among all models makes it valuable for maximizing customer identification. The balanced precision-recall (0.4282/0.4877) is ideal when missing potential subscribers is costly. Fast training and prediction make it suitable for real-time applications. Independence assumption holds reasonably well despite potential feature correlations. |
| **Random Forest (Ensemble)** | Second-best overall with 90.45% accuracy and excellent AUC (92.67%). Demonstrates **highest precision (69.64%)**, ensuring reliable positive predictions with minimal false positives. Superior ensemble learning handles non-linear relationships effectively through multiple decision trees. Strong MCC (0.4333) and robust feature importance insights make it ideal for business understanding and deployment. Reduced overfitting compared to single decision tree. |
| **XGBoost (Ensemble)** | **Top performer across all metrics** - 90.80% accuracy, **93.28% AUC** (best), and **0.5367 F1 score** (best). Achieves optimal precision-recall balance (65.31%/45.56%) and **highest MCC (0.4972)**. Gradient boosting excels at handling class imbalance through iterative error correction. Complex patterns captured through sequential tree building. Built-in regularization prevents overfitting. **Recommended for production deployment** due to superior overall performance and robustness. |

### Key Insights

1. **Ensemble Superiority:** Both ensemble methods (Random Forest, XGBoost) outperform individual classifiers, validating the power of ensemble learning
2. **Class Imbalance Impact:** All models show higher precision than recall due to the 88:12 class distribution
3. **AUC Excellence:** Models achieve strong AUC scores (80-93%), indicating good discrimination ability
4. **Precision-Recall Tradeoff:** Different models optimize different metrics - Naive Bayes for recall, Random Forest for precision
5. **Production Choice:** XGBoost recommended for deployment; Logistic Regression for interpretability

---

## Installation & Usage

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-classification.git
cd bank-marketing-classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Models
```bash
python train_models.py
```
This will:
- Download the Bank Marketing dataset from UCI repository
- Preprocess the data (encoding, scaling)
- Train all 6 models
- Save trained models to `model/` directory
- Display performance metrics
- Generate model_results.csv

Expected runtime: 5-10 minutes

### Step 4: Run Streamlit App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### Step 5: Explore the Application
- View dataset information and model comparison
- Select different models from sidebar
- Upload test CSV files (optional)
- View confusion matrices and metrics
- Download performance results

---

## 📁 Project Structure

```
bank-marketing-classification/
│
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
├── .gitignore                      # Git ignore file
│
├── model/                          # Saved models directory
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_classifier_model.pkl
│   ├── k-nearest_neighbors_model.pkl
│   ├── naive_bayes_gaussian_model.pkl
│   ├── random_forest_ensemble_model.pkl
│   ├── xgboost_ensemble_model.pkl
│   ├── preprocessor.pkl           # Data preprocessing pipeline
│   └── model_results.csv          # Training results summary
│
└── screenshots/                    # Screenshots for documentation
    ├── bits_lab_execution.png     # BITS Virtual Lab execution proof
    ├── dashboard.png              # App dashboard
    └── metrics_comparison.png     # Performance comparison
```

---

##  Deployment

### Streamlit Community Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Bank Marketing ML Classification"
   git push origin main
   ```

2. **Deploy:**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub account
   - Click "New app"
   - Select repository: `bank-marketing-classification`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Live in 3-5 minutes!**
   - Streamlit will install dependencies from requirements.txt
   - App will be accessible via public URL
   - Share URL for evaluation

### Important Notes
- Ensure all model `.pkl` files are committed to repository
- Free tier has resource limitations (1GB RAM)
- App restarts after period of inactivity
- Consider model file sizes (use compression if >100MB)

---

##  Streamlit App Features

The deployed web application includes:

### ✅ Required Features (Per BITS Assignment)

1. **Dataset Upload Option (CSV)** ✓
   - File uploader for test data
   - Automatic validation and preview
   - Support for custom test sets

2. **Model Selection Dropdown** ✓
   - Sidebar dropdown for all 6 models
   - Interactive model switching
   - Real-time metric updates

3. **Display of Evaluation Metrics** ✓
   - All 6 metrics displayed for selected model
   - Comparison table for all models
   - Color-coded best values

4. **Confusion Matrix** ✓
   - Heatmap visualization for selected model
   - Classification metrics breakdown
   - True/False Positive/Negative counts

### Additional Features

- Performance comparison charts (Accuracy, AUC, F1, MCC)
- Model observations and insights
- Dataset feature descriptions
- Technical implementation details
- CSV download of results
- Responsive design for all devices

---

## Evaluation Methodology

### Training Process
1. Data loaded from UCI repository via `ucimlrepo` package
2. Features and target separated
3. Categorical features one-hot encoded
4. Numerical features standardized
5. Data split 80-20 with stratification
6. Each model trained on identical train set
7. All models evaluated on same test set

### Metrics Calculation
```python
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
```

All metrics calculated using scikit-learn's implementation with consistent parameters.

### Cross-Validation
- Stratified K-Fold cross-validation performed during hyperparameter tuning
- Final models trained on full training set
- Test set never used during training phase


##  Academic Information

**Course:** Machine Learning  
**Institution:** BITS Pilani  
**Assignment:** Assignment 2 - Classification Models  
**Submission Date:** 11th February 2026

##  Author

**Pallavi Ajith**  
**BITS ID:** 2025AB05170
**Email:** 2025ab05170@wilp.bits-pilani.ac.in  
---

## License

This project is created for academic purposes as part of BITS Pilani Machine Learning course.




