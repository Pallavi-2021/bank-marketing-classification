"""
Bank Marketing Classification Model Training Script
====================================================
This script trains 6 different classification models on the Bank Marketing dataset
and saves them for use in the Streamlit application.

Models implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. LOAD AND PREPARE DATA
# ===========================

print("=" * 70)
print("BANK MARKETING CLASSIFICATION - MODEL TRAINING")
print("=" * 70)
print("\n Loading dataset...")

# Load the dataset using ucimlrepo
try:
    from ucimlrepo import fetch_ucirepo
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
except Exception as e:
    print(f"  Error loading from ucimlrepo: {e}")
    print("Please ensure you have the dataset CSV file or use: pip install ucimlrepo")
    exit(1)

print(f" Dataset loaded successfully!")
print(f"   - Total samples: {df.shape[0]}")
print(f"   - Total features: {df.shape[1] - 1}")
print(f"   - Target column: y")

# Display dataset info
print(f"\nDataset Information:")
print(df.info())
print(f"\n Target Distribution:")
print(df['y'].value_counts())
print(f"\n   Class Balance: {df['y'].value_counts(normalize=True) * 100}")

# ===========================
# 2. PREPROCESSING
# ===========================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

# Separate features and target
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

print(f"\n Feature columns: {list(X.columns)}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Create preprocessing pipeline
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

print(f"\n Preprocessing pipeline created")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Data split:")
print(f"   - Training samples: {X_train.shape[0]}")
print(f"   - Testing samples: {X_test.shape[0]}")

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nData preprocessed successfully")
print(f"   - Transformed feature dimensions: {X_train_processed.shape[1]}")

# Save the preprocessor
import os
os.makedirs('model', exist_ok=True)

with open('model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print(f"\n Preprocessor saved to model/preprocessor.pkl")

# ===========================
# 3. TRAIN MODELS
# ===========================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=10),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    "XGBoost (Ensemble)": XGBClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
}

# Dictionary to store results
results = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Train the model
    model.fit(X_train_processed, y_train)
    print(f" Model trained successfully")
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Get probability predictions if available
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_pred_proba = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.0
    
    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    })
    
    # Print metrics
    print(f"\n Performance Metrics:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • AUC Score: {auc:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • F1 Score:  {f1:.4f}")
    print(f"   • MCC Score: {mcc:.4f}")
    
    # Save the model
    model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_model.pkl"
    model_path = f'model/{model_filename}'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n Model saved to {model_path}")

# ===========================
# 4. RESULTS SUMMARY
# ===========================

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(f"\n{results_df.to_string(index=False)}")

# Save results to CSV
results_df.to_csv('model/model_results.csv', index=False)
print(f"\n Results saved to model/model_results.csv")

# Find best model
best_model_idx = results_df['F1 Score'].idxmax()
best_model = results_df.iloc[best_model_idx]

print(f"\n BEST MODEL (by F1 Score):")
print(f"   Model: {best_model['Model']}")
print(f"   F1 Score: {best_model['F1 Score']:.4f}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")
print(f"   AUC: {best_model['AUC']:.4f}")

print("\n" + "=" * 70)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print(f"\n Model files saved in './model/' directory")
print(f" You can now run the Streamlit app using: streamlit run app.py")
print("=" * 70)