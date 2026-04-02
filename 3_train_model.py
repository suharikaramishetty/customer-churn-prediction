"""
FILE 3: Model Training
- Trains Logistic Regression
- Trains Random Forest
- Cross-validates both
- Saves both models

Command: python 3_train_model.py
"""

import joblib
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# STEP 1: Load Preprocessed Data
# ─────────────────────────────────────────
print("="*55)
print("STEP 1: Loading Preprocessed Data")
print("="*55)

X_train = joblib.load("data/X_train.pkl")
y_train = joblib.load("data/y_train.pkl")

print(f"Training samples : {len(X_train)}")
print(f"Features         : {X_train.shape[1]}")

# ─────────────────────────────────────────
# STEP 2: Cross-Validation Setup
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Setting Up 5-Fold Cross Validation")
print("="*55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Using StratifiedKFold(n_splits=5)")
print("→ Data split into 5 folds, each fold used once as validation")
print("→ Stratified = churn ratio preserved in every fold")

# ─────────────────────────────────────────
# STEP 3: Train Logistic Regression
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: Training Logistic Regression")
print("="*55)

lr_model = LogisticRegression(
    C=1.0,              # Regularization (1.0 = balanced)
    max_iter=1000,      # More iterations for convergence
    random_state=42,
    solver="lbfgs"
)

print("Running 5-fold cross-validation on Logistic Regression...")
lr_cv_scores = cross_val_score(lr_model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"\nLR Cross-Validation ROC-AUC per fold:")
for i, score in enumerate(lr_cv_scores, 1):
    bar = "█" * int(score * 30)
    print(f"  Fold {i}: {score:.4f}  {bar}")

print(f"\n  Mean  : {lr_cv_scores.mean():.4f}")
print(f"  Std   : {lr_cv_scores.std():.4f}")

# Final fit on full training data
lr_model.fit(X_train, y_train)
print("\n✅ Logistic Regression trained on full training set")

# ─────────────────────────────────────────
# STEP 4: Train Random Forest
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Training Random Forest")
print("="*55)

rf_model = RandomForestClassifier(
    n_estimators=200,       # 200 decision trees
    max_depth=15,           # Each tree can go 15 levels deep
    min_samples_split=10,   # At least 10 samples to split a node
    min_samples_leaf=4,     # At least 4 samples at each leaf
    class_weight="balanced",# Handles any residual imbalance
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

print("Running 5-fold cross-validation on Random Forest...")
print("(This may take ~60 seconds...)")

rf_cv_scores = cross_val_score(rf_model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"\nRF Cross-Validation ROC-AUC per fold:")
for i, score in enumerate(rf_cv_scores, 1):
    bar = "█" * int(score * 30)
    print(f"  Fold {i}: {score:.4f}  {bar}")

print(f"\n  Mean  : {rf_cv_scores.mean():.4f}")
print(f"  Std   : {rf_cv_scores.std():.4f}")

# Final fit on full training data
rf_model.fit(X_train, y_train)
print("\n✅ Random Forest trained on full training set")

# ─────────────────────────────────────────
# STEP 5: Feature Importance (Random Forest)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 5: Top 10 Feature Importances (Random Forest)")
print("="*55)

feature_names = joblib.load("data/feature_names.pkl")
importances   = rf_model.feature_importances_

feat_imp = sorted(zip(feature_names, importances),
                  key=lambda x: x[1], reverse=True)

print(f"\n{'Feature':<30} {'Importance':>10}  Bar")
print("-" * 60)
for name, imp in feat_imp[:10]:
    bar = "█" * int(imp * 200)
    print(f"  {name:<28} {imp:>10.4f}  {bar}")

# ─────────────────────────────────────────
# STEP 6: Comparison Summary
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 6: Model Comparison")
print("="*55)

print(f"\n{'Model':<25} {'Mean AUC':>10} {'Std':>8}")
print("-" * 45)
print(f"  {'Logistic Regression':<23} {lr_cv_scores.mean():>10.4f} {lr_cv_scores.std():>8.4f}")
print(f"  {'Random Forest':<23} {rf_cv_scores.mean():>10.4f} {rf_cv_scores.std():>8.4f}")

# ─────────────────────────────────────────
# STEP 7: Save Both Models
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 7: Saving Models")
print("="*55)

import os
os.makedirs("models", exist_ok=True)

joblib.dump(lr_model, "models/logistic_regression.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")

print("✅ Saved: models/logistic_regression.pkl")
print("✅ Saved: models/random_forest.pkl")
