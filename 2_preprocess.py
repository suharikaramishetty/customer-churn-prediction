"""
FILE 2: Data Preprocessing Pipeline
- Cleans missing values
- Engineers 30-day behavioral features
- Label encodes categoricals
- Applies SMOTE to fix class imbalance
- Saves processed data

Command: python 2_preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
INPUT_PATH  = "data/telecom_customers.csv"
OUTPUT_DIR  = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────
print("="*55)
print("STEP 1: Loading Data")
print("="*55)

df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ─────────────────────────────────────────
# STEP 2: Drop ID Column (not a feature)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Dropping Non-Feature Columns")
print("="*55)

df.drop(columns=["customer_id"], inplace=True)
print("Dropped: customer_id")

# ─────────────────────────────────────────
# STEP 3: Handle Missing Values
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: Handling Missing Values")
print("="*55)

numerical_cols   = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target from numerical
if "churn" in numerical_cols:
    numerical_cols.remove("churn")

print(f"Numerical columns  : {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Impute numerical → median (robust to outliers)
for col in numerical_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {missing} missing in '{col}' with median={median_val:.2f}")

# Impute categorical → mode
for col in categorical_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled {missing} missing in '{col}' with mode='{mode_val}'")

print(f"\nMissing after imputation: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# STEP 4: Feature Engineering
# (30-day behavioral window features)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Feature Engineering")
print("="*55)

# 1. Call drop ratio: how much have calls dropped recently?
df["call_drop_ratio"] = (
    (df["avg_calls_alltime"] - df["avg_calls_30d"])
    / (df["avg_calls_alltime"] + 1)   # +1 avoids division by zero
)

# 2. Data usage drop ratio
df["data_drop_ratio"] = (
    (df["avg_data_usage_alltime"] - df["avg_data_usage_30d"])
    / (df["avg_data_usage_alltime"] + 0.001)
)

# 3. Complaint spike: recent complaints vs historical average per month
df["complaint_spike"] = (
    df["num_complaints_30d"]
    / (df["num_complaints_alltime"] / (df["tenure_months"] + 1) + 0.001)
)

# 4. Monthly charge per tenure month (value perception)
df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)

# 5. High risk flag: combination of complaints + late payments
df["risk_score"] = (
    df["num_complaints_30d"] * 2
    + df["late_payments_30d"] * 1.5
    + df["call_drop_ratio"] * 3
)

print("New features created:")
new_features = ["call_drop_ratio", "data_drop_ratio", "complaint_spike",
                "charge_per_tenure", "risk_score"]
for f in new_features:
    print(f"  + {f}")

# ─────────────────────────────────────────
# STEP 5: Label Encoding
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 5: Label Encoding Categorical Columns")
print("="*55)

le = LabelEncoder()
encoding_map = {}

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    encoding_map[col] = dict(zip(le.classes_,
                                  le.transform(le.classes_)))
    print(f"  '{col}' → {encoding_map[col]}")

# ─────────────────────────────────────────
# STEP 6: Separate Features and Target
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 6: Separating Features (X) and Target (y)")
print("="*55)

X = df.drop(columns=["churn"])
y = df["churn"]

print(f"Feature matrix shape : {X.shape}")
print(f"Target distribution  :\n{y.value_counts()}")
print(f"Churn rate before SMOTE: {y.mean()*100:.1f}%")

# ─────────────────────────────────────────
# STEP 7: Train/Test Split
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 7: Train/Test Split (80/20)")
print("="*55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y     # Keeps churn ratio same in both splits
)

print(f"Training set  : {X_train.shape[0]} records")
print(f"Test set      : {X_test.shape[0]} records")
print(f"Train churn % : {y_train.mean()*100:.1f}%")
print(f"Test churn %  : {y_test.mean()*100:.1f}%")

# ─────────────────────────────────────────
# STEP 8: SMOTE Oversampling (on TRAIN only!)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 8: Applying SMOTE on Training Data")
print("="*55)
print("⚠️  SMOTE is ONLY applied to training data.")
print("    Test data is left as-is for honest evaluation.")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nBefore SMOTE → Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")
print(f"After SMOTE  → Class 0: {(y_train_bal==0).sum()} | Class 1: {(y_train_bal==1).sum()}")

# ─────────────────────────────────────────
# STEP 9: Save Processed Data
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 9: Saving Processed Data")
print("="*55)

import joblib

joblib.dump(X_train_bal, "data/X_train.pkl")
joblib.dump(y_train_bal, "data/y_train.pkl")
joblib.dump(X_test,      "data/X_test.pkl")
joblib.dump(y_test,      "data/y_test.pkl")
joblib.dump(list(X.columns), "data/feature_names.pkl")

# Also save full processed dataframe for flagging later
df.to_csv("data/processed_customers.csv", index=False)

print("Saved:")
print("  data/X_train.pkl")
print("  data/y_train.pkl")
print("  data/X_test.pkl")
print("  data/y_test.pkl")
print("  data/feature_names.pkl")
print("  data/processed_customers.csv")

# ─────────────────────────────────────────
# PIPELINE ACCURACY REPORT
# ─────────────────────────────────────────
total = len(pd.read_csv(INPUT_PATH))
processed = len(df)
pipeline_accuracy = (processed / total) * 100
print(f"\n✅ Pipeline Accuracy: {pipeline_accuracy:.1f}%")
print(f"   ({processed}/{total} records successfully processed)")
