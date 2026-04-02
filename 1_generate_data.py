"""
FILE 1: Generate Synthetic Telecom Customer Data
Run this first to create the dataset.
Command: python 1_generate_data.py
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
np.random.seed(42)          # Makes results reproducible
NUM_CUSTOMERS = 7000
OUTPUT_PATH = "data/telecom_customers.csv"

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Generate Base Customer Features
# ─────────────────────────────────────────

customer_ids = [f"CUST_{i:05d}" for i in range(1, NUM_CUSTOMERS + 1)]

# Demographics
age              = np.random.randint(18, 70, NUM_CUSTOMERS)
gender           = np.random.choice(["Male", "Female"], NUM_CUSTOMERS)
location         = np.random.choice(["Urban", "Suburban", "Rural"], NUM_CUSTOMERS,
                                     p=[0.5, 0.3, 0.2])

# Account info
tenure_months    = np.random.randint(1, 72, NUM_CUSTOMERS)   # 1 to 72 months
contract_type    = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    NUM_CUSTOMERS, p=[0.55, 0.25, 0.20]
)
monthly_charges  = np.round(np.random.uniform(300, 2000, NUM_CUSTOMERS), 2)
payment_method   = np.random.choice(
    ["Credit Card", "Bank Transfer", "UPI", "Cash"],
    NUM_CUSTOMERS, p=[0.3, 0.3, 0.3, 0.1]
)

# Usage behavior (last 30 days)
avg_calls_30d       = np.random.randint(0, 200, NUM_CUSTOMERS)
avg_data_usage_30d  = np.round(np.random.uniform(0, 30, NUM_CUSTOMERS), 2)  # GB
avg_sms_30d         = np.random.randint(0, 100, NUM_CUSTOMERS)
num_complaints_30d  = np.random.poisson(0.5, NUM_CUSTOMERS)   # avg 0.5 complaints/month
late_payments_30d   = np.random.randint(0, 4, NUM_CUSTOMERS)

# Historical averages (all-time)
avg_calls_alltime      = np.random.randint(50, 200, NUM_CUSTOMERS)
avg_data_usage_alltime = np.round(np.random.uniform(5, 30, NUM_CUSTOMERS), 2)
num_complaints_alltime = np.random.randint(0, 10, NUM_CUSTOMERS)

# ─────────────────────────────────────────
# STEP 2: Engineer Churn Probability
# (We simulate realistic churn patterns)
# ─────────────────────────────────────────

# Base churn probability starts at 15%
churn_prob = np.full(NUM_CUSTOMERS, 0.15)

# Month-to-month contracts churn more
churn_prob += np.where(contract_type == "Month-to-month", 0.20, 0.0)
churn_prob += np.where(contract_type == "One year",       0.05, 0.0)

# Short tenure = higher churn risk
churn_prob += np.where(tenure_months < 12, 0.15, 0.0)
churn_prob += np.where(tenure_months < 6,  0.10, 0.0)

# High monthly charges = higher churn risk
churn_prob += np.where(monthly_charges > 1500, 0.10, 0.0)

# Recent complaints spike churn
churn_prob += num_complaints_30d * 0.10

# Late payments = churn signal
churn_prob += late_payments_30d * 0.08

# Drop in data usage compared to historical = churn signal
usage_drop = avg_data_usage_alltime - avg_data_usage_30d
churn_prob += np.where(usage_drop > 10, 0.10, 0.0)

# Drop in calls
call_drop = avg_calls_alltime - avg_calls_30d
churn_prob += np.where(call_drop > 50, 0.08, 0.0)

# Clip probability to [0, 1]
churn_prob = np.clip(churn_prob, 0, 1)

# Generate binary churn label
churn = (np.random.rand(NUM_CUSTOMERS) < churn_prob).astype(int)

# ─────────────────────────────────────────
# STEP 3: Introduce Realistic Noise
# ─────────────────────────────────────────

# Add ~3% random missing values in some columns
for col_arr in [avg_calls_30d, avg_data_usage_30d, num_complaints_30d]:
    mask = np.random.rand(NUM_CUSTOMERS) < 0.03
    col_arr = col_arr.astype(float)
    col_arr[mask] = np.nan

# ─────────────────────────────────────────
# STEP 4: Build DataFrame and Save
# ─────────────────────────────────────────

df = pd.DataFrame({
    "customer_id"            : customer_ids,
    "age"                    : age,
    "gender"                 : gender,
    "location"               : location,
    "tenure_months"          : tenure_months,
    "contract_type"          : contract_type,
    "monthly_charges"        : monthly_charges,
    "payment_method"         : payment_method,
    "avg_calls_30d"          : avg_calls_30d,
    "avg_data_usage_30d"     : avg_data_usage_30d,
    "avg_sms_30d"            : avg_sms_30d,
    "num_complaints_30d"     : num_complaints_30d,
    "late_payments_30d"      : late_payments_30d,
    "avg_calls_alltime"      : avg_calls_alltime,
    "avg_data_usage_alltime" : avg_data_usage_alltime,
    "num_complaints_alltime" : num_complaints_alltime,
    "churn"                  : churn
})

df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset created: {OUTPUT_PATH}")
print(f"   Total records  : {len(df)}")
print(f"   Churn rate     : {df['churn'].mean()*100:.1f}%")
print(f"   Churners       : {df['churn'].sum()}")
print(f"   Non-churners   : {(df['churn']==0).sum()}")
print(f"\nFirst 5 rows:")
print(df.head())
