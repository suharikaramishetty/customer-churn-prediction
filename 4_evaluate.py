"""
FILE 4: Model Evaluation
- Tests both models on unseen test data
- Generates ROC-AUC, Confusion Matrix, Classification Report
- Plots ROC curves
- Compares ML vs Rule-Based false positive rate

Command: python 4_evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Load Data & Models
# ─────────────────────────────────────────
print("="*55)
print("STEP 1: Loading Data & Models")
print("="*55)

X_test  = joblib.load("data/X_test.pkl")
y_test  = joblib.load("data/y_test.pkl")
lr_model = joblib.load("models/logistic_regression.pkl")
rf_model = joblib.load("models/random_forest.pkl")

print(f"Test samples : {len(X_test)}")
print(f"Churners     : {y_test.sum()}")
print(f"Non-Churners : {(y_test==0).sum()}")

# ─────────────────────────────────────────
# STEP 2: Generate Predictions
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Generating Predictions")
print("="*55)

# Probability scores (for ROC-AUC)
lr_proba = lr_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Hybrid score = average of both models
hybrid_proba = (lr_proba + rf_proba) / 2

# Binary predictions at threshold 0.5
lr_pred     = (lr_proba     >= 0.5).astype(int)
rf_pred     = (rf_proba     >= 0.5).astype(int)
hybrid_pred = (hybrid_proba >= 0.5).astype(int)

print("Predictions generated for:")
print("  - Logistic Regression")
print("  - Random Forest")
print("  - Hybrid (average of both)")

# ─────────────────────────────────────────
# STEP 3: ROC-AUC Scores
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: ROC-AUC Scores")
print("="*55)

lr_auc     = roc_auc_score(y_test, lr_proba)
rf_auc     = roc_auc_score(y_test, rf_proba)
hybrid_auc = roc_auc_score(y_test, hybrid_proba)

print(f"\n  Logistic Regression AUC : {lr_auc:.4f}")
print(f"  Random Forest AUC       : {rf_auc:.4f}")
print(f"  Hybrid AUC              : {hybrid_auc:.4f}")

# ─────────────────────────────────────────
# STEP 4: Confusion Matrices
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Confusion Matrices")
print("="*55)

def print_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) * 100
    print(f"\n  {name}")
    print(f"    True Negatives  (Correct Non-churn) : {tn}")
    print(f"    False Positives (Wrong alarm)        : {fp}  ← lower is better")
    print(f"    False Negatives (Missed churners)    : {fn}")
    print(f"    True Positives  (Caught churners)    : {tp}")
    print(f"    False Positive Rate: {fpr:.1f}%")
    return cm

cm_lr     = print_confusion("Logistic Regression", y_test, lr_pred)
cm_rf     = print_confusion("Random Forest",       y_test, rf_pred)
cm_hybrid = print_confusion("Hybrid Model",        y_test, hybrid_pred)

# ─────────────────────────────────────────
# STEP 5: Classification Reports
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 5: Classification Reports")
print("="*55)

print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_pred,
      target_names=["No Churn", "Churn"]))

print("\n--- Random Forest ---")
print(classification_report(y_test, rf_pred,
      target_names=["No Churn", "Churn"]))

print("\n--- Hybrid Model ---")
print(classification_report(y_test, hybrid_pred,
      target_names=["No Churn", "Churn"]))

# ─────────────────────────────────────────
# STEP 6: Rule-Based vs ML Comparison
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 6: Rule-Based vs ML Comparison")
print("="*55)

# Rule-based: flag anyone with ≥1 complaint OR ≥2 late payments in 30 days
X_test_df = X_test.copy()
feature_names = joblib.load("data/feature_names.pkl")

rule_pred = (
    (X_test_df["num_complaints_30d"] >= 1) |
    (X_test_df["late_payments_30d"]  >= 2)
).astype(int)

rule_cm = confusion_matrix(y_test, rule_pred)
_, fp_rule, _, _ = rule_cm.ravel()
_, fp_ml,   _, _ = cm_hybrid.ravel()

fp_reduction = (fp_rule - fp_ml) / fp_rule * 100

print(f"\n  Rule-Based False Positives : {fp_rule}")
print(f"  ML Hybrid False Positives  : {fp_ml}")
print(f"  Reduction in FP            : {fp_reduction:.1f}%")
print(f"\n  Rule-Based Total Alerts    : {rule_pred.sum()} (from {len(y_test)} test records)")
print(f"  ML Total Flags             : {hybrid_pred.sum()} (from {len(y_test)} test records)")

# Scale to full 7000 records
scale_factor = 7000 / len(y_test)
print(f"\n  Scaled to 7,000 records:")
print(f"    Rule-Based alerts : ~{int(rule_pred.sum()*scale_factor)}")
print(f"    ML flags          : ~{int(hybrid_pred.sum()*scale_factor)}")

# ─────────────────────────────────────────
# STEP 7: Plot Everything
# ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 7: Generating Evaluation Charts")
print("="*55)

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Churn Prediction — Model Evaluation Dashboard",
             fontsize=18, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Chart 1: ROC Curves ──
ax1 = fig.add_subplot(gs[0, :2])
for name, proba, auc_val, color, ls in [
    ("Logistic Regression", lr_proba,     lr_auc,     "#2196F3", "--"),
    ("Random Forest",       rf_proba,     rf_auc,     "#4CAF50", "-."),
    ("Hybrid Model",        hybrid_proba, hybrid_auc, "#FF5722", "-"),
]:
    fpr_c, tpr_c, _ = roc_curve(y_test, proba)
    ax1.plot(fpr_c, tpr_c, label=f"{name} (AUC={auc_val:.3f})",
             linewidth=2.5, linestyle=ls, color=color)

ax1.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.4, label="Random Classifier")
ax1.fill_between(*roc_curve(y_test, hybrid_proba)[:2], alpha=0.08, color="#FF5722")
ax1.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

# ── Chart 2: AUC Bar Chart ──
ax2 = fig.add_subplot(gs[0, 2])
models_names = ["Logistic\nRegression", "Random\nForest", "Hybrid"]
auc_vals     = [lr_auc, rf_auc, hybrid_auc]
colors       = ["#2196F3", "#4CAF50", "#FF5722"]
bars = ax2.bar(models_names, auc_vals, color=colors, edgecolor="white",
               linewidth=1.5, width=0.55)
for bar, val in zip(bars, auc_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylim(0.5, 1.0)
ax2.set_title("ROC-AUC Comparison", fontsize=13, fontweight="bold")
ax2.set_ylabel("ROC-AUC Score"); ax2.grid(axis="y", alpha=0.3)

# ── Charts 3-5: Confusion Matrices ──
for idx, (name, cm, ax_pos) in enumerate([
    ("Logistic Regression", cm_lr,     gs[1, 0]),
    ("Random Forest",       cm_rf,     gs[1, 1]),
    ("Hybrid Model",        cm_hybrid, gs[1, 2]),
]):
    ax = fig.add_subplot(ax_pos)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                linewidths=0.5, cbar=False)
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# ── Chart 6: False Positive Comparison ──
ax6 = fig.add_subplot(gs[2, 0])
fp_vals   = [fp_rule, fp_ml]
fp_labels = ["Rule-Based", "ML Hybrid"]
fp_colors = ["#F44336", "#4CAF50"]
bars = ax6.bar(fp_labels, fp_vals, color=fp_colors, edgecolor="white",
               linewidth=1.5, width=0.5)
for bar, val in zip(bars, fp_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax6.set_title(f"False Positives\n(↓{fp_reduction:.0f}% reduction with ML)",
              fontsize=11, fontweight="bold")
ax6.set_ylabel("False Positive Count"); ax6.grid(axis="y", alpha=0.3)

# ── Chart 7: Feature Importance ──
ax7 = fig.add_subplot(gs[2, 1:])
feature_names = joblib.load("data/feature_names.pkl")
importances   = rf_model.feature_importances_
feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True).tail(10)
colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
ax7.barh(feat_df["feature"], feat_df["importance"],
         color=colors_bar, edgecolor="white")
ax7.set_title("Top 10 Feature Importances (Random Forest)",
              fontsize=11, fontweight="bold")
ax7.set_xlabel("Importance Score"); ax7.grid(axis="x", alpha=0.3)

plt.savefig("outputs/evaluation_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.show()
print("\n✅ Dashboard saved: outputs/evaluation_dashboard.png")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*55)
print("FINAL EVALUATION SUMMARY")
print("="*55)
print(f"  Best Model       : Hybrid (LR + RF)")
print(f"  ROC-AUC          : {hybrid_auc:.4f}")
print(f"  FP Reduction     : {fp_reduction:.1f}% vs rule-based")
print(f"  High-risk flags  : ~{int(hybrid_pred.sum()*scale_factor)} (from 7,000 records)")
