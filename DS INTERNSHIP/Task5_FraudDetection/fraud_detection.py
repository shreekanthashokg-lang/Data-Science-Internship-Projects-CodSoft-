# ============================================================
#  CODSOFT DATA SCIENCE INTERNSHIP — TASK 5
#  Credit Card Fraud Detection
#  Author : SHREEKANTH A GUTTEDAR
#  Dataset: creditcard.csv (Kaggle) — PCA-transformed features
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION")
print("=" * 60)

# ── 1. Generate Realistic Synthetic Dataset ───────────────────
# Mirrors the Kaggle creditcard.csv structure (PCA V1-V28 + Amount + Time)
np.random.seed(42)
n_legit  = 9000
n_fraud  = 300   # ~3.2% fraud rate (realistic imbalance)

def make_transactions(n, fraud=False):
    rows = {}
    for i in range(1, 29):
        if fraud:
            rows[f"V{i}"] = np.random.normal(0, 2, n)
        else:
            rows[f"V{i}"] = np.random.normal(0, 1, n)
    rows["Amount"] = np.abs(np.random.exponential(88 if not fraud else 200, n))
    rows["Time"]   = np.random.uniform(0, 172792, n)
    rows["Class"]  = int(fraud)
    return pd.DataFrame(rows)

df = pd.concat([make_transactions(n_legit, fraud=False),
                make_transactions(n_fraud,  fraud=True)],
               ignore_index=True).sample(frac=1, random_state=42)

print(f"\n✅ Dataset ready: {df.shape[0]} transactions")
print(f"   Legitimate : {(df['Class']==0).sum()}")
print(f"   Fraudulent : {(df['Class']==1).sum()}")
print(f"   Fraud Rate : {df['Class'].mean()*100:.2f}%")

# ── 2. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Credit Card Fraud — EDA", fontsize=15, fontweight="bold")

# Class imbalance
counts = df["Class"].value_counts()
axes[0].pie(counts, labels=["Legitimate","Fraudulent"],
            autopct="%1.1f%%", colors=["#00B894","#E17055"],
            startangle=140, explode=[0, 0.1])
axes[0].set_title("Transaction Distribution")

# Amount distribution
for cls, lbl, col in [(0,"Legitimate","#00B894"),(1,"Fraudulent","#E17055")]:
    axes[1].hist(df[df["Class"]==cls]["Amount"].clip(0,500),
                 bins=40, alpha=0.6, label=lbl, color=col)
axes[1].set_xlabel("Transaction Amount ($)")
axes[1].set_title("Amount Distribution by Class")
axes[1].legend()

# Feature correlation with Class (top 10)
corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)[:10]
corr.plot(kind="barh", ax=axes[2], color="#6C5CE7")
axes[2].set_title("Top 10 Features Correlated with Fraud")

plt.tight_layout()
plt.savefig("fraud_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA plot saved → fraud_eda.png")

# ── 3. Preprocessing ─────────────────────────────────────────
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Handle Class Imbalance ─────────────────────────────────
print("\n⚖️  Applying SMOTE to training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE — 0:{(y_resampled==0).sum()} | 1:{(y_resampled==1).sum()}")

# ── 5. Train Models ───────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42, C=0.01),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42,
                                                   class_weight="balanced"),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100,
                                                        learning_rate=0.1,
                                                        random_state=42),
}

results = {}
print("\n🔍 Model Comparison (evaluated on imbalanced test set):")
print("-" * 70)
for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc     = roc_auc_score(y_test, y_prob)
    results[name] = {"precision": prec, "recall": rec,
                     "f1": f1, "roc_auc": roc,
                     "model": model, "pred": y_pred, "prob": y_prob}
    print(f"  {name:<23} Prec={prec:.4f} | Rec={rec:.4f} | "
          f"F1={f1:.4f} | AUC={roc:.4f}")

best_name = max(results, key=lambda k: results[k]["f1"])
best      = results[best_name]
y_pred    = best["pred"]
y_prob    = best["prob"]

print(f"\n🏆 Best Model (by F1): {best_name}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Legitimate","Fraudulent"]))

# ── 6. Evaluation Plots ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Legit","Fraud"]).plot(
    ax=axes[0], cmap="Blues")
axes[0].set_title(f"Confusion Matrix — {best_name}")

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["prob"])
    axes[1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
axes[1].plot([0,1],[0,1],"k--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves")
axes[1].legend(fontsize=8)

# Precision-Recall curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec_vals, prec_vals, color="#E17055", lw=2)
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title(f"Precision-Recall Curve — {best_name}")
axes[2].fill_between(rec_vals, prec_vals, alpha=0.1, color="#E17055")

plt.tight_layout()
plt.savefig("fraud_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Metrics comparison bar chart
metrics = ["precision", "recall", "f1", "roc_auc"]
x       = np.arange(len(metrics))
width   = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
colors  = ["#6C5CE7", "#00B894", "#E17055"]
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(["Precision","Recall","F1-Score","ROC-AUC"])
ax.set_ylim(0, 1.15)
ax.set_title("Model Metrics Comparison — Fraud Detection")
ax.legend()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.01,
            f"{p.get_height():.2f}", ha="center", fontsize=7.5)
plt.tight_layout()
plt.savefig("fraud_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ All plots saved")

# ── 7. Key Insights ───────────────────────────────────────────
tp = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]; tn = cm[0,0]
print("\n💼 Business Impact Summary:")
print(f"   True Positives  (Fraud Caught)   : {tp}")
print(f"   False Negatives (Fraud Missed)   : {fn}")
print(f"   False Positives (Legit Blocked)  : {fp}")
print(f"   True Negatives  (Legit Approved) : {tn}")
print(f"   Fraud Detection Rate             : {tp/(tp+fn)*100:.1f}%")

print("\n" + "=" * 60)
print(f"  ✅ TASK 5 COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  F1-Score   : {best['f1']:.4f}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print("=" * 60)
