# ============================================================
#  CODSOFT DATA SCIENCE INTERNSHIP — TASK 3
#  Iris Flower Classification
#  Author : SHREEKANTH A GUTTEDAR
#  Dataset: sklearn built-in / UCI Machine Learning Repository
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  IRIS FLOWER CLASSIFICATION")
print("=" * 60)

# ── 1. Load Dataset ───────────────────────────────────────────
iris = load_iris()
df   = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\n✅ Dataset loaded: {df.shape[0]} samples × {df.shape[1]} features")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["species"].value_counts())
print("\nStatistical Summary:")
print(df.describe().round(2))

# ── 2. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Flower EDA", fontsize=16, fontweight="bold")

features = iris.feature_names
colors   = {"setosa": "#E74C3C", "versicolor": "#2ECC71", "virginica": "#3498DB"}

for ax, feat in zip(axes.flatten()[:4], features):
    for species, grp in df.groupby("species"):
        ax.hist(grp[feat], alpha=0.6, label=species,
                color=colors[species], bins=15, edgecolor="white")
    ax.set_title(feat)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("iris_eda.png", dpi=150, bbox_inches="tight")
plt.show()

# Pairplot
pair_fig = sns.pairplot(df, hue="species",
                        palette={"setosa":"#E74C3C",
                                 "versicolor":"#2ECC71",
                                 "virginica":"#3498DB"},
                        diag_kind="kde", markers=["o","s","D"])
pair_fig.fig.suptitle("Iris Pairplot", y=1.02, fontsize=14)
pair_fig.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA plots saved")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, fmt=".2f",
            cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 3. Preprocessing ─────────────────────────────────────────
X = iris.data
y = iris.target

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Train & Compare Models ─────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", C=10, gamma=0.1, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
print("\n🔍 Model Comparison:")
print("-" * 55)
for name, model in models.items():
    cv   = cross_val_score(model, X_scaled, y, cv=10, scoring="accuracy")
    model.fit(X_train, y_train)
    acc  = accuracy_score(y_test, model.predict(X_test))
    results[name] = {"cv_mean": cv.mean(), "cv_std": cv.std(),
                     "test_acc": acc, "model": model}
    print(f"  {name:<26} CV={cv.mean():.4f}±{cv.std():.4f} | Test={acc:.4f}")

best_name  = max(results, key=lambda k: results[k]["test_acc"])
best_model = results[best_name]["model"]
y_pred     = best_model.predict(X_test)

print(f"\n🏆 Best Model: {best_name}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

# ── 5. Results Visualization ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(
    ax=axes[0], cmap="Purples")
axes[0].set_title(f"Confusion Matrix — {best_name}")

names = list(results.keys())
accs  = [results[n]["test_acc"] for n in names]
bars  = axes[1].barh(names, accs,
                     color=["#E74C3C","#2ECC71","#3498DB","#F39C12","#9B59B6"])
axes[1].set_xlim(0.8, 1.01)
axes[1].set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{acc:.4f}", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Results plot saved → iris_results.png")

# ── 6. Predict Custom Input ───────────────────────────────────
print("\n🔮 Sample Prediction:")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # classic setosa
sample_scaled = scaler.transform(sample)
pred = best_model.predict(sample_scaled)
print(f"   Input: sepal_length=5.1, sepal_width=3.5, "
      f"petal_length=1.4, petal_width=0.2")
print(f"   → Predicted Species: {iris.target_names[pred[0]].upper()}")

print("\n" + "=" * 60)
print(f"  ✅ TASK 3 COMPLETE — Best Accuracy: {max(accs):.4f}")
print("=" * 60)
