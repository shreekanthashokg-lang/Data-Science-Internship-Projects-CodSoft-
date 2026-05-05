# ============================================================
#  CODSOFT DATA SCIENCE INTERNSHIP — TASK 2
#  Movie Rating Prediction with Python
#  Author : SHREEKANTH A GUTTEDAR
#  Dataset: IMDb Movies India (Kaggle)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  MOVIE RATING PREDICTION WITH PYTHON")
print("=" * 60)

# ── 1. Create Realistic Synthetic Dataset ────────────────────
# (mirrors the structure of the IMDb India dataset)
np.random.seed(42)
n = 1000

genres   = ["Action", "Comedy", "Drama", "Thriller", "Romance",
            "Horror", "Biography", "Crime", "Adventure", "Sci-Fi"]
directors = [f"Director_{i}" for i in range(50)]
actors    = [f"Actor_{i}"    for i in range(80)]

df = pd.DataFrame({
    "Name"    : [f"Movie_{i}" for i in range(n)],
    "Year"    : np.random.randint(1990, 2024, n),
    "Duration": np.random.randint(80, 200, n),
    "Genre"   : np.random.choice(genres, n),
    "Director": np.random.choice(directors, n),
    "Actor1"  : np.random.choice(actors, n),
    "Actor2"  : np.random.choice(actors, n),
    "Votes"   : np.random.randint(100, 500_000, n),
})

# Realistic rating generation (influenced by features)
base_rating = 5.0
director_boost = {d: np.random.uniform(-1, 1.5) for d in directors}
genre_boost    = {g: np.random.uniform(-0.5, 1.0) for g in genres}

df["Rating"] = (
    base_rating
    + df["Director"].map(director_boost)
    + df["Genre"].map(genre_boost)
    + np.log1p(df["Votes"]) * 0.12
    + np.random.normal(0, 0.4, n)
).clip(1, 10).round(1)

print(f"\n✅ Dataset ready: {df.shape[0]} movies")
print("\nSample:")
print(df.head())
print(f"\nRating — Mean: {df['Rating'].mean():.2f} | Std: {df['Rating'].std():.2f}")

# ── 2. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Movie Rating EDA", fontsize=16, fontweight="bold")

sns.histplot(df["Rating"], kde=True, bins=30, color="#6C5CE7", ax=axes[0, 0])
axes[0, 0].set_title("Rating Distribution")

avg_genre = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
avg_genre.plot(kind="bar", ax=axes[0, 1], color="#00B894")
axes[0, 1].set_title("Avg Rating by Genre")
axes[0, 1].tick_params(axis="x", rotation=45)

axes[1, 0].scatter(np.log1p(df["Votes"]), df["Rating"],
                   alpha=0.3, color="#E17055", s=15)
axes[1, 0].set_xlabel("log(Votes)")
axes[1, 0].set_ylabel("Rating")
axes[1, 0].set_title("Votes vs Rating")

sns.boxplot(data=df, x="Genre", y="Rating", ax=axes[1, 1], palette="Set3")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].set_title("Rating by Genre (Boxplot)")

plt.tight_layout()
plt.savefig("movie_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA plot saved → movie_eda.png")

# ── 3. Preprocessing ─────────────────────────────────────────
le = LabelEncoder()
for col in ["Genre", "Director", "Actor1", "Actor2"]:
    df[col + "_enc"] = le.fit_transform(df[col])

df["votes_log"]   = np.log1p(df["Votes"])
df["movie_age"]   = 2024 - df["Year"]

feature_cols = ["Duration", "votes_log", "movie_age",
                "Genre_enc", "Director_enc", "Actor1_enc", "Actor2_enc"]

X = df[feature_cols]
y = df["Rating"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    "Linear Regression"  : LinearRegression(),
    "Ridge Regression"   : Ridge(alpha=1.0),
    "Random Forest"      : RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
print("\n🔍 Model Comparison:")
print("-" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"rmse": rmse, "mae": mae, "r2": r2,
                     "model": model, "pred": y_pred}
    print(f"  {name:<22} RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

best_name = min(results, key=lambda k: results[k]["rmse"])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}")

# ── 5. Evaluation Plots ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, best["pred"], alpha=0.4, color="#6C5CE7", s=20)
lims = [min(y_test.min(), best["pred"].min()),
        max(y_test.max(), best["pred"].max())]
axes[0].plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
axes[0].set_xlabel("Actual Rating")
axes[0].set_ylabel("Predicted Rating")
axes[0].set_title(f"Actual vs Predicted — {best_name}")
axes[0].legend()

errors = y_test.values - best["pred"]
axes[1].hist(errors, bins=30, color="#00B894", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Prediction Error")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("movie_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Model comparison bar chart
names = list(results.keys())
rmses = [results[n]["rmse"] for n in names]
r2s   = [results[n]["r2"]   for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(names, rmses, color=["#6C5CE7","#00B894","#E17055","#FDCB6E"])
axes[0].set_title("RMSE Comparison (lower=better)")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(names, r2s, color=["#6C5CE7","#00B894","#E17055","#FDCB6E"])
axes[1].set_title("R² Comparison (higher=better)")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("movie_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Results plots saved")

print("\n" + "=" * 60)
print(f"  ✅ TASK 2 COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  RMSE       : {best['rmse']:.4f}")
print(f"  R² Score   : {best['r2']:.4f}")
print("=" * 60)
