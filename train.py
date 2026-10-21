"""
Titanic survival prediction: end-to-end ML pipeline.

Steps: load -> clean -> EDA -> feature engineering -> feature selection ->
model comparison (cross-validation) -> hyperparameter tuning -> evaluation ->
save model + figures + metrics.

Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "titanic.csv"
FIG = ROOT / "figures"
MODEL_DIR = ROOT / "model"
FIG.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42
TARGET = "Survived"


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
raw = pd.read_csv(DATA)
print(f"Raw shape: {raw.shape}")
print("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0], "\n")

df = raw.copy()
df = df.drop_duplicates()
# Cabin is ~77% missing: keep only whether a cabin was recorded (engineered below)
df["HasCabin"] = df["Cabin"].notna().astype(int)
# Embarked: 2 missing -> mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Clip extreme fares (heavy right tail) at the 99th percentile
df["Fare"] = df["Fare"].clip(upper=df["Fare"].quantile(0.99))

# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir",
    "Jonkheer", "Dona", "the Countess",
}
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Title"] = df["Title"].where(~df["Title"].isin(RARE_TITLES), "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
# Age is imputed inside the pipeline (median by fold) - keep NaN here to avoid leakage

# --------------------------------------------------------------------------
# 3. EDA
# --------------------------------------------------------------------------
print("Survival rate:", round(df[TARGET].mean(), 3))
print("\nSurvival rate by Sex:\n", df.groupby("Sex")[TARGET].mean().round(3))
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")[TARGET].mean().round(3))
print("\nSurvival rate by Title:\n", df.groupby("Title")[TARGET].agg(["mean", "count"]).round(3), "\n")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
sns.countplot(data=df, x=TARGET, ax=ax[0, 0])
ax[0, 0].set_title("Target distribution (0 = died, 1 = survived)")
sns.barplot(data=df, x="Sex", y=TARGET, ax=ax[0, 1], errorbar=None)
ax[0, 1].set_title("Survival rate by sex")
sns.barplot(data=df, x="Pclass", y=TARGET, hue="Sex", ax=ax[0, 2], errorbar=None)
ax[0, 2].set_title("Survival rate by class and sex")
sns.histplot(data=df, x="Age", hue=TARGET, bins=30, kde=True, ax=ax[1, 0])
ax[1, 0].set_title("Age distribution by outcome")
sns.barplot(data=df, x="FamilySize", y=TARGET, ax=ax[1, 1], errorbar=None)
ax[1, 1].set_title("Survival rate by family size")
sns.barplot(data=df, x="Title", y=TARGET, ax=ax[1, 2], errorbar=None)
ax[1, 2].set_title("Survival rate by title")
ax[1, 2].tick_params(axis="x", rotation=30)
save_fig("eda_overview.png")

plt.figure(figsize=(8, 6))
num_cols_eda = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]
sns.heatmap(df[num_cols_eda].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
save_fig("correlation_heatmap.png")

plt.figure(figsize=(7, 4))
missing = raw.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0] * 100
sns.barplot(x=missing.values, y=missing.index, color="#c44e52")
plt.xlabel("% missing (raw data)")
plt.title("Missing values before cleaning")
save_fig("missing_values.png")

# --------------------------------------------------------------------------
# 4. Feature selection
# --------------------------------------------------------------------------
# Dropped: PassengerId (row id), Name (replaced by Title), Ticket (high-cardinality
# identifier), Cabin (replaced by HasCabin), SibSp/Parch (captured by FamilySize).
NUM_FEATURES = ["Age", "Fare", "FamilySize", "FarePerPerson"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin"]
FEATURES = NUM_FEATURES + CAT_FEATURES

X = df[FEATURES].copy()
X["Pclass"] = X["Pclass"].astype(str)
X["IsAlone"] = X["IsAlone"].astype(str)
X["HasCabin"] = X["HasCabin"].astype(str)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])


# --------------------------------------------------------------------------
# 5. Model comparison (5-fold stratified CV on the training set)
# --------------------------------------------------------------------------
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, clf in candidates.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    rows.append({"Model": name, "CV Accuracy": acc.mean(), "CV Acc Std": acc.std(), "CV ROC-AUC": auc.mean()})
cmp_df = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
print("Model comparison (5-fold CV, training set):\n", cmp_df.round(4).to_string(index=False), "\n")

plt.figure(figsize=(8, 4.5))
order = cmp_df.sort_values("CV Accuracy")
plt.barh(order["Model"], order["CV Accuracy"], xerr=order["CV Acc Std"], color="#4c72b0")
plt.xlim(0.7, 0.9)
plt.xlabel("Cross-validated accuracy")
plt.title("Model comparison")
save_fig("model_comparison.png")

# --------------------------------------------------------------------------
# 6. Hyperparameter tuning (top models)
# --------------------------------------------------------------------------
grids = {
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8, None],
         "clf__min_samples_leaf": [1, 2, 4]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
         "clf__max_depth": [2, 3, 4]},
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.1, 0.3, 1, 3, 10]},
    ),
}
tuned = {}
tune_rows = []
for name, (clf, grid) in grids.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    tune_rows.append({"Model": name, "Best CV ROC-AUC": gs.best_score_, "Best params": gs.best_params_})
    print(f"Tuned {name}: CV AUC={gs.best_score_:.4f}  {gs.best_params_}")
tune_df = pd.DataFrame(tune_rows)

# --------------------------------------------------------------------------
# 7. Final evaluation on held-out test set
# --------------------------------------------------------------------------
results = []
for name, model in tuned.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
res_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("\nHeld-out test results (tuned models):\n", res_df.round(4).to_string(index=False))

best_name = tune_df.sort_values("Best CV ROC-AUC", ascending=False).iloc[0]["Model"]  # chosen by CV, not test
best = tuned[best_name]
print(f"\nSelected model (best CV ROC-AUC): {best_name}")
pred = best.predict(X_test)
print(classification_report(y_test, pred, target_names=["Died", "Survived"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"], ax=ax[0], cmap="Blues")
ax[0].set_title(f"Confusion matrix - {best_name}")
for name, model in tuned.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], name=name)
ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
ax[1].set_title("ROC curves (test set)")
save_fig("evaluation.png")

# Feature importance (permutation, model-agnostic, on test set, raw features)
perm = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc")
imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
plt.figure(figsize=(7, 5))
plt.barh(imp.index, imp.values, xerr=perm.importances_std[[FEATURES.index(i) for i in imp.index]], color="#55a868")
plt.xlabel("Drop in ROC-AUC when shuffled")
plt.title(f"Permutation feature importance - {best_name}")
save_fig("feature_importance.png")
print("\nFeature importance:\n", imp.sort_values(ascending=False).round(4).to_string())

# --------------------------------------------------------------------------
# 8. Save artefacts
# --------------------------------------------------------------------------
joblib.dump(best, MODEL_DIR / "titanic_model.joblib")
metrics = {
    "selected_model": best_name,
    "test_metrics": res_df[res_df.Model == best_name].iloc[0].drop("Model").to_dict(),
    "cv_comparison": cmp_df.to_dict(orient="records"),
    "tuned": [{"Model": r["Model"], "cv_auc": r["Best CV ROC-AUC"], "params": r["Best params"]} for r in tune_rows],
    "test_all_models": res_df.to_dict(orient="records"),
    "features": FEATURES,
    "defaults": {
        "Age": float(X_train["Age"].median()),
        "Fare": float(X_train["Fare"].median()),
    },
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
cmp_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
print("\nSaved model, metrics and figures.")

# --------------------------------------------------------------------------
# 9. Display prediction results on sample test passengers
# --------------------------------------------------------------------------
sample = X_test.head(10).copy()
sample["Actual"] = y_test.head(10).map({0: "Died", 1: "Survived"})
sample["Predicted"] = pd.Series(best.predict(sample[FEATURES]), index=sample.index).map({0: "Died", 1: "Survived"})
sample["P(survive)"] = best.predict_proba(sample[FEATURES])[:, 1].round(3)
print("\nSample predictions:\n", sample[["Sex", "Pclass", "Age", "Title", "Actual", "Predicted", "P(survive)"]].to_string())
sample.to_csv(MODEL_DIR / "sample_predictions.csv")
