"""
train_models.py
Run this ONCE to train models on the anemia.csv dataset and save them as .pkl files.
Usage: python train_models.py
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Load dataset ────────────────────────────────────────────────────────────
CSV_PATH = "anemia.csv"   # <-- place anemia.csv in the project root

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found. Place anemia.csv in the project root folder."
    )

df = pd.read_csv(CSV_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── Features & target ───────────────────────────────────────────────────────
FEATURES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
X = df[FEATURES].values
y = df["Result"].values

SEED = 15
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# ── Scale features ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ───────────────────────────────────────────────────────────────────
models = {
    "svm": SVC(probability=True, kernel="rbf", C=1.0, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
os.makedirs("model", exist_ok=True)

print("\n" + "="*60)
for name, clf in models.items():
    # SVM needs scaled data; tree models can use raw but scaled is fine
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=cv, scoring="accuracy")
    print(f"\n{name.upper()}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV  Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Anemia", "Anemia"]))
    joblib.dump(clf, f"model/{name}.pkl")
    print(f"  Saved → model/{name}.pkl")

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")
print("\nScaler saved → model/scaler.pkl")
print("="*60)
print("\n✅ Training complete! Run: python app.py")
