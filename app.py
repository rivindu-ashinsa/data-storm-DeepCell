# ----------------------- 1. Imports -----------------------
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# optional: pip install category_encoders
from category_encoders.target_encoder import TargetEncoder

# ----------------------- 2. Load --------------------------
df = pd.read_csv("./data/train_storming_round.csv")

# ----------------------- 3. Dates -------------------------
date_cols = ["agent_join_month", "first_policy_sold_month", "year_month"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], format="%m/%d/%Y")

# engineered durations
df["agent_tenure_months"] = (df["year_month"] - df["agent_join_month"]).dt.days // 30
df["months_since_first_sale"] = (
    df["year_month"] - df["first_policy_sold_month"]
).dt.days // 30
df["calendar_month"] = df["year_month"].dt.month
df["is_first_month"] = (df["agent_tenure_months"] == 0).astype(int)

# Drop raw index & raw date fields if you want
df = df.drop(columns=["row_id"])

# ----------------------- 4. Target ------------------------
TARGET = "one_month_nill"  # <-- CHANGE if your label column is named differently
y = df[TARGET]
X = df.drop(columns=[TARGET])

# ----------------------- 5. Column buckets ---------------
numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_feats = ["agent_code"]  # the only discreet string after drops

# ----------------------- 6. Transformers -----------------
numeric_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ]
)

# choose ONE of the following encoders
categorical_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", TargetEncoder()),  # or OrdinalEncoder(...)
    ]
)

preprocess = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_feats),
        ("cat", categorical_pipe, categorical_feats),
    ],
    remainder="drop",
)

# ----------------------- 7. Model ------------------------
model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,  # let it choose
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
)

clf = Pipeline([("prep", preprocess), ("lgbm", model)])

# ----------------------- 8. Train / test -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

# ----------------------- 9. Eval -------------------------
pred_proba = clf.predict_proba(X_test)[:, 1]
print("ROC‑AUC:", roc_auc_score(y_test, pred_proba).round(4))
print(classification_report(y_test, (pred_proba > 0.5).astype(int)))
