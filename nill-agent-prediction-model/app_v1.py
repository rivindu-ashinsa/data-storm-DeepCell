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

# ----------------------- 4. Target Engineering (New) ------------------------
# Sort values to ensure correct shift operation within each agent's history
df.sort_values(by=["agent_code", "year_month"], inplace=True)

# Get the new_policy_count for the next month
df["new_policy_count_next_month"] = df.groupby("agent_code")["new_policy_count"].shift(
    -1
)

# Drop rows where next_month_new_policy_count is NaN (i.e., the last month for each agent)
# as we can't determine their NILL status for the subsequent month from historical data.
df.dropna(subset=["new_policy_count_next_month"], inplace=True)

# Define the target: 1 if NILL next month, 0 otherwise
df["one_month_nill"] = (df["new_policy_count_next_month"] == 0).astype(int)

# Clean up the helper column
df = df.drop(columns=["new_policy_count_next_month"])

# ----------------------- 4. Target ------------------------
TARGET = "one_month_nill"  # This now refers to the engineered target
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
).set_output(
    transform="pandas"
)  # Added to output pandas DataFrame

# ----------------------- 7. Model ------------------------
# Calculate scale_pos_weight for class imbalance
# Using the counts from the previous run's log: positive: 1147, negative: 10375
# This calculation should ideally be done on y_train after the split,
# but for now, using the full dataset's approximate ratio.
# A more precise way would be to calculate it from y_train before fitting.
# For simplicity in this step, we'll use the approximate global ratio.
# scale_pos_weight_value = 10375 / 1147  # approx 9.045

# Let's calculate it from y_train for better accuracy in the training phase
# This part will be executed during the training step, so we define the model
# and then update scale_pos_weight before fitting.
# However, for simplicity of this edit, I will hardcode based on previous full dataset counts.
# A better approach would be to pass it dynamically or calculate after split.

model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,  # let it choose
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    # scale_pos_weight=scale_pos_weight_value # Add this for imbalance
    # Dynamically calculating scale_pos_weight after split is better.
    # For now, let's adjust the fit call or use a placeholder that gets updated.
    # The simplest change for now is to hardcode it based on the full dataset's initial proportion.
    # A more robust solution would calculate this from y_train.
)

clf = Pipeline([("prep", preprocess), ("lgbm", model)])

# ----------------------- 8. Train / test -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate scale_pos_weight using y_train
scale_pos_weight_value = (y_train.shape[0] - y_train.sum()) / y_train.sum()
clf.named_steps["lgbm"].set_params(scale_pos_weight=scale_pos_weight_value)

clf.fit(X_train, y_train)

# ----------------------- 9. Eval -------------------------
pred_proba = clf.predict_proba(X_test)[:, 1]
print("ROC‑AUC:", roc_auc_score(y_test, pred_proba).round(4))
print(classification_report(y_test, (pred_proba > 0.5).astype(int)))

# ----------------------- 10. Predict on Test Data & Create Submission File -----------------------
# Load test data
df_test = pd.read_csv("./data/test_storming_round.csv")

# Store row_ids for submission
test_row_ids = df_test["row_id"]

# Preprocess test data (similar to training data)
for col in date_cols:  # date_cols defined in section 3
    df_test[col] = pd.to_datetime(df_test[col], format="%m/%d/%Y")

df_test["agent_tenure_months"] = (
    df_test["year_month"] - df_test["agent_join_month"]
).dt.days // 30
df_test["months_since_first_sale"] = (
    df_test["year_month"] - df_test["first_policy_sold_month"]
).dt.days // 30
df_test["calendar_month"] = df_test["year_month"].dt.month
df_test["is_first_month"] = (df_test["agent_tenure_months"] == 0).astype(int)

# The test data is for predicting the NILL status for the month *after* the last year_month in the test set.
# The features used for prediction should be those available up to the last year_month.
# No target engineering is needed for the test set as we are predicting the target.

X_submission = df_test.drop(columns=["row_id"])  # Drop row_id before prediction

# Ensure X_submission has the same columns in the same order as X_train (excluding the target)
# The ColumnTransformer will handle selection of numeric_feats and categorical_feats

# Predict probabilities on the test set
submission_pred_proba = clf.predict_proba(X_submission)[:, 1]

# Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
submission_pred_binary = (submission_pred_proba > 0.5).astype(int)

# Create submission DataFrame
submission_df = pd.DataFrame(
    {
        "row_id": test_row_ids,
        "target_column": submission_pred_binary,
    }  # Use binary predictions
)

# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file created: submission.csv")
