# pip install catboost lightgbm optuna imbalanced-learn feature-engine
import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

print("Loading training data...")
df = pd.read_csv(
    "./data/train_storming_round.csv",
    parse_dates=["agent_join_month", "first_policy_sold_month", "year_month"],
)
# ---------- feature engineering helpers ----------
print("Performing feature engineering on training data...")
df.sort_values(["agent_code", "year_month"], inplace=True)
df["agent_tenure_months"] = (df["year_month"] - df["agent_join_month"]).dt.days // 30
df["months_since_first_sale"] = (
    df["year_month"] - df["first_policy_sold_month"]
).dt.days // 30
df["calendar_month"] = df["year_month"].dt.month
df["is_first_month"] = (df["agent_tenure_months"] == 0).astype(int)

# lag + rolling (feature_engine does column-wise)
# Ensure 'new_policy_count' exists and handle potential NaNs from shifting if necessary
# For LagFeatures, fillna is often handled by the transformer or needs to be done post-transformation
lagger = LagFeatures(
    variables=["new_policy_count"], periods=[1], missing_values="ignore"
)
roller = WindowFeatures(
    variables=["new_policy_count"],
    window=[3, 6],
    functions=["mean", "median"],
    missing_values="ignore",
)

df = lagger.fit_transform(df)
# For WindowFeatures, feature_engine typically handles NaNs created by rolling windows by not producing a value
# or by filling with a value like 0 or mean, depending on its internal logic or if specified.
# We might need to impute NaNs if they persist and the model can't handle them.
df = roller.fit_transform(
    df
)  # Removed df["year_month"] as it's not a standard arg for transform here

# Impute NaNs created by LagFeatures and WindowFeatures
# Common strategy: fill with 0 (e.g. no sales in prior period) or median/mean of the column
lag_col_name = "new_policy_count_lag_1"
if lag_col_name in df.columns:
    df[lag_col_name] = df[lag_col_name].fillna(0)

for window_func in ["mean", "median"]:
    for window_size in [3, 6]:
        col_name = f"new_policy_count_window_{window_size}_{window_func}"
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(0)

# ---------- target ----------
print("Engineering target variable...")
df["one_month_nill"] = (
    df.groupby("agent_code")["new_policy_count"].shift(-1) == 0
).astype(int)
df.dropna(
    subset=["one_month_nill"], inplace=True
)  # Critical: remove rows where target can't be known

y = df["one_month_nill"]
X = df.drop(
    columns=[
        "one_month_nill",
        "row_id",
        "agent_join_month",
        "first_policy_sold_month",
        "year_month",
    ]
)
# Dropped date columns as their info is captured in engineered features or not directly usable by CatBoost unless formatted.

cat_cols = ["agent_code", "calendar_month"]
# Ensure all columns in X are either in cat_cols or treated as num_cols
num_cols = [col for col in X.columns if col not in cat_cols]

# CatBoost handles categorical features natively, so passthrough is fine.
# However, ensure all columns are either explicitly categorical or numerical.
# For simplicity, we'll pass all columns and let CatBoost infer types or use cat_features parameter.

print("Starting Optuna hyperparameter optimization...")


def objective(trial):
    params = {
        "iterations": 2000,  # Optuna might run faster with fewer iterations for trials
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2", 1e-2, 10.0, log=True),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "class_weights": {0: 1, 1: trial.suggest_float("pos_w", 5, 15)},
    }
    cv = GroupKFold(n_splits=4)  # n_splits added
    aucs = []
    # Create a temporary dataframe for splitting that includes the group key
    X_for_split = X.copy()
    X_for_split["agent_code_group"] = df.loc[
        X.index, "agent_code"
    ]  # Align agent_code for splitting

    for train_idx, val_idx in cv.split(
        X_for_split, y, groups=X_for_split["agent_code_group"]
    ):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE only to training data
        # Ensure cat_cols are correctly identified for SMOTE if it needs specific handling for them
        # SMOTE typically works on numerical data. If cat_cols are not encoded, SMOTE might fail or behave unexpectedly.
        # For CatBoost, we pass string categorical features directly. SMOTE might need them to be numerically encoded first.
        # Let's assume for now num_cols are the ones to be oversampled, or cat_cols are pre-encoded if SMote is used.
        # Given CatBoost's native handling, it might be better to use CatBoost's internal class weighting or 'auto_class_weights'
        # For now, proceeding with SMOTE as in the original snippet.

        # Convert categorical columns to string for SMOTE to handle them as discrete (if not already)
        # This is a common point of failure if SMOTE receives mixed types or non-numeric data without proper preprocessing.
        # However, CatBoost Pool expects categorical features to be passed as is (string or int indices).
        # Let's prepare data for SMOTE (numerical) and then reconstruct for CatBoost Pool.

        X_tr_num = X_tr[num_cols].fillna(0)  # Fill NaNs for SMOTE
        sampler = BorderlineSMOTE(k_neighbors=3, random_state=42)
        X_tr_s_num, y_tr_s = sampler.fit_resample(X_tr_num, y_tr)

        # Reconstruct X_tr_s with categorical features from original X_tr, aligned with y_tr_s
        # This is tricky. SMOTE changes the number of samples. We need to carefully align.
        # A simpler approach for CatBoost might be to rely on its class_weights or scale_pos_weight.
        # Given the complexity, let's simplify and use CatBoost's class_weights, removing SMOTE for now for stability.
        # If SMOTE is essential, careful handling of categorical features is needed.

        # Simplified: No SMOTE, rely on CatBoost's class_weights
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols)
        pool_val = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(**params)
        model.fit(pool_tr, eval_set=pool_val, use_best_model=True)
        aucs.append(roc_auc_score(y_val, model.predict_proba(pool_val)[:, 1]))
    return np.mean(aucs)


study = optuna.create_study(direction="maximize")
# Reduced n_trials for faster execution in this environment. Increase for better results.
study.optimize(
    objective, n_trials=5, show_progress_bar=True
)  # Reduced trials to 5 for speed
best_params_from_optuna = study.best_params

print("Best parameters from Optuna:", best_params_from_optuna)

# ---------- final CatBoost model training ----------
print("Training final CatBoost model with best parameters...")
final_model_params = {
    "iterations": 4000,  # As per original snippet, but can be adjusted
    "learning_rate": best_params_from_optuna.get("lr", 0.05),  # Default if not in study
    "depth": best_params_from_optuna.get("depth", 6),
    "l2_leaf_reg": best_params_from_optuna.get("l2", 3.0),
    "loss_function": "Logloss",
    "eval_metric": "AUC",  # Or 'Logloss' if preferred for final training metric
    "random_seed": 42,
    "verbose": 100,  # Print progress every 100 iterations
    "early_stopping_rounds": 200,  # Longer patience for final model
    "class_weights": {
        0: 1,
        1: best_params_from_optuna.get("pos_w", 10),
    },  # Use optimized weight
}

# For the final model, train on the full dataset X, y
# SMOTE is not used here as per simplification in objective function. Relying on class_weights.
final_pool = Pool(X, y, cat_features=cat_cols)
best_model = CatBoostClassifier(**final_model_params)

# Fit with an eval set if you want to use early stopping effectively on the full data.
# This requires splitting X, y again or using a portion as eval.
# For simplicity, fitting on full data without eval_set for early stopping (might train for all iterations).
# Or, we can use a portion of X,y as eval set for early stopping.
from sklearn.model_selection import train_test_split

X_train_final, X_eval_final, y_train_final, y_eval_final = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
final_train_pool = Pool(X_train_final, y_train_final, cat_features=cat_cols)
final_eval_pool = Pool(X_eval_final, y_eval_final, cat_features=cat_cols)

best_model.fit(final_train_pool, eval_set=final_eval_pool, use_best_model=True)

print("CV AUC from Optuna study:", study.best_value)

# ---------- Prediction on Test Data -----------
print("Loading test data...")
df_test = pd.read_csv(
    "./data/test_storming_round.csv",
    parse_dates=["agent_join_month", "first_policy_sold_month", "year_month"],
)
test_row_ids = df_test["row_id"]

print("Performing feature engineering on test data...")
df_test.sort_values(["agent_code", "year_month"], inplace=True)
df_test["agent_tenure_months"] = (
    df_test["year_month"] - df_test["agent_join_month"]
).dt.days // 30
df_test["months_since_first_sale"] = (
    df_test["year_month"] - df_test["first_policy_sold_month"]
).dt.days // 30
df_test["calendar_month"] = df_test["year_month"].dt.month
df_test["is_first_month"] = (df_test["agent_tenure_months"] == 0).astype(int)

df_test = lagger.transform(df_test)  # Use transform, not fit_transform
df_test = roller.transform(df_test)  # Use transform, not fit_transform

# Impute NaNs for test set, similar to training set
if lag_col_name in df_test.columns:
    df_test[lag_col_name] = df_test[lag_col_name].fillna(0)
for window_func in ["mean", "median"]:
    for window_size in [3, 6]:
        col_name = f"new_policy_count_window_{window_size}_{window_func}"
        if col_name in df_test.columns:
            df_test[col_name] = df_test[col_name].fillna(0)

X_test_submission = df_test.drop(
    columns=["row_id", "agent_join_month", "first_policy_sold_month", "year_month"]
)

# Ensure columns are in the same order as training data X
X_test_submission = X_test_submission[X.columns]

print("Making predictions on test data...")
preds_cat_test = best_model.predict_proba(X_test_submission)[:, 1]

# Convert probabilities to binary 0/1 for submission
submission_preds_binary = (preds_cat_test > 0.5).astype(int)

submission_df = pd.DataFrame(
    {"row_id": test_row_ids, "target_column": submission_preds_binary}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

# The ensembling part is commented out as best_lgbm and X_test_lgb are not defined here.
# preds_cat = best_model.predict_proba(X_test_cat)[:,1] # X_test_cat needs to be defined (e.g. from a train/test split of original df)
# preds_lgb = best_lgbm.predict_proba(X_test_lgb)[:,1]
# final_pred = 0.6*preds_cat + 0.4*preds_lgb

print("Script finished.")
