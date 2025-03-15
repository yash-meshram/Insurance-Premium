import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
import warnings

warnings.filterwarnings("ignore")


def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading data...")
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Display basic information about the dataset
print("\nSample of training data:")
print(train_data.head())

# Check for missing values
print("\nMissing values in training data:")
missing_values = train_data.isnull().sum()
print(missing_values[missing_values > 0])

# IMPORTANT: Use log transform on target variable
print("Applying log transform to target variable")
y_train_raw = train_data["Premium Amount"].copy()
train_data["Premium Amount"] = np.log1p(train_data["Premium Amount"])

# Convert Policy Start Date to datetime and extract features
train_data["Policy Start Date"] = pd.to_datetime(
    train_data["Policy Start Date"], errors="coerce"
)
test_data["Policy Start Date"] = pd.to_datetime(
    test_data["Policy Start Date"], errors="coerce"
)

# Enhanced datetime features
for df in [train_data, test_data]:
    df["Policy_Start_Year"] = df["Policy Start Date"].dt.year
    df["Policy_Start_Month"] = df["Policy Start Date"].dt.month
    df["Policy_Start_Day"] = df["Policy Start Date"].dt.day
    df["Policy_Start_Weekday"] = df["Policy Start Date"].dt.weekday
    df["Policy_Start_Quarter"] = df["Policy Start Date"].dt.quarter
    df["Policy_Start_DayOfYear"] = df["Policy Start Date"].dt.dayofyear
    df["Policy_Start_IsWeekend"] = df["Policy_Start_Weekday"].isin([5, 6]).astype(int)
    df["Policy_Start_IsMonthEnd"] = df["Policy Start Date"].dt.is_month_end.astype(int)
    df["Policy_Start_IsMonthStart"] = df["Policy Start Date"].dt.is_month_start.astype(
        int
    )

# Drop the original datetime column
train_data = train_data.drop("Policy Start Date", axis=1)
test_data = test_data.drop("Policy Start Date", axis=1)

# Advanced Feature Engineering
for df in [train_data, test_data]:
    # Age-related features
    df["Age_Squared"] = df["Age"] ** 2
    df["Age_Cubed"] = df["Age"] ** 3
    df["Log_Age"] = np.log1p(df["Age"])
    df["Age_Bins"] = pd.qcut(df["Age"], q=5, labels=False, duplicates="drop")

    # Income-related features
    df["Log_Income"] = np.log1p(df["Annual Income"])
    df["Income_Squared"] = df["Annual Income"] ** 2
    df["Income_Bins"] = pd.qcut(
        df["Annual Income"], q=5, labels=False, duplicates="drop"
    )

    # Health-related features
    df["Health_Squared"] = df["Health Score"] ** 2
    df["Health_Cubed"] = df["Health Score"] ** 3
    df["Log_Health"] = np.log1p(df["Health Score"])
    df["Health_Bins"] = pd.qcut(
        df["Health Score"], q=5, labels=False, duplicates="drop"
    )

    # Credit-related features
    df["Log_Credit"] = np.log1p(df["Credit Score"])
    df["Credit_Squared"] = df["Credit Score"] ** 2
    df["Credit_Bins"] = pd.qcut(
        df["Credit Score"], q=5, labels=False, duplicates="drop"
    )

    # Duration-related features
    df["Log_Duration"] = np.log1p(df["Insurance Duration"])
    df["Duration_Squared"] = df["Insurance Duration"] ** 2
    df["Duration_Bins"] = pd.qcut(
        df["Insurance Duration"], q=5, labels=False, duplicates="drop"
    )

    # Claim-related features
    df["Log_Claims"] = np.log1p(df["Previous Claims"])
    df["Claims_Squared"] = df["Previous Claims"] ** 2
    df["Claims_Bins"] = pd.qcut(
        df["Previous Claims"], q=5, labels=False, duplicates="drop"
    )

    # Vehicle-related features
    df["Log_Vehicle_Age"] = np.log1p(df["Vehicle Age"])
    df["Vehicle_Age_Squared"] = df["Vehicle Age"] ** 2
    df["Vehicle_Age_Bins"] = pd.qcut(
        df["Vehicle Age"], q=5, labels=False, duplicates="drop"
    )

    # Dependent-related features
    df["Log_Dependents"] = np.log1p(df["Number of Dependents"])
    df["Dependents_Squared"] = df["Number of Dependents"] ** 2
    df["Dependents_Bins"] = pd.qcut(
        df["Number of Dependents"], q=5, labels=False, duplicates="drop"
    )

    # Advanced interaction features
    df["Age_Income"] = df["Age"] * df["Annual Income"]
    df["Health_Credit"] = df["Health Score"] * df["Credit Score"]
    df["Age_Health"] = df["Age"] * df["Health Score"]
    df["Income_Dependents"] = df["Annual Income"] / (df["Number of Dependents"] + 1)
    df["Age_Claims"] = df["Age"] * df["Previous Claims"]
    df["Duration_Claims"] = df["Insurance Duration"] * df["Previous Claims"]
    df["Health_Income"] = df["Health Score"] * df["Annual Income"]
    df["Credit_Duration"] = df["Credit Score"] * df["Insurance Duration"]
    df["Age_Duration"] = df["Age"] * df["Insurance Duration"]
    df["Claims_Dependents"] = df["Previous Claims"] * df["Number of Dependents"]

    # Ratio features
    df["Income_Per_Dependent"] = df["Annual Income"] / (df["Number of Dependents"] + 1)
    df["Health_Per_Age"] = df["Health Score"] / df["Age"]
    df["Credit_Per_Claim"] = df["Credit Score"] / (df["Previous Claims"] + 1)
    df["Duration_Per_Age"] = df["Insurance Duration"] / df["Age"]

# Separate features and target
X_train = train_data.drop(["Premium Amount", "id"], axis=1)
y_train = train_data["Premium Amount"]  # Already log-transformed
X_test = test_data.drop(["id"], axis=1)

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns

# Handle missing values for numeric features
num_imputer = SimpleImputer(strategy="median")
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

# Handle missing values for categorical features
for col in categorical_cols:
    most_frequent = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(most_frequent)
    X_test[col] = X_test[col].fillna(most_frequent)

# Apply Yeo-Johnson power transformation to numeric columns
# This normalizes the data better than simple scaling
power_transformer = PowerTransformer(method="yeo-johnson")
for col in numeric_cols:
    if X_train[col].nunique() > 5:  # Only transform features with enough unique values
        X_train[col] = power_transformer.fit_transform(
            X_train[col].values.reshape(-1, 1)
        )
        X_test[col] = power_transformer.transform(X_test[col].values.reshape(-1, 1))

# Convert categorical to numeric using one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure X_train and X_test have the same columns
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

# Add columns in test that are not in train with zeros
missing_cols = set(X_test.columns) - set(X_train.columns)
for col in missing_cols:
    X_train[col] = 0

# Ensure column order is the same
X_test = X_test[X_train.columns]

# Split the data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define base models
print("\nTraining LightGBM models...")
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "num_iterations": 1000,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
}

# Create lightgbm datasets
train_dataset = lgb.Dataset(X_train_split, label=y_train_split)
val_dataset = lgb.Dataset(X_val, label=y_val)

# Train model with cross-validation
n_splits = 3  # Reduced number of folds for stability
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nTraining fold {fold}/{n_splits}")
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_dataset_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_dataset_fold = lgb.Dataset(X_val_fold, label=y_val_fold)

    model = lgb.train(
        lgb_params,
        train_dataset_fold,
        valid_sets=[train_dataset_fold, val_dataset_fold],
        num_boost_round=500,  # Reduced number of rounds
        callbacks=[
            lgb.early_stopping(stopping_rounds=30)
        ],  # Reduced early stopping rounds
    )

    models.append(model)

    # Evaluate on validation fold
    val_pred = model.predict(X_val_fold)
    val_pred_original = np.expm1(val_pred)
    y_val_original = np.expm1(y_val_fold)
    fold_rmsle = rmsle(y_val_original, val_pred_original)
    cv_scores.append(fold_rmsle)
    print(f"Fold {fold} RMSLE: {fold_rmsle:.4f}")

print(f"\nMean CV RMSLE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train final model on all training data
print("\nTraining final model on all data...")
train_dataset_full = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    lgb_params,
    train_dataset_full,
    num_boost_round=500,  # Removed early stopping callback for final model
)

# Make predictions on test data
print("\nMaking predictions on test data...")
test_predictions_log = final_model.predict(X_test)

# Ensemble predictions from all models
test_predictions_log_ensemble = np.zeros(len(X_test))
for model in models:
    test_predictions_log_ensemble += model.predict(X_test)
test_predictions_log_ensemble /= len(models)

# Average predictions from final model and ensemble
test_predictions_log = (test_predictions_log + test_predictions_log_ensemble) / 2

# Transform predictions back to original scale
test_predictions = np.expm1(test_predictions_log)

# Create submission file
submission = pd.DataFrame({"id": test_data["id"], "premium": test_predictions})

# Ensure predictions are positive
submission["premium"] = np.maximum(0, submission["premium"])

# Save predictions
submission.to_csv("predictions.csv", index=False)
print("\nPredictions saved to 'predictions.csv'")

# Feature importance
feature_importance = pd.DataFrame(
    {
        "Feature": final_model.feature_name(),
        "Importance": final_model.feature_importance(importance_type="gain"),
    }
)
feature_importance = feature_importance.sort_values("Importance", ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
