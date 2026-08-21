import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import time
import numpy as np
import tempfile
import joblib
from sklearn.metrics import f1_score, recall_score
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------- Load dataset (cached locally after first download) ----------------
local_path = "data/Telco-Customer-Churn.csv"
if not os.path.exists(local_path):
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.to_csv(local_path, index=False)
else:
    df = pd.read_csv(local_path)

# Drop customer ID
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric (handling spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Target: Churn -> 1 if Yes, 0 if No
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Protected groups for fairness evaluation (gender)
groups_train = X_train['gender'].map({'Female': 0, 'Male': 1})
groups_test = X_test['gender'].map({'Female': 0, 'Male': 1})

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Churn rate (train):", y_train.mean())
print("Churn rate (test):", y_test.mean())

# ---------------- MLflow setup ----------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Telco_Churn_Promotion")
client = MlflowClient()

promotion_criteria = {
    "f1_score_min": 0.55,
    "latency_max_ms": 10,
    "model_size_max_mb": 100,
    "fairness_max_disparity": 0.10,
}

print("MLflow tracking URI set. Promotion criteria:", promotion_criteria)

# ---------------- Evaluation helper ----------------
def evaluate_model(model, X_test, y_test, groups_test, criteria):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    start = time.time()
    for _ in range(100):
        model.predict(X_test[:1])
    latency_ms = (time.time() - start) * 1000 / 100

    recall_group0 = recall_score(y_test[groups_test == 0], y_pred[groups_test == 0])
    recall_group1 = recall_score(y_test[groups_test == 1], y_pred[groups_test == 1])
    fairness_disparity = abs(recall_group0 - recall_group1)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        model_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        os.unlink(tmp.name)

    return {
        "f1_score": f1,
        "latency_ms": latency_ms,
        "model_size_mb": model_size_mb,
        "fairness_disparity": fairness_disparity,
    }

print("evaluate_model function defined.")

PIP_REQS = ["-r requirements.txt"]

# ---------------- Experiment 1: Logistic Regression ----------------
with mlflow.start_run(run_name="logistic_regression") as run:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])

    mlflow.sklearn.log_model(
    pipeline,
    "model",
    pip_requirements=PIP_REQS,
    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
)
    run_id_lr = run.info.run_id
    print("Logistic Regression:", metrics)

# ---------------- Experiment 2: Random Forest ----------------
with mlflow.start_run(run_name="random_forest") as run:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])

    mlflow.sklearn.log_model(pipeline, "model", pip_requirements=PIP_REQS)
    run_id_rf = run.info.run_id
    print("Random Forest:", metrics)

# ---------------- Experiment 3: XGBoost ----------------
with mlflow.start_run(run_name="xgboost") as run:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)

    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])

    # Log the full pipeline (not just the classifier) so the saved model
    mlflow.sklearn.log_model(pipeline, "model", pip_requirements=PIP_REQS, skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"])
    run_id_xgb = run.info.run_id
    print("XGBoost:", metrics)

# ---------------- Evaluate against promotion criteria ----------------
experiment = mlflow.get_experiment_by_name("Telco_Churn_Promotion")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print(runs_df[["run_id", "tags.mlflow.runName", "metrics.f1_score", "metrics.latency_ms",
               "metrics.model_size_mb", "metrics.fairness_disparity"]])

best_run = runs_df.loc[runs_df["metrics.f1_score"].idxmax()]
best_run_id = best_run["run_id"]
print(f"Best run: {best_run_id} ({best_run['tags.mlflow.runName']})")

criteria_checks = {
    "f1_score": best_run["metrics.f1_score"] >= promotion_criteria["f1_score_min"],
    "latency_ms": best_run["metrics.latency_ms"] <= promotion_criteria["latency_max_ms"],
    "model_size_mb": best_run["metrics.model_size_mb"] <= promotion_criteria["model_size_max_mb"],
    "fairness_disparity": best_run["metrics.fairness_disparity"] <= promotion_criteria["fairness_max_disparity"],
}
print("Criteria checks:", criteria_checks)

if all(criteria_checks.values()):
    print("Promoting model to Staging and Production")
    try:
        client.create_registered_model("Telco_Churn_Model")
    except Exception:
        pass

    model_uri = f"runs:/{best_run_id}/model"
    mv = client.create_model_version(
        name="Telco_Churn_Model",
        run_id=best_run_id,
        source=model_uri
    )
    print(f"Created model version {mv.version}")

    client.transition_model_version_stage(
        name="Telco_Churn_Model",
        version=mv.version,
        stage="Staging"
    )
    client.transition_model_version_stage(
        name="Telco_Churn_Model",
        version=mv.version,
        stage="Production"
    )
    print("Model promoted to Production.")
else:
    print("Best model does not meet all promotion criteria.")