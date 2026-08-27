"""Simulates an incident: a poorly evaluated model gets pushed straight to
Production without going through the promotion criteria in telco_promotion.py.
Used to exercise the rollback procedure in rollback.py."""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Telco_Churn_Promotion")

df = pd.read_csv("data/Telco-Customer-Churn.csv").drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run(run_name="incident_bad_model") as run:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "dummy_most_frequent")
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id
    print(f"Trained deliberately weak model, run {run_id}, f1_score={f1:.4f}")

client = MlflowClient()
mv = client.create_model_version(
    name="Telco_Churn_Model",
    run_id=run_id,
    source=f"runs:/{run_id}/model",
)
# Simulates a bad manual promotion that skipped the automated criteria check.
client.transition_model_version_stage(
    name="Telco_Churn_Model",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Incident: version {mv.version} (f1_score={f1:.4f}) pushed to Production, bypassing promotion criteria.")
