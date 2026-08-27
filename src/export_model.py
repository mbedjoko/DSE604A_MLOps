import os
import json
import sqlite3

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "Telco_Churn_Model"
STAGE = os.environ.get("MODEL_STAGE", "Production")
OUT_DIR = "models"
DB_PATH = os.environ.get("MLFLOW_DB_PATH", "mlflow.db")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", "mlartifacts")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

versions = [
    v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
    if v.current_stage == STAGE
]
if not versions:
    raise SystemExit(f"No version of '{MODEL_NAME}' currently in stage '{STAGE}'.")

mv = max(versions, key=lambda v: int(v.version))

# The registry entry points at a runs:/ URI that mlflow's newer logged-model
# storage does not resolve directly, so look up the underlying logged-model
# artifact location straight from the backend store and load it from disk.
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(
    "SELECT artifact_location FROM logged_models WHERE source_run_id = ? "
    "ORDER BY creation_timestamp_ms DESC LIMIT 1",
    (mv.run_id,),
)
row = cur.fetchone()
conn.close()
if not row:
    raise SystemExit(f"No logged model found for run {mv.run_id}")

artifact_location = row[0].replace("mlflow-artifacts:/", "")
local_model_path = os.path.join(ARTIFACT_ROOT, artifact_location)
model = mlflow.sklearn.load_model(local_model_path)

os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(model, os.path.join(OUT_DIR, "telco_churn_model.pkl"))

run = client.get_run(mv.run_id)
info = {
    "model_name": MODEL_NAME,
    "version": mv.version,
    "stage": mv.current_stage,
    "run_id": mv.run_id,
    "metrics": run.data.metrics,
    "params": run.data.params,
}
with open(os.path.join(OUT_DIR, "model_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"Exported {MODEL_NAME} v{mv.version} ({STAGE}, run {mv.run_id}) -> {OUT_DIR}/telco_churn_model.pkl")
