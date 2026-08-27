"""Detects a degraded Production model and rolls back to the most recent
previous version that still meets the promotion criteria, logging the
incident for the postmortem record (Lab 5.3)."""
import datetime
import json
import os
import subprocess
import sys

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Telco_Churn_Model"
F1_MIN = 0.55
INCIDENT_LOG = "artifacts/incident_log.jsonl"

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()


def f1_of(mv):
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("f1_score")


def log_incident(entry):
    os.makedirs("artifacts", exist_ok=True)
    with open(INCIDENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        print("No Production version found. Nothing to check.")
        return

    prod = max(production, key=lambda v: int(v.version))
    prod_f1 = f1_of(prod)
    print(f"Current Production: version {prod.version}, f1_score={prod_f1}")

    if prod_f1 is not None and prod_f1 >= F1_MIN:
        print("Production model meets the F1 threshold. No rollback needed.")
        return

    candidates = [
        v for v in versions
        if v.version != prod.version and f1_of(v) is not None and f1_of(v) >= F1_MIN
    ]
    if not candidates:
        print("ALERT: Production model is degraded and no eligible previous version was found.")
        sys.exit(1)

    target = max(candidates, key=lambda v: int(v.version))
    target_f1 = f1_of(target)
    print(f"Rolling back to version {target.version} (f1_score={target_f1})")

    client.transition_model_version_stage(
        name=MODEL_NAME, version=prod.version, stage="Archived",
    )
    client.transition_model_version_stage(
        name=MODEL_NAME, version=target.version, stage="Production",
    )

    incident = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action": "rollback",
        "model_name": MODEL_NAME,
        "failed_version": prod.version,
        "failed_version_f1": prod_f1,
        "restored_version": target.version,
        "restored_version_f1": target_f1,
        "reason": f"Production f1_score {prod_f1} fell below minimum threshold {F1_MIN}",
    }
    log_incident(incident)
    print(f"Incident logged to {INCIDENT_LOG}")

    print("Re-exporting the restored model for serving...")
    subprocess.run([sys.executable, "src/export_model.py"], check=True,
                    env={**os.environ, "MLFLOW_TRACKING_URI": tracking_uri})


if __name__ == "__main__":
    main()
