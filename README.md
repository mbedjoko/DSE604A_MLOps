# DSE604A — Telco Churn MLOps Pipeline

A complete MLOps pipeline for telecom customer churn prediction: experiment
tracking, model comparison, automated promotion, rollback, containerised
serving, Kubernetes deployment, and drift monitoring.

Course: Cloud Data Computing (DSE604A / MLOps & Industrialization)

## Project layout

```
DSE604A_MLOps/
├── data/                      Telco-Customer-Churn.csv (cached after first download)
├── models/                    exported Production model for serving
│   ├── telco_churn_model.pkl
│   └── model_info.json
├── src/
│   ├── telco_promotion.py     train, compare, evaluate, promote (Labs 1-5)
│   ├── export_model.py        pulls the Production model out of the registry to disk
│   ├── app.py                 Flask prediction API (Lab 9)
│   ├── monitor_drift.py       data/model drift detection (Lab 14)
│   ├── simulate_bad_deployment.py  injects a bad Production version, for drills
│   └── rollback.py            detects a degraded Production model and rolls back (Lab 5.3)
├── docker/
│   └── Dockerfile.serve       serving image (Flask API)
├── Dockerfile                 training image (runs telco_promotion.py)
├── kubernetes/
│   ├── deployment.yaml        2-replica Deployment for the serving API
│   └── service.yaml           NodePort Service exposing it
├── tests/                     pytest suite, run in CI on every push/PR
├── .github/workflows/ci.yml   GitHub Actions CI
└── artifacts/                 drift_report.json, incident_log.jsonl
```

## 1. Environment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2-5. Experiment tracking, comparison, registry, promotion

`src/telco_promotion.py` is the single entry point that covers Labs 1-5:

- loads and cleans the Telco Customer Churn dataset
- trains three candidate models (Logistic Regression, Random Forest, XGBoost) behind a
  shared preprocessing pipeline
- logs parameters, metrics (F1-score, latency, model size, fairness disparity) and
  the model artifact to MLflow for every run
- selects the best run by F1-score and checks it against the promotion criteria
- registers and promotes the winning model to `Production` in the MLflow Model Registry

Start the tracking server (backed by the existing `mlflow.db`), then run the pipeline:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python3 src/telco_promotion.py
```

Open the UI at http://127.0.0.1:5000 to inspect experiments, runs, and the
`Telco_Churn_Model` registry entry.

### Rollback drill (Lab 5.3)

```bash
python3 src/simulate_bad_deployment.py   # pushes a deliberately weak model to Production
python3 src/rollback.py                  # detects it, rolls back, logs the incident
```

`rollback.py` reads the F1-score of the current Production version; if it falls below
the promotion threshold (0.55) it archives that version, promotes the most recent
previous version that still meets the threshold, re-exports the restored model for
serving, and appends a record to `artifacts/incident_log.jsonl`:

```json
{"timestamp": "...", "action": "rollback", "model_name": "Telco_Churn_Model",
 "failed_version": "3", "failed_version_f1": 0.0,
 "restored_version": "1", "restored_version_f1": 0.608,
 "reason": "Production f1_score 0.0 fell below minimum threshold 0.55"}
```

This is why rollback matters operationally: it turns "a bad model reached production"
from an outage into a scripted, auditable, few-second recovery.

## 6. Git / GitHub

Standard branch-per-feature workflow:

```bash
git checkout -b feature/<name>
git add <files>
git commit -m "..."
git push -u origin feature/<name>
# open a pull request, review, merge into main
```

`.gitignore` excludes `venv/`, `__pycache__/`, `mlruns/`, `mlartifacts/`, `.env`, `*.db`.
`models/*.pkl` and `model_info.json` **are** tracked, since the serving image and CI
tests both depend on them being present without needing a live tracking server.

## 7. Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs
`requirements.txt` and runs `pytest`. The suite covers data-cleaning logic
(`test_data.py`), the prediction API (`test_app.py`), and the drift-scoring math
(`test_monitor_drift.py`) — all self-contained, no server required.

## 8-9. Containerisation and model serving

Two images, two concerns:

- `Dockerfile` (root) — training: runs the full pipeline in Lab 2-5.
- `docker/Dockerfile.serve` — serving: a minimal Flask API over the exported model.

```bash
# export the current Production model to models/telco_churn_model.pkl
python3 src/export_model.py

# build and run the serving image
docker build -f docker/Dockerfile.serve -t dse604a-telco-serve:1.0 .
docker run -p 8080:8080 dse604a-telco-serve:1.0
```

API surface:

| Route          | Method | Purpose                          |
|----------------|--------|-----------------------------------|
| `/`            | GET    | model name/version/stage          |
| `/health`      | GET    | liveness check                    |
| `/model-info`  | GET    | full metadata for the loaded model|
| `/predict`     | POST   | churn prediction (single or batch)|

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 170.5
}'
# -> {"churn_label":"Yes","churn_prediction":1,"churn_probability":0.626}
```

## 10-11. Kubernetes deployment and service

```bash
minikube image load dse604a-telco-serve:1.0
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods -l app=telco-churn-model
kubectl get svc telco-churn-model-service
```

`kubernetes/deployment.yaml` runs 2 replicas of the serving container with
readiness/liveness probes on `/health`. `kubernetes/service.yaml` exposes them
through a single stable `NodePort` Service — Pods are ephemeral and their IPs
change on every restart or reschedule, so clients (and other services) target
the Service's fixed cluster IP/port instead of a Pod directly; the Service also
load-balances requests across whichever replicas are currently healthy.

```bash
curl http://$(minikube ip):$(kubectl get svc telco-churn-model-service -o jsonpath='{.spec.ports[0].nodePort}')/health
```

## 12. Scaling

```bash
kubectl scale deployment telco-churn-model --replicas=5
kubectl get pods -l app=telco-churn-model
```

Kubernetes schedules the additional Pod replicas across available cluster
capacity; the Service immediately starts load-balancing across all of them
once their readiness probe passes, without any client-side change. This is
how the deployment absorbs a traffic increase — horizontally, by adding more
identical, stateless replicas, rather than resizing a single instance.

## 13. End-to-end pipeline

```
telco_promotion.py  →  MLflow tracking + registry  →  export_model.py
        │                                                    │
        ▼                                                    ▼
  GitHub Actions CI                              docker/Dockerfile.serve
   (pytest on push)                                          │
                                                               ▼
                                                    Kubernetes Deployment
                                                       + Service (2+ pods)
                                                               │
                                                               ▼
                                                     monitor_drift.py (scheduled)
                                                               │
                                                     drift detected? → rollback.py
```

Every stage was exercised against this repository: training and promotion via
`telco_promotion.py`, tests via `pytest`, the image via `docker build`/`docker run`,
the cluster via a running `minikube` instance (`kubectl apply`, verified predictions
through the Service, scaled to 5 replicas and back to 2).

## 14. Monitoring and drift detection

```bash
python3 src/monitor_drift.py
```

`monitor_drift.py` splits a held-out reference sample from the training data,
simulates a later production batch (tenure and monthly-charges distributions
shifted, to mimic a changing customer base), and computes:

- **KS-test** (p-value) and **Population Stability Index (PSI)** per numerical feature
  (`tenure`, `MonthlyCharges`, `TotalCharges`)
- **PSI** per categorical feature
- **model performance drift**: accuracy/F1 of the current Production model on the
  reference sample vs. the simulated batch

A run is flagged (`alert: true`) when any feature's p-value drops below 0.10 or PSI
exceeds 0.20, or when accuracy drops by more than 0.05 versus the reference. The full
report is written to `artifacts/drift_report.json`. In production this would run on a
schedule (e.g. a daily cron job or a Kubernetes CronJob) against real incoming traffic
rather than a simulated batch, feeding an alert into the same `rollback.py` path used
in the Lab 5.3 drill.

Interpreting a drop from 95% training accuracy to 87% production accuracy: this
8-point gap, if consistent rather than one bad day, points to **data drift** (the
production feature distribution has moved away from what the model was trained on)
if input features have shifted, or **concept drift** (the relationship between features
and churn itself has changed, e.g. a new competitor or price change) if features look
stable but the target relationship hasn't. Both cases warrant retraining on recent data;
a pure implementation bug is unlikely to survive from training straight into a stable
low-accuracy plateau without also raising errors or failed predictions in the logs.

## 15. Capstone checklist

| Requirement                              | Status | Where |
|-------------------------------------------|--------|-------|
| Data-processing pipeline                   | done | `src/telco_promotion.py` |
| At least two model configurations           | done | 3 models: Logistic Regression, Random Forest, XGBoost |
| Experiment tracking in MLflow               | done | `telco_promotion.py`, viewable at :5000 |
| Select and register best model              | done | `telco_promotion.py` (criteria-driven) |
| Prediction API                              | done | `src/app.py` |
| Containerisation                            | done | `docker/Dockerfile.serve` |
| Kubernetes deployment                       | done | `kubernetes/deployment.yaml`, live-verified |
| CI testing                                  | done | `.github/workflows/ci.yml` |
| Monitoring indicators                       | done | `src/monitor_drift.py` |
| Drift and retraining strategy               | done | PSI/KS thresholds trigger `rollback.py`; sustained drift → rerun `telco_promotion.py` on fresh data |
| Rollback                                    | done | `src/rollback.py`, drilled via `simulate_bad_deployment.py`, logged to `artifacts/incident_log.jsonl` |

## Running the tests

```bash
pytest
```
