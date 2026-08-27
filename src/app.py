import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "models/telco_churn_model.pkl")
MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", "models/model_info.json")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)

model_info = {}
if os.path.exists(MODEL_INFO_PATH):
    with open(MODEL_INFO_PATH) as f:
        model_info = json.load(f)

REQUIRED_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


@app.route("/")
def home():
    return jsonify({
        "service": "Telco Churn Model API",
        "model": model_info.get("model_name"),
        "version": model_info.get("version"),
        "stage": model_info.get("stage"),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model-info")
def info():
    return jsonify(model_info)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    records = payload if isinstance(payload, list) else [payload]

    missing = [f for f in REQUIRED_FIELDS if any(f not in r for r in records)]
    if missing:
        return jsonify({"error": f"Missing required fields: {sorted(set(missing))}"}), 400

    df = pd.DataFrame.from_records(records)[REQUIRED_FIELDS]

    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for i, pred in enumerate(preds):
        entry = {"churn_prediction": int(pred), "churn_label": "Yes" if pred == 1 else "No"}
        if probs is not None:
            entry["churn_probability"] = round(float(probs[i]), 4)
        results.append(entry)

    return jsonify(results if isinstance(payload, list) else results[0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
