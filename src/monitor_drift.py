import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/telco_churn_model.pkl"

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

KS_ALERT_THRESHOLD = 0.10   # p-value below this -> statistically significant drift
PSI_ALERT_THRESHOLD = 0.20  # population stability index above this -> significant drift
ACCURACY_DROP_ALERT = 0.05  # accuracy drop vs reference -> significant drift


def population_stability_index(reference, current, bins=10):
    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def simulate_production_batch(df, seed=123):
    """Simulate a later production batch by resampling with a shifted
    tenure/charges distribution, mimicking a customer-base shift over time."""
    rng = np.random.default_rng(seed)
    batch = df.sample(frac=0.3, random_state=seed).copy()
    batch["tenure"] = np.clip(batch["tenure"] - rng.normal(8, 3, len(batch)), 0, None)
    batch["MonthlyCharges"] = batch["MonthlyCharges"] * rng.normal(1.15, 0.05, len(batch))
    return batch


def numerical_drift_report(reference, current):
    report = {}
    for col in NUMERICAL_COLS:
        stat, p_value = ks_2samp(reference[col], current[col])
        psi = population_stability_index(reference[col].values, current[col].values)
        report[col] = {
            "ks_statistic": round(float(stat), 4),
            "ks_p_value": round(float(p_value), 4),
            "psi": round(psi, 4),
            "drift_detected": bool(p_value < KS_ALERT_THRESHOLD or psi > PSI_ALERT_THRESHOLD),
        }
    return report


def categorical_drift_report(reference, current):
    report = {}
    for col in CATEGORICAL_COLS:
        ref_dist = reference[col].value_counts(normalize=True)
        cur_dist = current[col].value_counts(normalize=True)
        categories = ref_dist.index.union(cur_dist.index)
        ref_aligned = ref_dist.reindex(categories, fill_value=1e-6)
        cur_aligned = cur_dist.reindex(categories, fill_value=1e-6)
        psi = float(np.sum((cur_aligned - ref_aligned) * np.log(cur_aligned / ref_aligned)))
        report[col] = {
            "psi": round(psi, 4),
            "drift_detected": bool(psi > PSI_ALERT_THRESHOLD),
        }
    return report


def performance_report(model, reference, current):
    def score(subset):
        X = subset.drop("Churn", axis=1)
        y = subset["Churn"]
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred), f1_score(y, y_pred)

    ref_acc, ref_f1 = score(reference)
    cur_acc, cur_f1 = score(current)
    accuracy_drop = ref_acc - cur_acc

    return {
        "reference_accuracy": round(ref_acc, 4),
        "current_accuracy": round(cur_acc, 4),
        "accuracy_drop": round(accuracy_drop, 4),
        "reference_f1": round(ref_f1, 4),
        "current_f1": round(cur_f1, 4),
        "performance_drift_detected": bool(accuracy_drop > ACCURACY_DROP_ALERT),
    }


def main():
    df = load_and_prepare()
    reference, holdout = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])
    current = simulate_production_batch(holdout)

    model = joblib.load(MODEL_PATH)

    num_report = numerical_drift_report(reference, current)
    cat_report = categorical_drift_report(reference, current)
    perf_report = performance_report(model, reference, current)

    any_data_drift = any(v["drift_detected"] for v in num_report.values()) or \
        any(v["drift_detected"] for v in cat_report.values())

    result = {
        "reference_size": len(reference),
        "current_batch_size": len(current),
        "numerical_feature_drift": num_report,
        "categorical_feature_drift": cat_report,
        "model_performance": perf_report,
        "data_drift_detected": any_data_drift,
        "model_drift_detected": perf_report["performance_drift_detected"],
        "alert": bool(any_data_drift or perf_report["performance_drift_detected"]),
    }

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/drift_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nReport written to {out_path}")
    if result["alert"]:
        print("ALERT: drift detected -> investigate and consider retraining.")
    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    main()
