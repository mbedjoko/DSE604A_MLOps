import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 170.5,
}


@pytest.fixture
def client():
    import app as app_module
    app_module.app.testing = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_home_reports_model_info(client):
    resp = client.get("/")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["model"] == "Telco_Churn_Model"


def test_predict_single_record(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["churn_prediction"] in (0, 1)
    assert body["churn_label"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_batch(client):
    resp = client.post("/predict", json=[SAMPLE_CUSTOMER, SAMPLE_CUSTOMER])
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body) == 2


def test_predict_missing_fields(client):
    incomplete = {"gender": "Female"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_rejects_non_json(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
