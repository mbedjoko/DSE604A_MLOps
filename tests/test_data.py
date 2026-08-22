import pandas as pd

def test_churn_mapping():
    df = pd.DataFrame({"Churn": ["Yes", "No", "Yes"]})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    assert df["Churn"].tolist() == [1, 0, 1]

def test_total_charges_conversion():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5"]})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert df["TotalCharges"].isna().sum() == 1
