import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSE604A_First_Experiment")

with mlflow.start_run():
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("precision", 0.90)
    print("Experiment completed successfully.")
