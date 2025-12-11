import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def main():
    mlflow.set_tracking_uri("http://mlflow:5000")  # override in env if needed
    iris = load_iris()
    X = iris.data
    y = iris.target

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "rf_model")
        mlflow.log_param("n_estimators", 10)
        mlflow.log_metric("dummy_acc", 0.95)

if __name__ == "__main__":
    main()
