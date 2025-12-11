from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import os

app = FastAPI(title="FastAPI MLflow App")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
mlflow.set_tracking_uri(MLFLOW_URI)


# ---------- Request Body ----------
class PredictData(BaseModel):
    data: dict


# ---------- Health Check ----------
@app.get("/health")
def health_check():
    return {"status": "ok", "mlflow_uri": MLFLOW_URI}


# ---------- Predict Endpoint ----------
@app.post("/predict")
def predict(payload: PredictData):
    """This is a dummy predict endpoint.
       Add your model loading + prediction code inside."""
    
    # Example MLflow tracking
    with mlflow.start_run(nested=True):
        mlflow.log_param("endpoint", "predict")
        mlflow.log_param("input_size", len(payload.data))

    # Dummy output
    return {
        "prediction": "dummy_prediction",
        "received": payload.data
    }


# --------- Root Endpoint ---------
@app.get("/")
def root():
    return {"message": "FastAPI MLflow App Running"}
