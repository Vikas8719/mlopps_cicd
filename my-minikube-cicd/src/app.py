from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from utils import predict_message, log_prediction_to_mlflow

app = FastAPI()

class SMS(BaseModel):
    message: str

@app.post("/predict")
def predict_api(payload: SMS):
    pred, label = predict_message(payload.message)

    # MLflow Logging
    with mlflow.start_run(nested=True):
        log_prediction_to_mlflow(payload.message, pred)

    return {
        "prediction": label,
        "raw": int(pred),
        "input": payload.message
    }
