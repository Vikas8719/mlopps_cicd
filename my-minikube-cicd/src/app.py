from flask import Flask, jsonify, request
import os
import mlflow

app = Flask(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # placeholder: do model inference here
    # log request to mlflow as an example
    with mlflow.start_run(nested=True):
        mlflow.log_param("example", "predict_called")
        mlflow.log_metric("dummy_metric", 0.123)
    return jsonify({"prediction":"dummy", "received": data})
