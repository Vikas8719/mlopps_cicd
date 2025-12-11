import pickle
import mlflow
import os

# Paths (inside Docker/Kubernetes container)
MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# -------------------------
# Load TF-IDF + Model
# -------------------------
def load_model_and_vectorizer():
    """Load saved model and vectorizer."""
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


# -------------------------
# Prediction Function
# -------------------------
def predict_message(message: str):
    """Predict spam/ham from text message."""
    
    vectorizer, model = load_model_and_vectorizer()

    # Vectorize input text
    msg_vec = vectorizer.transform([message])

    # Prediction
    pred = model.predict(msg_vec)[0]

    # Human readable output
    label = "SPAM" if pred == 1 else "HAM"

    return pred, label


# -------------------------
# MLflow Logging Utility
# -------------------------
def log_prediction_to_mlflow(message: str, prediction: int):
    """Logs prediction event into MLflow."""
    mlflow.log_param("input_length", len(message))
    mlflow.log_param("prediction_raw", int(prediction))
