import pickle
import mlflow
import os
from pathlib import Path

# -------------------------
# File Paths (Docker/K8s safe)
# -------------------------
from pathlib import Path
import os

# -------------------------
# File Paths (Docker/K8s safe)
# -------------------------
BASE_DIR = Path(__file__).parent  # src folder ke andar
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# -------------------------
# Load model + vectorizer only once (FAST)
# -------------------------

# Global cache (loaded once)
_vectorizer = None
_model = None

def load_model_and_vectorizer():
    """Load saved model and vectorizer once (cached)."""

    global _vectorizer, _model

    if _vectorizer is not None and _model is not None:
        return _vectorizer, _model

    # File checks
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file missing: {VECTORIZER_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

    # Load vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    # Load ML model
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    return _vectorizer, _model


# -------------------------
# Prediction Function
# -------------------------
def predict_message(message: str):
    """Predict SPAM or HAM from input text."""

    vectorizer, model = load_model_and_vectorizer()

    # Vectorize
    msg_vec = vectorizer.transform([message])

    # Predict
    pred = model.predict(msg_vec)[0]   # returns 0 or 1

    # Convert numeric → label
    label = "SPAM" if pred == 1 else "HAM"

    return pred, label


# -------------------------
# MLflow Logging Utility
# -------------------------
def log_prediction_to_mlflow(message: str, prediction: int):
    """
    Logs prediction event to MLflow.
    Safe for Docker, Minikube, AWS, GCP, Azure.
    """
    try:
        mlflow.log_param("input_text_length", len(message))
        mlflow.log_param("prediction_raw", int(prediction))
        mlflow.log_text(message, "input_text.txt")
    except Exception as e:
        # Avoid app crash during logging
        print(f"MLflow logging error: {e}")
