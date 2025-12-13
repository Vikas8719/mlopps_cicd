import joblib
import mlflow
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

_vectorizer = None
_model = None

def load_model_and_vectorizer():
    global _vectorizer, _model

    if _vectorizer and _model:
        return _vectorizer, _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer missing: {VECTORIZER_PATH}")

    _vectorizer = joblib.load(VECTORIZER_PATH)
    _model = joblib.load(MODEL_PATH)

    return _vectorizer, _model
