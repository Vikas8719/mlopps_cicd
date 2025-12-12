import unittest
from utils import load_model_and_vectorizer, predict_message
from fastapi.testclient import TestClient
from src.app import app  # ensure app.py is inside src/

client = TestClient(app)

class TestSpamModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model + vectorizer once for all tests
        cls.vectorizer, cls.model = load_model_and_vectorizer()

    def test_model_loading(self):
        """Check if model + vectorizer load successfully"""
        self.assertIsNotNone(self.vectorizer, "Vectorizer not loaded")
        self.assertIsNotNone(self.model, "Model not loaded")

    def test_prediction_function(self):
        """Check prediction output structure"""
        msg = "Congratulations! You won a free ticket!"
        pred, label = predict_message(msg)
        self.assertIn(pred, [0, 1])
        self.assertIn(label, ["HAM", "SPAM"])

    def test_fastapi_predict_endpoint(self):
        """Test FastAPI /predict API"""
        response = client.post(
            "/predict",
            json={"message": "Win a free lottery now!"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["prediction"], ["HAM", "SPAM"])
        self.assertEqual(response.json()["raw"], 0 if response.json()["prediction"]=="HAM" else 1)

if __name__ == "__main__":
    unittest.main()
