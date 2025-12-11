import unittest
from utils import load_model_and_vectorizer, predict_message
from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


class TestSpamModel(unittest.TestCase):

    def test_model_loading(self):
        """Check if model + vectorizer load successfully"""
        vectorizer, model = load_model_and_vectorizer()
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(model)

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


if __name__ == "__main__":
    unittest.main()
