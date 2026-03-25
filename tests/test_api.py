from unittest.mock import patch
from fastapi.testclient import TestClient

def test_health_check():
    with patch("src.api.main.model") as mock_model:
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

# test_api.py loads the full BERT model which is 419MB — stored in saved_model/ which is in .gitignore. 
# The CI machine won't have it.

def test_predict_endpoint():
    with patch("src.api.main.model") as mock_model:
        mock_model.predict.return_value = ("lost_or_stolen_card", 94.32)
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": "I lost my card"})
        assert response.status_code == 200
        assert "predicted_intent" in response.json()