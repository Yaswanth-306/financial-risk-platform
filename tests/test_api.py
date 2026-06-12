"""
Comprehensive pytest tests for FastAPI endpoints
Tests all 7 endpoints with various scenarios and edge cases
"""

import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app"""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint"""

    def test_health_check_status(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client):
        """Test health endpoint returns proper structure"""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_has_model_loaded(self, client):
        """Test health endpoint includes model_loaded flag"""
        response = client.get("/health")
        data = response.json()

        assert "model_loaded" in data
        assert data["model_loaded"] is True


class TestTickersEndpoint:
    """Tests for GET /tickers endpoint"""

    def test_tickers_returns_200(self, client):
        """Test tickers endpoint returns 200"""
        response = client.get("/tickers")
        assert response.status_code == 200

    def test_tickers_returns_list(self, client):
        """Test tickers endpoint returns a list in 'tickers' key"""
        response = client.get("/tickers")
        data = response.json()

        assert isinstance(data["tickers"], list)

    def test_tickers_has_10_items(self, client):
        """Test tickers endpoint returns exactly 10 tickers"""
        response = client.get("/tickers")
        data = response.json()

        assert len(data["tickers"]) == 10

    def test_tickers_contains_expected_symbols(self, client):
        """Test tickers include expected stock symbols"""
        response = client.get("/tickers")
        data = response.json()
        tickers = data["tickers"]

        expected = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN',
                   'META', 'NVDA', 'JPM', 'BAC', 'GS']

        assert all(ticker in tickers for ticker in expected)


class TestRiskSummaryEndpoint:
    """Tests for GET /risk-summary endpoint"""

    def test_risk_summary_returns_200(self, client):
        """Test risk-summary endpoint returns 200"""
        response = client.get("/risk-summary")
        assert response.status_code == 200

    def test_risk_summary_returns_list(self, client):
        """Test risk-summary endpoint returns a list in 'summary' key"""
        response = client.get("/risk-summary")
        data = response.json()

        assert isinstance(data["summary"], list)

    def test_risk_summary_has_10_items(self, client):
        """Test risk-summary endpoint returns data for 10 stocks"""
        response = client.get("/risk-summary")
        data = response.json()

        assert len(data["summary"]) == 10

    def test_risk_summary_item_structure(self, client):
        """Test each risk summary item has required fields"""
        response = client.get("/risk-summary")
        data = response.json()

        for item in data["summary"]:
            assert "ticker" in item
            assert "risk_rate" in item or "avg_volatility" in item
            assert "ticker" in item


class TestPredictEndpoint:
    """Tests for POST /predict endpoint"""

    def test_predict_valid_ticker_returns_200(self, client):
        """Test predict endpoint with valid ticker returns 200"""
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 200

    def test_predict_valid_ticker_response_structure(self, client):
        """Test predict endpoint returns proper structure"""
        response = client.post("/predict", json={"ticker": "AAPL"})
        data = response.json()

        assert "ticker" in data
        assert "risk_score" in data
        assert "risk_label" in data
        assert "confidence" in data

    def test_predict_valid_ticker_values(self, client):
        """Test predict endpoint returns valid values"""
        response = client.post("/predict", json={"ticker": "AAPL"})
        data = response.json()

        assert data["ticker"] == "AAPL"
        assert data["risk_label"] in ["HIGH", "LOW", "HIGH RISK", "LOW RISK"]
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["risk_score"] <= 1

    def test_predict_any_ticker_returns_200(self, client):
        """Test predict endpoint returns 200 for any ticker"""
        response = client.post("/predict", json={"ticker": "INVALID"})
        assert response.status_code == 200

    def test_predict_returns_response_for_any_ticker(self, client):
        """Test predict endpoint returns valid response for any ticker"""
        response = client.post("/predict", json={"ticker": "INVALID"})
        data = response.json()

        assert "ticker" in data
        assert "risk_label" in data

    def test_predict_missing_ticker_returns_422(self, client):
        """Test predict endpoint with missing ticker returns 422"""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_different_tickers(self, client):
        """Test predict endpoint works for multiple valid tickers"""
        tickers = ["MSFT", "GOOGL", "TSLA", "AMZN"]

        for ticker in tickers:
            response = client.post("/predict", json={"ticker": ticker})
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == ticker


class TestExplainEndpoint:
    """Tests for POST /explain endpoint"""

    def test_explain_valid_ticker_returns_200(self, client):
        """Test explain endpoint with valid ticker returns 200"""
        response = client.post("/explain", json={"ticker": "TSLA"})
        assert response.status_code == 200

    def test_explain_response_structure(self, client):
        """Test explain endpoint returns proper structure"""
        response = client.post("/explain", json={"ticker": "TSLA"})
        data = response.json()

        assert "ticker" in data
        assert "top_features" in data

    def test_explain_has_features_list(self, client):
        """Test explain endpoint returns list of features"""
        response = client.post("/explain", json={"ticker": "TSLA"})
        data = response.json()

        assert isinstance(data["top_features"], list)
        assert len(data["top_features"]) > 0

    def test_explain_feature_structure(self, client):
        """Test each feature has feature name and impact"""
        response = client.post("/explain", json={"ticker": "TSLA"})
        data = response.json()

        for feature in data["top_features"]:
            assert "feature" in feature or "name" in feature or "feature_name" in feature
            assert "impact" in feature or "importance" in feature or "shap_value" in feature

    def test_explain_any_ticker_returns_200(self, client):
        """Test explain endpoint returns 200 for any ticker"""
        response = client.post("/explain", json={"ticker": "INVALID"})
        assert response.status_code == 200


class TestAskEndpoint:
    """Tests for POST /ask endpoint (RAG endpoint)"""

    def test_ask_valid_question_returns_200(self, client):
        """Test ask endpoint with valid question returns 200"""
        response = client.post("/ask", json={
            "question": "What are Tesla risk factors?"
        })
        assert response.status_code == 200

    def test_ask_response_structure(self, client):
        """Test ask endpoint returns proper structure"""
        response = client.post("/ask", json={
            "question": "What are Tesla risk factors?"
        })
        data = response.json()

        assert "answer" in data
        assert "sources" in data

    def test_ask_answer_is_not_empty(self, client):
        """Test ask endpoint returns non-empty answer"""
        response = client.post("/ask", json={
            "question": "What are Tesla risk factors?"
        })
        data = response.json()

        assert len(data["answer"]) > 0

    def test_ask_sources_is_list(self, client):
        """Test ask endpoint returns list of sources"""
        response = client.post("/ask", json={
            "question": "What are Tesla risk factors?"
        })
        data = response.json()

        assert isinstance(data["sources"], list)

    def test_ask_multiple_questions(self, client):
        """Test ask endpoint with various questions"""
        questions = [
            "What are Apple risk factors?",
            "How do banks manage liquidity risk?",
            "What market risks does Google face?",
        ]

        for question in questions:
            response = client.post("/ask", json={"question": question})
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert len(data["answer"]) > 0

    def test_ask_missing_question_returns_422(self, client):
        """Test ask endpoint with missing question returns 422"""
        response = client.post("/ask", json={})
        assert response.status_code == 422


class TestEndpointErrorHandling:
    """Tests for error handling across endpoints"""

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_wrong_method_returns_405(self, client):
        """Test wrong HTTP method returns 405"""
        response = client.post("/health")
        assert response.status_code == 405


class TestEndpointPerformance:
    """Tests for endpoint performance and reliability"""

    def test_predict_response_time(self, client):
        """Test predict endpoint responds in reasonable time"""
        import time

        start = time.time()
        response = client.post("/predict", json={"ticker": "AAPL"})
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0  # Should respond within 5 seconds

    def test_explain_response_time(self, client):
        """Test explain endpoint responds in reasonable time"""
        import time

        start = time.time()
        response = client.post("/explain", json={"ticker": "TSLA"})
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0  # Should respond within 5 seconds

    def test_ask_response_time(self, client):
        """Test ask endpoint responds in reasonable time"""
        import time

        start = time.time()
        response = client.post("/ask", json={
            "question": "What is risk?"
        })
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 60.0  # RAG embedding and LLM takes longer


class TestIntegration:
    """Integration tests combining multiple endpoints"""

    def test_workflow_get_tickers_then_predict(self, client):
        """Test getting tickers then predicting for first one"""
        # Get tickers
        tickers_response = client.get("/tickers")
        response_data = tickers_response.json()
        tickers = response_data["tickers"]

        # Predict for first ticker
        ticker = tickers[0]
        predict_response = client.post("/predict", json={"ticker": ticker})

        assert predict_response.status_code == 200
        data = predict_response.json()
        assert data["ticker"] == ticker

    def test_workflow_predict_then_explain(self, client):
        """Test predicting then getting explanation"""
        ticker = "AAPL"

        # Predict
        predict_response = client.post("/predict", json={"ticker": ticker})
        assert predict_response.status_code == 200

        # Explain
        explain_response = client.post("/explain", json={"ticker": ticker})
        assert explain_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
