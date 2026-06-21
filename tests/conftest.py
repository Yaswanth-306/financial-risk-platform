import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path as PathlibPath
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

@pytest.fixture(scope="session", autouse=True)
def setup_models():
    ml_models_dir = PathlibPath("ml_models")
    ml_models_dir.mkdir(exist_ok=True)
    
    scaler = StandardScaler()
    X_dummy = np.random.randn(100, 9)
    scaler.fit(X_dummy)
    joblib.dump(scaler, str(ml_models_dir / "scaler.pkl"))
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, str(ml_models_dir / "lgbm_model.pkl"))
    
    yield

@pytest.fixture
def mock_db_data():
    return {
        "tickers": [("AAPL",), ("TSLA",), ("GOOGL",), ("MSFT",), ("AMZN",),
                   ("META",), ("NVDA",), ("JPM",), ("BAC",), ("GS",)],
        "risk_summary": [
            ("AAPL", 0.015, 0.3, 100),
            ("TSLA", 0.025, 0.7, 95),
        ],
        "features": [0.02, 150.5, 149.8, 148.5, 0.015, 0.012, 2.5, 1.1, 1.02]
    }

@pytest.fixture(autouse=True)
def mock_database(mock_db_data, monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    def mock_execute(query, *args, **kwargs):
        if "DISTINCT ticker" in query:
            mock_cursor.fetchall.return_value = mock_db_data["tickers"]
        elif "GROUP BY ticker" in query:
            mock_cursor.fetchall.return_value = mock_db_data["risk_summary"]
        elif "WHERE ticker" in query:
            mock_cursor.fetchall.return_value = [mock_db_data["features"]]
        return None
    
    mock_cursor.execute.side_effect = mock_execute
    
    def mock_connect(*args, **kwargs):
        return mock_conn
    
    monkeypatch.setattr("psycopg2.connect", mock_connect)
