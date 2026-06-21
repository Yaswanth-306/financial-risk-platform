import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path as PathlibPath
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys
import joblib

# Create dummy models FIRST
ml_models_dir = PathlibPath("ml_models")
ml_models_dir.mkdir(exist_ok=True)

_dummy_scaler = StandardScaler()
X_dummy = np.random.randn(100, 9)
_dummy_scaler.fit(X_dummy)

_dummy_model = LogisticRegression(random_state=42, max_iter=1000)
y_dummy = np.random.randint(0, 2, 100)
_dummy_model.fit(X_dummy, y_dummy)

# Patch joblib.load GLOBALLY before importing the app
_original_joblib_load = joblib.load

def _mock_joblib_load(path, *args, **kwargs):
    path_str = str(path)
    if "scaler.pkl" in path_str:
        return _dummy_scaler
    elif "lgbm_model.pkl" in path_str:
        return _dummy_model
    return _original_joblib_load(path, *args, **kwargs)

joblib.load = _mock_joblib_load

# Mock genai package and submodules BEFORE importing the app
def _mock_load_index():
    embedder = MagicMock()
    index = MagicMock()
    chunks = [{"text": "dummy chunk", "ticker": "AAPL"}]
    return index, chunks, embedder

def _mock_query_rag(*args, **kwargs):
    return {
        "answer": "Based on SEC filings, the main risk factors are market volatility and regulatory changes.",
        "sources": ["AAPL"],
        "chunks": ["Risk factor 1", "Risk factor 2"]
    }

def _mock_investigate(*args, **kwargs):
    return {
        "key_risks": ["Market Risk", "Regulatory Risk"],
        "analysis": "Detailed risk analysis"
    }

# Create genai package mock
genai_mock = MagicMock()
genai_mock.agents.investigate = _mock_investigate
genai_mock.rag_system.load_index = _mock_load_index
genai_mock.rag_system.query_rag = _mock_query_rag

# Register all genai modules
sys.modules['genai'] = genai_mock
sys.modules['genai.agents'] = genai_mock.agents
sys.modules['genai.rag_system'] = genai_mock.rag_system

# NOW import the app (after mocks are patched)
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app"""
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_database(monkeypatch):
    """Mock all database and external service calls"""

    # Mock pandas.read_sql
    tickers_df = pd.DataFrame({"ticker": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
                                          "META", "NVDA", "JPM", "BAC", "GS"]})

    risk_summary_data = [
        ("AAPL", 0.015, 0.3, 100),
        ("TSLA", 0.025, 0.7, 95),
        ("GOOGL", 0.012, 0.4, 98),
        ("MSFT", 0.018, 0.5, 102),
        ("AMZN", 0.020, 0.6, 97),
        ("META", 0.022, 0.65, 96),
        ("NVDA", 0.028, 0.8, 94),
        ("JPM", 0.010, 0.2, 105),
        ("BAC", 0.011, 0.25, 103),
        ("GS", 0.012, 0.3, 101),
    ]

    risk_summary_df = pd.DataFrame(risk_summary_data, columns=["ticker", "avg_volatility", "risk_rate", "data_points"])

    features_df = pd.DataFrame({
        "daily_return": [0.02],
        "ma_7": [150.5],
        "ma_14": [149.8],
        "ma_30": [148.5],
        "volatility_7d": [0.015],
        "volatility_30d": [0.012],
        "high_low_range": [2.5],
        "volume_ratio": [1.1],
        "price_vs_ma30": [1.02]
    })

    def mock_read_sql(query, conn, *args, **kwargs):
        if "DISTINCT ticker" in query:
            return tickers_df
        elif "GROUP BY ticker" in query:
            return risk_summary_df
        elif "WHERE ticker" in query:
            return features_df
        return pd.DataFrame()

    monkeypatch.setattr("pandas.read_sql", mock_read_sql)

    # Mock psycopg2.connect
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = risk_summary_data

    def mock_connect(*args, **kwargs):
        return mock_conn

    monkeypatch.setattr("psycopg2.connect", mock_connect)
