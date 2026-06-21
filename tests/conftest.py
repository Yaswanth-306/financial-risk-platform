import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path as PathlibPath
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import sys

# Create dummy models BEFORE any imports
def create_dummy_models():
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
    
    return scaler, model

# Create models at pytest startup (BEFORE any test collection)
scaler, model = create_dummy_models()

@pytest.fixture(scope="session", autouse=True)
def setup_session():
    yield

@pytest.fixture(autouse=True)
def mock_everything(monkeypatch):
    """Mock all external dependencies"""
    
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
    
    # Mock RAG load_index
    def mock_load_index():
        embedder = MagicMock()
        index = MagicMock()
        chunks = [{"text": "dummy chunk", "ticker": "AAPL"}]
        return index, chunks, embedder
    
    monkeypatch.setattr("genai.rag_system.load_index", mock_load_index)
    
    # Mock query_rag
    def mock_query_rag(*args, **kwargs):
        return {
            "answer": "Based on SEC filings, the main risk factors are market volatility and regulatory changes.",
            "sources": ["AAPL"],
            "chunks": ["Risk factor 1", "Risk factor 2"]
        }
    
    monkeypatch.setattr("genai.rag_system.query_rag", mock_query_rag)
