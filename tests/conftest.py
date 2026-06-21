import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path as PathlibPath
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Create dummy ML models for testing"""
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

@pytest.fixture(autouse=True)
def mock_database_and_models(monkeypatch):
    """Mock database and model loading"""
    
    # Mock test data
    tickers_df = pd.DataFrame({"ticker": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
                                          "META", "NVDA", "JPM", "BAC", "GS"]})
    
    risk_summary_df = pd.DataFrame({
        "ticker": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "JPM", "BAC", "GS"],
        "avg_volatility": [0.015, 0.025, 0.012, 0.018, 0.020, 0.022, 0.028, 0.010, 0.011, 0.012],
        "risk_rate": [0.3, 0.7, 0.4, 0.5, 0.6, 0.65, 0.8, 0.2, 0.25, 0.3],
        "data_points": [100, 95, 98, 102, 97, 96, 94, 105, 103, 101]
    })
    
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
    
    # Mock pd.read_sql to return DataFrames
    def mock_read_sql(query, conn, *args, **kwargs):
        if "DISTINCT ticker" in query:
            return tickers_df
        elif "GROUP BY ticker" in query:
            return risk_summary_df
        elif "WHERE ticker" in query:
            return features_df
        return pd.DataFrame()
    
    monkeypatch.setattr("pandas.read_sql", mock_read_sql)
    
    # Mock psycopg2.connect to return a mock connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = risk_summary_df.values.tolist()
    
    def mock_connect(*args, **kwargs):
        return mock_conn
    
    monkeypatch.setattr("psycopg2.connect", mock_connect)
    
    # Mock mlflow to return our dummy model
    def mock_load_model(*args, **kwargs):
        return joblib.load("ml_models/lgbm_model.pkl")
    
    monkeypatch.setattr("mlflow.sklearn.load_model", mock_load_model)
    
    # Mock RAG system to return dummy data
    def mock_load_index():
        import faiss
        embedder = MagicMock()
        index = MagicMock()
        chunks = [{"text": "dummy chunk", "ticker": "AAPL"}]
        return index, chunks, embedder
    
    monkeypatch.setattr("genai.rag_system.load_index", mock_load_index)
    
    def mock_query_rag(*args, **kwargs):
        return {
            "answer": "Based on SEC filings, the main risk factors are...",
            "sources": ["AAPL"],
            "chunks": ["Risk factor 1", "Risk factor 2"]
        }
    
    monkeypatch.setattr("genai.rag_system.query_rag", mock_query_rag)
