import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from genai.agents import investigate
import shap
import sys
sys.path.append('/home/chitr/financial-risk-platform')
from genai.rag_system import load_index, query_rag
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

app = FastAPI(
    title="Financial Risk Intelligence API",
    description="ML-powered stock risk prediction with explainability",
    version="1.0.0"
)

# ── Load model at startup ─────────────────────────────────
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

import joblib
print("Loading model and scaler...")
try:
    model = mlflow.sklearn.load_model("models:/FinancialRiskModel/latest")
    scaler = joblib.load("/home/chitr/financial-risk-platform/ml_models/scaler.pkl")
    print("✅ Model and scaler loaded successfully")
except (FileNotFoundError, OSError, PermissionError, Exception) as e:
    print(f"❌ Load error: {e}")
    model = None
    scaler = None
# Lazy-load RAG index on first /ask call
rag_index = None
rag_chunks = None
rag_embedder = None

def _load_rag_if_needed():
    global rag_index, rag_chunks, rag_embedder
    if rag_index is None:
        try:
            print("Loading RAG index...")
            rag_index, rag_chunks, rag_embedder = load_index()
            print("OK RAG index loaded")
            return True
        except Exception as e:
            print(f"ERR RAG error: {e}")
            return False
    return True

# ── Feature columns ───────────────────────────────────────
FEATURES = [
    'daily_return', 'ma_7', 'ma_14', 'ma_30',
    'volatility_7d', 'volatility_30d',
    'high_low_range', 'volume_ratio', 'price_vs_ma30'
]

# ── Pydantic schemas ──────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str
    daily_return: Optional[float] = None
    ma_7: Optional[float] = None
    ma_14: Optional[float] = None
    ma_30: Optional[float] = None
    volatility_7d: Optional[float] = None
    volatility_30d: Optional[float] = None
    high_low_range: Optional[float] = None
    volume_ratio: Optional[float] = None
    price_vs_ma30: Optional[float] = None

class PredictResponse(BaseModel):
    ticker: str
    risk_score: float
    risk_label: str
    confidence: float

class ExplainResponse(BaseModel):
    ticker: str
    top_features: list

# ── DB helper ─────────────────────────────────────────────
def get_latest_features(ticker: str):
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    df = pd.read_sql(f"""
        SELECT {', '.join([f'AVG({f}) as {f}' for f in FEATURES])}
        FROM (
            SELECT {', '.join(FEATURES)}
            FROM stock_features
            WHERE ticker = '{ticker.upper()}'
            ORDER BY date DESC
            LIMIT 5
        ) recent
    """, conn)
    conn.close()
    return df

# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Financial Risk Intelligence API",
        "endpoints": ["/predict", "/explain", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Get features from DB or request
    df = get_latest_features(request.ticker)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker {request.ticker}"
        )

    X = scaler.transform(df[FEATURES].values)
    risk_prob = model.predict_proba(X)[0][1]
    risk_label = "HIGH RISK" if risk_prob > 0.5 else "LOW RISK"

    return PredictResponse(
        ticker=request.ticker.upper(),
        risk_score=round(float(risk_prob), 4),
        risk_label=risk_label,
        confidence=round(float(max(risk_prob, 1 - risk_prob)), 4)
    )

@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = get_latest_features(request.ticker)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker {request.ticker}"
        )

    X = scaler.transform(df[FEATURES].values)

    # SHAP explanation
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]
        

        feature_impacts = [
            {
               "feature": FEATURES[i],
                "value": round(float(X[0][i]), 4),
                "impact": round(float(sv[i]), 4)
            }
            for i in range(len(FEATURES))
        ]
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)

    except (FileNotFoundError, OSError, PermissionError, Exception) as e:
        feature_impacts = [{"error": str(e)}]

    return ExplainResponse(
        ticker=request.ticker.upper(),
        top_features=feature_impacts[:5]
    )

@app.get("/tickers")
def get_tickers():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    df = pd.read_sql(
        "SELECT DISTINCT ticker FROM stock_features ORDER BY ticker",
        conn
    )
    conn.close()
    return {"tickers": df['ticker'].tolist()}

@app.get("/risk-summary")
def risk_summary():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker,
               AVG(volatility_30d) as avg_volatility,
               AVG(risk_label) as risk_rate,
               COUNT(*) as data_points
        FROM stock_features
        GROUP BY ticker
        ORDER BY avg_volatility DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = [
        {
            "ticker":         row[0],
            "avg_volatility": float(row[1]),
            "risk_rate":      float(row[2]),
            "data_points":    int(row[3])
        }
        for row in rows
    ]
    return {"summary": result}
    df = pd.read_sql("""
        SELECT ticker,
               AVG(volatility_30d) as avg_volatility,
               AVG(risk_label) as risk_rate,
               COUNT(*) as data_points
        FROM stock_features
        GROUP BY ticker
        ORDER BY avg_volatility DESC
    """, conn)
    conn.close()
class AskRequest(BaseModel):
    question: str
    ticker: Optional[str] = None

@app.post("/ask")
def ask(request: AskRequest):
    if rag_index is None:
        raise HTTPException(status_code=500, detail="RAG index not loaded")
    result = query_rag(
        request.question,
        rag_index,
        rag_chunks,
        rag_embedder
    )
    return result
    return {"summary": df.to_dict(orient='records')}
class InvestigateRequest(BaseModel):
    ticker: str
    question: Optional[str] = None

@app.post("/investigate")
def investigate_ticker(request: InvestigateRequest):
    try:
        result = investigate(request.ticker, request.question or "")
        return {
            "ticker":     result['ticker'],
            "risk_score": result['risk_score'],
            "risk_label": result['risk_label'],
            "sentiment":  result['sentiment'],
            "research":   result['research'],
            "report":     result['report'],
            "steps":      result['steps']
        }
    except (FileNotFoundError, OSError, PermissionError, Exception) as e:
        raise HTTPException(status_code=500, detail=str(e))
