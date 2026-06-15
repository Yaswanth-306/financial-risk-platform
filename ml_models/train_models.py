import os
import psycopg2
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# ── Database ──────────────────────────────────────────────
def load_features():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    df = pd.read_sql("SELECT * FROM stock_features", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from stock_features")
    return df

# ── Feature columns ───────────────────────────────────────
FEATURES = [
    'daily_return', 'ma_7', 'ma_14', 'ma_30',
    'volatility_7d', 'volatility_30d',
    'high_low_range', 'volume_ratio', 'price_vs_ma30'
]
TARGET = 'risk_label'

# ── Metrics helper ────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }

# ── Train one model ───────────────────────────────────────
def train_and_log(name, model, X_train, X_test, y_train, y_test, params):
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics, model

# ── Main ──────────────────────────────────────────────────
def main():
    # Load data
    df = load_features()
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Risk label distribution:\n{y.value_counts()}")

    # MLflow experiment
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("financial_risk_models")

    results = {}

    # 1. Logistic Regression
    params = {"C": 1.0, "max_iter": 1000}
    metrics, _ = train_and_log(
        "LogisticRegression",
        LogisticRegression(**params),
        X_train, X_test, y_train, y_test, params
    )
    results["LogisticRegression"] = metrics

    # 2. Random Forest
    params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    metrics, _ = train_and_log(
        "RandomForest",
        RandomForestClassifier(**params),
        X_train, X_test, y_train, y_test, params
    )
    results["RandomForest"] = metrics

    # 3. XGBoost
    params = {"n_estimators": 100, "max_depth": 4,
              "learning_rate": 0.1, "random_state": 42}
    metrics, xgb_model = train_and_log(
        "XGBoost",
        XGBClassifier(**params, eval_metric='logloss'),
        X_train, X_test, y_train, y_test, params
    )
    results["XGBoost"] = metrics

    # 4. LightGBM
    params = {"n_estimators": 100, "max_depth": 4,
              "learning_rate": 0.1, "random_state": 42}
    metrics, _ = train_and_log(
        "LightGBM",
        LGBMClassifier(**params, verbose=-1),
        X_train, X_test, y_train, y_test, params
    )
    results["LightGBM"] = metrics

    # ── Model Comparison ──────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
    print("="*55)
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    for name, m in results.items():
        marker = " ← BEST" if name == best_model_name else ""
        print(f"{name:<22} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['roc_auc']:>6.3f}{marker}")
    print("="*55)

    # ── SHAP Explainability on XGBoost ────────────────────
    print("\nGenerating SHAP values for XGBoost...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:100])
    print("Top feature importances (SHAP):")
    shap_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    print(shap_importance.to_string(index=False))

    # ── Register best model ───────────────────────────────
    print(f"\nRegistering best model: {best_model_name}")
    with mlflow.start_run(run_name=f"{best_model_name}_final"):
        if best_model_name == "XGBoost":
            best_model = XGBClassifier(n_estimators=100, max_depth=4,
                                       learning_rate=0.1, random_state=42,
                                       eval_metric='logloss')
        elif best_model_name == "LightGBM":
            best_model = LGBMClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=42,
                                        verbose=-1)
        elif best_model_name == "RandomForest":
            best_model = RandomForestClassifier(n_estimators=100,
                                                max_depth=6, random_state=42)
        else:
            best_model = LogisticRegression(C=1.0, max_iter=1000)

        best_model.fit(X_train, y_train)
        metrics = evaluate(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name="FinancialRiskModel"
        )
        print(f"Model registered as 'FinancialRiskModel' in MLflow!")

    print("\n✅ Training complete! Open http://localhost:5000 to view results")

if __name__ == "__main__":
    main()
