import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #020817;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;

    border-right: 1px solid #1e293b !important;
}

[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

/* Hide default header */
header { visibility: hidden; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px !important;
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    gap: 4px !important;
}

[data-testid="stRadio"] label {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

/* Plotly charts background */
.js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Custom cards */
.risk-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 8px;
}

.hero-sub {
    font-size: 16px;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 32px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}

.tag-high { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.tag-low  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.tag-api  { background: #0c1a2e; color: #38bdf8; border: 1px solid #1e3a5f; }

.stat-pill {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 99px;
    padding: 6px 16px;
    font-size: 13px;
    color: #94a3b8;
    display: inline-block;
    margin: 4px;
}

.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 24px 0;
}

.info-box {
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 16px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.7;
}

.arch-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #64748b;
    line-height: 2;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Chart theme ───────────────────────────────────────────
CHART_THEME = {
    "paper_bgcolor": "#0f172a",
    "plot_bgcolor":  "#0f172a",
    "font_color":    "#94a3b8",
    "gridcolor":     "#1e293b",
}

def styled_chart(fig, height=400):
    fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font_color=CHART_THEME["font_color"],
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=CHART_THEME["gridcolor"],
                   showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=CHART_THEME["gridcolor"],
                   showgrid=True, zeroline=False),
    )
    return fig

# ── API helpers ───────────────────────────────────────────
@st.cache_data(ttl=60)
def get_tickers():
    try:
        r = requests.get(f"{API_URL}/tickers", timeout=5)
        return r.json()["tickers"]
    except:
        return ["AAPL","TSLA","MSFT","GOOGL","AMZN","META","NVDA","JPM","BAC","GS"]

def get_prediction(ticker):
    try:
        r = requests.post(f"{API_URL}/predict",
                          json={"ticker": ticker}, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_explanation(ticker):
    try:
        r = requests.post(f"{API_URL}/explain",
                          json={"ticker": ticker}, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60)
def get_risk_summary():
    try:
        r = requests.get(f"{API_URL}/risk-summary", timeout=5)
        return r.json()["summary"]
    except:
        return []

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:22px; font-weight:800; color:#f1f5f9;'>⚡ FinRisk AI</div>
        <div style='font-size:11px; color:#475569; letter-spacing:2px; 
                    text-transform:uppercase; margin-top:4px;'>
            Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "🎯  Risk Predictor",
        "🔍  Explainability",
        "📋  Risk Summary",
        "💬  Document Q&A",
        "🤖  Agent Workflow"
    ])

    st.markdown("<hr style='border-color:#1e293b; margin:24px 0'>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px; color:#334155; text-transform:uppercase; 
                letter-spacing:1.5px; margin-bottom:12px;'>System Status</div>
    """, unsafe_allow_html=True)

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        status = r.json()
        st.markdown('<span class="tag tag-api">● API ONLINE</span>',
                    unsafe_allow_html=True)
    except:
        st.markdown(
            '<span class="tag tag-high">● API OFFLINE</span>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#334155; line-height:1.8;'>
        <div>🗄 PostgreSQL</div>
        <div>🔄 Airflow 2.9</div>
        <div>⚗️ MLflow</div>
        <div>⚡ FastAPI</div>
        <div>🤖 XGBoost + SHAP</div>
    </div>
    """, unsafe_allow_html=True)

# ── Page 1: Overview ──────────────────────────────────────
if "Overview" in page:
    st.markdown('<div class="hero-title">Financial Risk Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">AI-powered stock risk analysis · '
        'Real-time predictions · Explainable ML</div>',
        unsafe_allow_html=True)

    tickers = get_tickers()
    summary = get_risk_summary()
    high_risk = sum(1 for s in summary if s.get('risk_rate', 0) > 0.25)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📈 Stocks Tracked", len(tickers))
    with c2: st.metric("🤖 Models Trained", "5")
    with c3: st.metric("🔴 High Risk", high_risk)
    with c4: st.metric("🟢 Low Risk", len(tickers) - high_risk)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if summary:
        df = pd.DataFrame(summary)
        df['avg_volatility'] = df['avg_volatility'].round(4)
        df['risk_rate'] = df['risk_rate'].round(3)
        df['risk_level'] = df['risk_rate'].apply(
            lambda x: "HIGH" if x > 0.25 else "LOW")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="section-title">📊 Volatility by Stock</div>',
                        unsafe_allow_html=True)
            colors = ["#ef4444" if r > 0.3 else "#38bdf8"
                      for r in df['risk_rate']]
            fig = go.Figure(go.Bar(
                x=df['ticker'],
                y=df['avg_volatility'],
                marker_color=colors,
                text=df['avg_volatility'].round(4),
                textposition='outside',
                textfont=dict(color='#94a3b8', size=11)
            ))
            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Avg 30D Volatility"
            )
            st.plotly_chart(styled_chart(fig, 350),
                            use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">🥧 Risk Distribution</div>',
                        unsafe_allow_html=True)
            risk_counts = df['risk_level'].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.6,
                marker=dict(colors=[
                     "#ef4444" if label == "HIGH" else "#38bdf8"
                     for label in risk_counts.index
                ]),
                textfont=dict(color='#94a3b8')
            ))
            fig2.update_layout(
                showlegend=True,
                legend=dict(font=dict(color='#94a3b8')),
                annotations=[dict(
                    text=f"<b>{len(df)}</b><br>Stocks",
                    x=0.5, y=0.5,
                    font=dict(size=16, color='#f1f5f9'),
                    showarrow=False
                )]
            )
            st.plotly_chart(
                styled_chart(fig2, 350).update_layout(
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                ),
                use_container_width=True
            )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏗 System Architecture</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-box">
        <span style="color:#38bdf8">yfinance</span> + 
        <span style="color:#38bdf8">SEC EDGAR</span> + 
        <span style="color:#38bdf8">NewsAPI</span><br>
        ↓<br>
        <span style="color:#818cf8">Apache Airflow DAGs</span><br>
        ↓<br>
        <span style="color:#818cf8">PySpark Feature Engineering</span><br>
        ↓<br>
        <span style="color:#34d399">PostgreSQL Feature Store</span><br>
        ↓<br>
        <span style="color:#fb923c">XGBoost · LightGBM · RandomForest · LogReg · TF</span><br>
        ↓<br>
        <span style="color:#f472b6">MLflow Tracking + Model Registry</span><br>
        ↓<br>
        <span style="color:#fbbf24">FastAPI  /predict  /explain  /risk-summary</span><br>
        ↓<br>
        <span style="color:#38bdf8">Streamlit Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

# ── Page 2: Risk Predictor ────────────────────────────────
elif "Predictor" in page:
    st.markdown('<div class="hero-title">Risk Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Get ML-powered risk scores for any tracked stock</div>',
        unsafe_allow_html=True)

    tickers = get_tickers()
    col1, col2 = st.columns([1, 3])

    with col1:
        ticker = st.selectbox("Select Stock", tickers)
        predict_btn = st.button("🔮 Predict Risk", type="primary")

    if predict_btn:
        with st.spinner(f"Analyzing {ticker}..."):
            result = get_prediction(ticker)

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            risk_score = result['risk_score']
            risk_label = result['risk_label']
            confidence = result['confidence']

            is_high = risk_score > 0.5
            label_color = "#ef4444" if is_high else "#4ade80"
            label_bg    = "#450a0a" if is_high else "#052e16"

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Risk Score", f"{risk_score:.4f}")
            with c2:
                st.metric("Confidence", f"{confidence:.2%}")
            with c3:
                st.metric("Label", risk_label)

            # Gauge
            gauge_color = "#ef4444" if is_high else "#38bdf8"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                number={'suffix': '%',
                        'font': {'size': 48, 'color': '#f1f5f9'}},
                title={'text': f"<b>{ticker}</b> Risk Score",
                       'font': {'size': 18, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': '#334155',
                             'tickfont': {'color': '#64748b'}},
                    'bar': {'color': gauge_color, 'thickness': 0.25},
                    'bgcolor': '#0f172a',
                    'bordercolor': '#1e293b',
                    'steps': [
                        {'range': [0, 33],  'color': '#052e16'},
                        {'range': [33, 66], 'color': '#1c1917'},
                        {'range': [66, 100],'color': '#450a0a'}
                    ],
                    'threshold': {
                        'line': {'color': '#f59e0b', 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#94a3b8",
                height=380,
                margin=dict(l=40, r=40, t=60, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            if is_high:
                st.markdown(f"""
                <div style='background:#450a0a; border:1px solid #7f1d1d; 
                            border-left:4px solid #ef4444; border-radius:10px; 
                            padding:16px; color:#fca5a5; font-size:14px;'>
                    ⚠️ <b>{ticker}</b> is classified as <b>HIGH RISK</b>. 
                    Elevated volatility patterns detected.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#052e16; border:1px solid #166534; 
                            border-left:4px solid #4ade80; border-radius:10px; 
                            padding:16px; color:#86efac; font-size:14px;'>
                    ✅ <b>{ticker}</b> is classified as <b>LOW RISK</b>. 
                    Stable volatility patterns detected.
                </div>""", unsafe_allow_html=True)

# ── Page 3: Explainability ────────────────────────────────
elif "Explainability" in page:
    st.markdown('<div class="hero-title">Explainability</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Understand why the model made its prediction using SHAP values</div>',
        unsafe_allow_html=True)

    tickers = get_tickers()
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.selectbox("Select Stock", tickers)
        explain_btn = st.button("🧠 Explain", type="primary")

    if explain_btn:
        with st.spinner("Generating SHAP explanation..."):
            result = get_explanation(ticker)
            pred   = get_prediction(ticker)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            features = result.get('top_features', [])

            if features and 'feature' in features[0]:
                df = pd.DataFrame(features)

                risk_score = pred.get('risk_score', 0)
                risk_label = pred.get('risk_label', '')
                c1, c2 = st.columns(2)
                with c1: st.metric("Risk Score", f"{risk_score:.4f}")
                with c2: st.metric("Risk Label", risk_label)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">📊 SHAP Feature Impact</div>',
                    unsafe_allow_html=True)

                colors = ["#ef4444" if v > 0 else "#38bdf8"
                          for v in df['impact']]
                fig = go.Figure(go.Bar(
                    x=df['impact'],
                    y=df['feature'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in df['impact']],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=12)
                ))
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(styled_chart(fig, 380),
                                use_container_width=True)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        '<div class="section-title">📋 Feature Details</div>',
                        unsafe_allow_html=True)
                    df['direction'] = df['impact'].apply(
                        lambda x: "⬆️ Increases Risk"
                        if x > 0 else "⬇️ Decreases Risk"
                    )
                    st.dataframe(df[['feature','value','impact','direction']],
                                 use_container_width=True)

                with col2:
                    st.markdown(
                        '<div class="section-title">ℹ️ How to Read</div>',
                        unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-box">
                        🔴 <b>Red bars</b> → feature pushes risk score higher<br>
                        🔵 <b>Blue bars</b> → feature pushes risk score lower<br>
                        📏 <b>Bar length</b> → strength of influence<br><br>
                        SHAP values show the <b>marginal contribution</b> 
                        of each feature to the final prediction, 
                        starting from the model's base rate.
                    </div>
                    """, unsafe_allow_html=True)

# ── Page 4: Risk Summary ──────────────────────────────────
elif "Summary" in page:
    st.markdown('<div class="hero-title">Risk Summary</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Portfolio-level risk overview across all tracked stocks</div>',
        unsafe_allow_html=True)

    summary = get_risk_summary()

    if summary:
        df = pd.DataFrame(summary)
        df['avg_volatility'] = df['avg_volatility'].round(4)
        df['risk_rate'] = df['risk_rate'].round(3)
        df['risk_level'] = df['risk_rate'].apply(
            lambda x: "HIGH" if x > 0.25 else "LOW")

        st.markdown(
            '<div class="section-title">🎯 Risk vs Volatility</div>',
            unsafe_allow_html=True)
        fig = px.scatter(
            df,
            x='avg_volatility', y='risk_rate',
            text='ticker', size='data_points',
            color='risk_rate',
            color_continuous_scale=[[0,'#38bdf8'],[0.5,'#f59e0b'],[1,'#ef4444']],
            labels={'avg_volatility': 'Avg Volatility',
                    'risk_rate': 'Risk Rate'}
        )
        fig.update_traces(
            textposition='top center',
            textfont=dict(color='#f1f5f9', size=12),
            marker=dict(line=dict(color='#0f172a', width=1))
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="section-title">🔴 Highest Risk</div>',
                unsafe_allow_html=True)
            top_risk = df.nlargest(5, 'risk_rate')[
                ['ticker','risk_rate','avg_volatility']]
            st.dataframe(top_risk, use_container_width=True)

        with col2:
            st.markdown(
                '<div class="section-title">🟢 Lowest Risk</div>',
                unsafe_allow_html=True)
            low_risk = df.nsmallest(5, 'risk_rate')[
                ['ticker','risk_rate','avg_volatility']]
            st.dataframe(low_risk, use_container_width=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📊 Full Portfolio Table</div>',
            unsafe_allow_html=True)
        st.dataframe(
            df[['ticker','risk_level','risk_rate',
                'avg_volatility','data_points']],
            use_container_width=True
        )
    else:
        st.warning("No data available. Make sure FastAPI is running on port 8000.")
#---- page 5 Document Q&A -------------------

elif "Q&A" in page:
    st.markdown('<div class="hero-title">Document Q&A</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Ask questions about SEC filings · Powered by RAG + Groq LLaMA 3</div>',
        unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Example questions
    st.markdown("**💡 Example questions:**")
    examples = [
        "What are Tesla's main risk factors?",
        "How do banks manage liquidity risk?",
        "What cybersecurity risks does Apple face?",
        "How does NVIDIA manage competition risk?",
        "What are Goldman Sachs credit risks?"
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.question = ex

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    question = st.text_input(
        "Ask a question about SEC filings:",
        value=st.session_state.get('question', ''),
        placeholder="e.g. What liquidity risks do banks face?"
    )

    ask_btn = st.button("🔍 Search Filings", type="primary")

    if ask_btn and question:
        with st.spinner("Searching SEC filings and generating answer..."):
            try:
                r = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=30
                )
                result = r.json()

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # Answer
                st.markdown(
                    '<div class="section-title">💬 Answer</div>',
                    unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:#0f172a; border:1px solid #1e293b;
                            border-left:4px solid #818cf8; border-radius:10px;
                            padding:20px; color:#e2e8f0; line-height:1.8;
                            font-size:14px;'>
                    {result.get('answer', 'No answer generated')}
                </div>
                """, unsafe_allow_html=True)

                # Sources
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">📚 Sources</div>',
                    unsafe_allow_html=True)
                sources = result.get('sources', [])
                source_html = " ".join([
                    f'<span class="tag tag-api">{s}</span>'
                    for s in sources
                ])
                st.markdown(source_html, unsafe_allow_html=True)

                # Retrieved chunks
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">📄 Retrieved Passages</div>',
                    unsafe_allow_html=True)
                for i, chunk in enumerate(result.get('chunks', []), 1):
                    with st.expander(f"Passage {i}"):
                        st.markdown(f"""
                        <div style='font-family: JetBrains Mono, monospace;
                                    font-size:12px; color:#94a3b8;
                                    line-height:1.7;'>
                            {chunk}
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

    elif ask_btn and not question:
        st.warning("Please enter a question first.")
#--------- agents -------------------

elif "Agent" in page:
    st.markdown('<div class="hero-title">Agent Workflow</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">LangGraph multi-agent investigation · '
        'Supervisor → Research → Sentiment → Risk → Report</div>',
        unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Agent pipeline visualization
    st.markdown('<div class="section-title">🔄 Agent Pipeline</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; align-items:center; gap:8px; 
                flex-wrap:wrap; margin-bottom:24px;'>
        <div style='background:#1e293b; border:1px solid #334155; 
                    border-radius:8px; padding:10px 16px; 
                    font-size:13px; color:#38bdf8; font-weight:600;'>
            🎯 Supervisor
        </div>
        <div style='color:#334155; font-size:20px;'>→</div>
        <div style='background:#1e293b; border:1px solid #334155; 
                    border-radius:8px; padding:10px 16px; 
                    font-size:13px; color:#818cf8; font-weight:600;'>
            🔍 Research
        </div>
        <div style='color:#334155; font-size:20px;'>→</div>
        <div style='background:#1e293b; border:1px solid #334155; 
                    border-radius:8px; padding:10px 16px; 
                    font-size:13px; color:#f472b6; font-weight:600;'>
            💭 Sentiment
        </div>
        <div style='color:#334155; font-size:20px;'>→</div>
        <div style='background:#1e293b; border:1px solid #334155; 
                    border-radius:8px; padding:10px 16px; 
                    font-size:13px; color:#fb923c; font-weight:600;'>
            ⚠️ Risk
        </div>
        <div style='color:#334155; font-size:20px;'>→</div>
        <div style='background:#1e293b; border:1px solid #334155; 
                    border-radius:8px; padding:10px 16px; 
                    font-size:13px; color:#4ade80; font-weight:600;'>
            📝 Report
        </div>
    </div>
    """, unsafe_allow_html=True)

    tickers = get_tickers()
    col1, col2 = st.columns([1, 3])

    with col1:
        ticker = st.selectbox("Select Stock", tickers, key="agent_ticker")
        investigate_btn = st.button("🚀 Investigate", type="primary")

    if investigate_btn:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Show live agent steps
        st.markdown('<div class="section-title">⚡ Agent Execution</div>',
                    unsafe_allow_html=True)

        step_placeholders = {}
        agent_names = [
            ("🎯 Supervisor",  "#38bdf8"),
            ("🔍 Research",    "#818cf8"),
            ("💭 Sentiment",   "#f472b6"),
            ("⚠️ Risk",        "#fb923c"),
            ("📝 Report",      "#4ade80")
        ]

        for name, color in agent_names:
            step_placeholders[name] = st.markdown(f"""
            <div style='background:#0f172a; border:1px solid #1e293b;
                        border-radius:8px; padding:12px 16px; margin:4px 0;
                        font-size:13px; color:#334155;'>
                {name} — waiting...
            </div>
            """, unsafe_allow_html=True)

        with st.spinner(f"Running multi-agent investigation for {ticker}... (30-60 seconds)"):
            try:
                r = requests.post(
                    f"{API_URL}/investigate",
                    json={"ticker": ticker},
                    timeout=120
                )
                result = r.json()

                # Show completed steps
                steps = result.get('steps', [])
                st.markdown('<div class="section-title">✅ Completed Steps</div>',
                            unsafe_allow_html=True)
                for i, step in enumerate(steps):
                    colors = ["#38bdf8","#818cf8","#f472b6","#fb923c","#4ade80"]
                    color  = colors[i % len(colors)]
                    st.markdown(f"""
                    <div style='background:#0f172a; border:1px solid #1e293b;
                                border-left:3px solid {color}; border-radius:8px;
                                padding:12px 16px; margin:4px 0;
                                font-size:13px; color:#94a3b8;'>
                        <b style='color:{color};'>Step {i+1}:</b> {step}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # Risk badge
                risk_score = result.get('risk_score', 0)
                risk_label = result.get('risk_label', '')
                is_high    = risk_score > 0.5
                badge_color = "#ef4444" if is_high else "#4ade80"
                badge_bg    = "#450a0a" if is_high else "#052e16"

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("ML Risk Score", f"{risk_score:.4f}")
                with c2:
                    st.metric("Risk Label", risk_label)

                # Executive Report
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">📊 Executive Report</div>',
                    unsafe_allow_html=True)
                report = result.get('report', 'No report generated')
                st.markdown(f"""
                <div style='background:#0f172a; border:1px solid #1e293b;
                            border-radius:12px; padding:24px;
                            color:#e2e8f0; line-height:1.8; font-size:14px;'>
                    {report.replace(chr(10), '<br>').replace('**', '<b>').replace('## ', '<h3 style="color:#818cf8; margin-top:16px;">')}
                </div>
                """, unsafe_allow_html=True)

                # Sentiment
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">💭 Sentiment Analysis</div>',
                    unsafe_allow_html=True)
                sentiment = result.get('sentiment', '')
                st.markdown(f"""
                <div style='background:#0f172a; border:1px solid #1e293b;
                            border-left:4px solid #f472b6; border-radius:10px;
                            padding:16px; color:#94a3b8; font-size:13px;
                            line-height:1.8;'>
                    {sentiment.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
