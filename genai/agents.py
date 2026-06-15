import os
import json
import requests
from typing import TypedDict, Annotated, List
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
import operator

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
API_URL      = "http://localhost:8000"
MODEL        = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

def llm(prompt: str, system: str = "You are a financial risk analyst.") -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=800,
        temperature=0.1
    )
    return r.choices[0].message.content

# ── Agent State ───────────────────────────────────────────
class AgentState(TypedDict):
    ticker:        str
    question:      str
    research:      str
    sentiment:     str
    risk_score:    float
    risk_label:    str
    shap_features: list
    report:        str
    steps:         Annotated[List[str], operator.add]

# ── Agent 1: Supervisor ───────────────────────────────────
def supervisor_agent(state: AgentState) -> AgentState:
    print(f"\n🎯 Supervisor: Starting investigation for {state['ticker']}")
    step = f"Supervisor: Routing investigation for {state['ticker']}"
    return {**state, "steps": [step]}

# ── Agent 2: Research Agent ───────────────────────────────
def research_agent(state: AgentState) -> AgentState:
    ticker = state['ticker']
    print(f"🔍 Research Agent: Fetching data for {ticker}...")

    # Call RAG /ask endpoint
    try:
        r = requests.post(f"{API_URL}/ask", json={
            "question": f"What are the main risks and business outlook for {ticker}?",
            "ticker": ticker
        }, timeout=30)
        rag_result = r.json()
        rag_answer = rag_result.get('answer', 'No data found')
    except Exception as e:
        rag_answer = f"Could not fetch SEC filing data: {e}"

    # Summarize with LLM
    research = llm(
        f"Summarize the key findings about {ticker} from this SEC filing analysis:\n{rag_answer}",
        system="You are a financial research analyst. Be concise and factual."
    )

    step = f"Research Agent: Retrieved SEC filing data and news for {ticker}"
    print(f"✅ Research complete")
    return {**state, "research": research, "steps": [step]}

# ── Agent 3: Sentiment Agent ──────────────────────────────
def sentiment_agent(state: AgentState) -> AgentState:
    ticker   = state['ticker']
    research = state['research']
    print(f"💭 Sentiment Agent: Analyzing sentiment for {ticker}...")

    sentiment = llm(
        f"""Analyze the sentiment of this financial research about {ticker}.
        
Research: {research}

Provide:
1. Overall sentiment: Bullish / Bearish / Neutral
2. Confidence: High / Medium / Low  
3. Key positive factors (2-3 points)
4. Key negative factors (2-3 points)
5. Sentiment score: -1.0 (very bearish) to +1.0 (very bullish)

Be concise and structured.""",
        system="You are a financial sentiment analyst."
    )

    step = f"Sentiment Agent: Analyzed market sentiment for {ticker}"
    print(f"✅ Sentiment analysis complete")
    return {**state, "sentiment": sentiment, "steps": [step]}

# ── Agent 4: Risk Agent ───────────────────────────────────
def risk_agent(state: AgentState) -> AgentState:
    ticker = state['ticker']
    print(f"⚠️  Risk Agent: Getting ML risk score for {ticker}...")

    risk_score    = 0.5
    risk_label    = "UNKNOWN"
    shap_features = []

    try:
        # Get prediction
        r1 = requests.post(f"{API_URL}/predict",
                           json={"ticker": ticker}, timeout=10)
        pred       = r1.json()
        risk_score = pred.get('risk_score', 0.5)
        risk_label = pred.get('risk_label', 'UNKNOWN')

        # Get SHAP explanation
        r2 = requests.post(f"{API_URL}/explain",
                           json={"ticker": ticker}, timeout=10)
        expl          = r2.json()
        shap_features = expl.get('top_features', [])

    except Exception as e:
        print(f"API error: {e}")

    step = (f"Risk Agent: ML model predicts {risk_label} "
            f"(score: {risk_score:.4f}) for {ticker}")
    print(f"✅ Risk assessment complete: {risk_label} ({risk_score:.4f})")
    return {
        **state,
        "risk_score":    risk_score,
        "risk_label":    risk_label,
        "shap_features": shap_features,
        "steps":         [step]
    }

# ── Agent 5: Report Agent ─────────────────────────────────
def report_agent(state: AgentState) -> AgentState:
    ticker   = state['ticker']
    print(f"📝 Report Agent: Generating executive report for {ticker}...")

    shap_str = "\n".join([
        f"  - {f['feature']}: {f['impact']:+.4f}"
        for f in state['shap_features'][:5]
        if 'feature' in f
    ])

    report = llm(
        f"""Generate a professional executive risk report for {ticker}.

Use this data:

RESEARCH FINDINGS:
{state['research']}

SENTIMENT ANALYSIS:
{state['sentiment']}

ML RISK ASSESSMENT:
- Risk Score: {state['risk_score']:.4f} (0=low, 1=high)
- Risk Label: {state['risk_label']}
- Top SHAP Features:
{shap_str}

Format the report with these sections:
## Executive Summary
## Key Risk Factors  
## Sentiment Analysis
## ML Model Assessment
## Top Contributing Factors (SHAP)
## Recommendation

Be professional, concise, and actionable.""",
        system="You are a senior financial risk officer writing executive reports."
    )

    step = f"Report Agent: Generated executive risk report for {ticker}"
    print(f"✅ Report generated")
    return {**state, "report": report, "steps": [step]}

# ── Build LangGraph ───────────────────────────────────────
def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("research",   research_agent)
    graph.add_node("sentiment",  sentiment_agent)
    graph.add_node("risk",       risk_agent)
    graph.add_node("report",     report_agent)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "research")
    graph.add_edge("research",   "sentiment")
    graph.add_edge("sentiment",  "risk")
    graph.add_edge("risk",       "report")
    graph.add_edge("report",     END)

    return graph.compile()

# ── Run Investigation ─────────────────────────────────────
def investigate(ticker: str, question: str = "") -> dict:
    app = build_agent_graph()

    initial_state = AgentState(
        ticker        = ticker.upper(),
        question      = question or f"Investigate risk for {ticker}",
        research      = "",
        sentiment     = "",
        risk_score    = 0.0,
        risk_label    = "",
        shap_features = [],
        report        = "",
        steps         = []
    )

    print(f"\n{'='*60}")
    print(f"🤖 LANGGRAPH MULTI-AGENT INVESTIGATION: {ticker}")
    print(f"{'='*60}")

    result = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("📊 FINAL REPORT")
    print(f"{'='*60}")
    print(result['report'])
    print(f"\n📋 Agent Steps:")
    for i, step in enumerate(result['steps'], 1):
        print(f"  {i}. {step}")

    return result

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = investigate("TSLA")
