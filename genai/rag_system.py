import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import faiss
import numpy as np
import json
import re

load_dotenv('./.env')

_embedder_cache = None
def _get_embedder():
    global _embedder_cache
    if _embedder_cache is None:
        from sentence_transformers import SentenceTransformer
        _embedder_cache = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder_cache

# ── Config ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DATA_DIR     = Path('./data/sec_filings')
INDEX_PATH   = Path('./data/faiss_index')
CHUNKS_PATH  = Path('./data/chunks.json')


# ── SEC EDGAR Fetcher ─────────────────────────────────────
HEADERS = {'User-Agent': 'FinRiskAI chitr@example.com'}

def fetch_sec_filing(ticker: str):
    print(f"Fetching SEC filing for {ticker}...")
    try:
        # Get company CIK
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K"
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"

        # Use EDGAR full-text search
        r = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K",
            headers=HEADERS, timeout=10
        )

        # Simpler approach: use company search
        ticker_url = f"https://data.sec.gov/submissions/CIK{get_cik(ticker)}.json"
        return get_cik(ticker)
    except Exception as e:
        print(f"Error: {e}")
        return None

CIK_MAP = {
    'AAPL': '0000320193', 'MSFT': '0000789019',
    'GOOGL': '0001652044', 'TSLA': '0001318605',
    'AMZN': '0001018724', 'META': '0001326801',
    'NVDA': '0001045810', 'JPM':  '0000019617',
    'BAC':  '0000070858', 'GS':   '0000886982'
}

def get_filing_text(ticker: str) -> str:
    """
    Get 10-K filing text for a ticker.
    Returns cached file if it exists, otherwise fetches from SEC EDGAR and saves.
    """
    file_path = DATA_DIR / f"{ticker}_10K.txt"

    # Return cached if exists
    if file_path.exists():
        print(f"Loading cached filing for {ticker}")
        return file_path.read_text()

    print(f"Fetching 10-K for {ticker} from SEC EDGAR...")
    cik = CIK_MAP.get(ticker.upper())
    if not cik:
        return f"No CIK found for {ticker}"

    try:
        # Get submissions
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(sub_url, headers=HEADERS, timeout=15)
        data = r.json()

        # Find latest 10-K
        filings = data['filings']['recent']
        forms   = filings['form']
        accessions = filings['accessionNumber']

        tenk_idx = next(
            (i for i, f in enumerate(forms) if f == '10-K'), None
        )
        if tenk_idx is None:
            return f"No 10-K found for {ticker}"

        accession = accessions[tenk_idx].replace('-', '')
        cik_plain = cik.lstrip('0')

        # Get filing index
        idx_url = (f"https://www.sec.gov/Archives/edgar/data/"
                   f"{cik_plain}/{accession}/{accessions[tenk_idx]}-index.htm")
        r2 = requests.get(idx_url, headers=HEADERS, timeout=15)

        # Fetch actual filing text
        r3 = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=HEADERS, timeout=15
        )

        text = r3.text if r3.status_code == 200 else ""
        file_path.write_text(text)
        print(f"✅ Saved {ticker} filing ({len(text)} chars)")
        return text

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return ""

# ── Text Chunking ─────────────────────────────────────────
def chunk_text(text: str, chunk_size=400, overlap=50):
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── Embedding + FAISS ─────────────────────────────────────
def build_index(tickers):
    print("Loading embedding model...")
    embedder = _get_embedder()  # 'all-MiniLM-L6-v2')

    all_chunks = []
    all_meta   = []

    for ticker in tickers:
        text   = get_filing_text(ticker)
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_meta.append({'ticker': ticker, 'text': chunk})

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, 'w') as f:
        json.dump(all_meta, f)

    print(f"✅ FAISS index built: {index.ntotal} vectors")
    return index, all_meta, embedder

def load_index():
    index    = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    embedder = _get_embedder()  # 'all-MiniLM-L6-v2')
    return index, chunks, embedder

# ── RAG Query ─────────────────────────────────────────────
def query_rag(question: str, index, chunks, embedder, top_k=5):
    # Embed question
    q_emb = embedder.encode([question]).astype('float32')

    # Search FAISS
    distances, indices = index.search(q_emb, top_k)

    # Get relevant chunks
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    context  = "\n\n---\n\n".join([c['text'] for c in relevant])
    tickers  = list(set([c['ticker'] for c in relevant]))

    # Call Groq LLM
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""You are a financial risk analyst. 
Answer the question based on the SEC filing excerpts below.
Be specific and cite relevant risk factors.

SEC FILING EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1
    )

    answer = response.choices[0].message.content

    return {
        "answer":   answer,
        "sources":  tickers,
        "chunks":   [c['text'][:200] + "..." for c in relevant[:3]]
    }

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    TICKERS = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN',
               'META', 'NVDA', 'JPM', 'BAC', 'GS']

    # Build index
    if INDEX_PATH.exists():
        print("Loading existing FAISS index...")
        index, chunks, embedder = load_index()
    else:
        print("Building FAISS index from SEC filings...")
        index, chunks, embedder = build_index(TICKERS)

    # Test queries
    questions = [
        "What are the main risk factors for Tesla?",
        "What liquidity risks do the banks face?",
        "How does Apple manage market risk?",
    ]

    print("\n" + "="*60)
    print("RAG SYSTEM TEST")
    print("="*60)

    for q in questions:
        print(f"\n❓ {q}")
        result = query_rag(q, index, chunks, embedder)
        print(f"💬 {result['answer'][:300]}...")
        print(f"📚 Sources: {result['sources']}")
