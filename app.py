import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from src.data import (
    get_sp500_constituents, company_description, get_market_cap,
    simple_financial_health, news_sentiment, minmax
)
from src.scoring import alignment_scores, size_compatibility, combine_scores

st.set_page_config(page_title="AI M&A Target Screening", page_icon="📈", layout="wide")

st.title("📈 AI‑Enhanced M&A Target Screening")
st.caption("Prototype app: rank potential acquisition targets based on your strategic objective.")

with st.sidebar:
    st.header("⚙️ Settings")
    universe_choice = st.radio("Company universe", ["S&P 500 (auto)", "Upload CSV"], index=0,
                               help="CSV must have columns: ticker,name,sector")
    max_companies = st.slider("Max companies to screen", 50, 500, 150, step=25)
    st.markdown("---")
    st.subheader("Weights")
    w_align = st.slider("Alignment", 0.0, 1.0, 0.5, 0.05)
    w_sent  = st.slider("News sentiment", 0.0, 1.0, 0.2, 0.05)
    w_size  = st.slider("Size compatibility", 0.0, 1.0, 0.2, 0.05)
    w_fin   = st.slider("Financial health", 0.0, 1.0, 0.1, 0.05)
    st.caption("Weights are normalized automatically.")
    st.markdown("---")
    st.subheader("Anchor size (optional)")
    anchor_ticker = st.text_input("Anchor company ticker (e.g., AAPL)", value="")
    run_btn = st.button("🔍 Run Screening")

st.markdown("### 1) Define your **strategic objective**")
objective_text = st.text_area(
    "Describe what you're seeking (capabilities, technologies, markets, etc.).",
    placeholder="e.g., 'We want ADAS/autonomous driving perception and simulation expertise'",
    height=120,
)

def normalize_weights(a, b, c, d):
    arr = np.array([a, b, c, d], dtype=float)
    s = arr.sum()
    if s <= 0:
        return dict(alignment=0.5, sentiment=0.2, size=0.2, financial=0.1)
    arr = arr / s
    return dict(alignment=arr[0], sentiment=arr[1], size=arr[2], financial=arr[3])

weights = normalize_weights(w_align, w_sent, w_size, w_fin)

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_universe_from_wiki(max_n: int):
    df = get_sp500_constituents()
    if len(df) > max_n:
        df = df.sample(max_n, random_state=42).reset_index(drop=True)
    return df

def load_universe():
    if universe_choice == "S&P 500 (auto)":
        return load_universe_from_wiki(max_companies)
    else:
        f = st.file_uploader("Upload CSV with columns: ticker,name,sector", type=["csv"])
        if f is None:
            st.info("Upload a CSV to continue, or switch to S&P 500.")
            st.stop()
        df = pd.read_csv(f)
        needed = {"ticker", "name", "sector"}
        if not needed.issubset(set(df.columns.str.lower())):
            st.error("CSV must include columns: ticker, name, sector.")
            st.stop()
        # normalize column names
        df.columns = [c.lower() for c in df.columns]
        df = df[list(needed)]
        if len(df) > max_companies:
            df = df.sample(max_companies, random_state=42).reset_index(drop=True)
        return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_company_blob(name: str, ticker: str):
    desc = company_description(name, ticker)
    mcap = get_market_cap(ticker)
    fin  = simple_financial_health(ticker)
    sent = news_sentiment(name)
    return dict(description=desc, market_cap=mcap, financial_health=fin, sentiment=sent)

def explain_row(row, obj_text: str) -> str:
    bits = []
    # Alignment
    if row["alignment"] >= 0.7:
        bits.append("strong semantic alignment")
    elif row["alignment"] >= 0.5:
        bits.append("moderate alignment")
    else:
        bits.append("low alignment")
    # Sentiment
    if row["sentiment_norm"] >= 0.6:
        bits.append("positive news sentiment")
    elif row["sentiment_norm"] <= 0.4:
        bits.append("caution: weaker sentiment")
    # Size
    if row["size_compat"] >= 0.6:
        bits.append("size fits anchor")
    # Financial
    if row["financial_health"] >= 0.6:
        bits.append("solid basic financials")
    elif row["financial_health"] <= 0.4:
        bits.append("potential financial risk")
    return ", ".join(bits)

if run_btn:
    if not objective_text.strip():
        st.warning("Please enter a strategic objective first.")
        st.stop()

    st.markdown("### 2) Building the universe & fetching signals")
    with st.spinner("Collecting data (descriptions, market cap, sentiment, basic financials)…"):
        uni = load_universe()
        blobs = []
        for _, r in uni.iterrows():
            blob = fetch_company_blob(r["name"], r["ticker"])
            blobs.append(blob)
        uni = uni.assign(
            description=[b["description"] for b in blobs],
            market_cap=[b["market_cap"] for b in blobs],
            financial_health=[b["financial_health"] for b in blobs],
            sentiment_raw=[b["sentiment"] for b in blobs],
        )

    st.markdown("### 3) Scoring & ranking")
    with st.spinner("Computing embeddings and scores… (first run downloads model)"):
        # Alignment
        descs = uni["description"].fillna("").replace({np.nan: ""}).tolist()
        align = alignment_scores(objective_text, descs)
        uni["alignment"] = align

        # Sentiment (VADER compound is -1..1 → normalize 0..1)
        uni["sentiment_norm"] = (uni["sentiment_raw"] + 1.0) / 2.0

        # Size compatibility
        anchor_cap: Optional[float] = None
        if anchor_ticker.strip():
            anchor_cap = float(get_market_cap(anchor_ticker.strip().upper()) or np.nan)
        size = size_compatibility(uni["market_cap"].astype(float).to_numpy(), anchor_cap)
        uni["size_compat"] = size

        # Combine
        ranked = combine_scores(uni.copy(), weights)

        # Explanations
        ranked["explanation"] = ranked.apply(lambda r: explain_row(r, objective_text), axis=1)

        show_cols = [
            "ticker", "name", "sector", "score",
            "alignment", "sentiment_norm", "size_compat", "financial_health",
            "market_cap", "explanation"
        ]
        st.success("Done!")
        st.dataframe(
            ranked[show_cols].style.format({
                "score": "{:.3f}",
                "alignment": "{:.3f}",
                "sentiment_norm": "{:.3f}",
                "size_compat": "{:.3f}",
                "financial_health": "{:.3f}",
                "market_cap": "{:,.0f}"
            }),
            use_container_width=True,
            height=540,
        )

        csv = ranked[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results (CSV)", data=csv, file_name="screening_results.csv", mime="text/csv")

        with st.expander("ℹ️ Notes on scoring"):
            st.write(f"""
**Weights:** alignment={weights['alignment']:.2f}, sentiment={weights['sentiment']:.2f},
size={weights['size']:.2f}, financial={weights['financial']:.2f}.

- **Alignment**: semantic similarity between your objective and company description.
- **Sentiment**: VADER on recent Google News headlines.
- **Size**: market cap closeness to anchor (or median).
- **Financial**: heuristic signals from yfinance info; neutral (0.5) if missing.
""")

else:
    st.info("Enter a strategic objective, adjust settings, and click **Run Screening**.")

st.markdown("---")
st.caption("Built for demonstration purposes only. Not investment advice.")
