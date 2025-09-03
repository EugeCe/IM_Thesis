# AI-Enhanced M&A Target Screening (Streamlit)

This is a lightweight, end‑to‑end **Streamlit app** that ranks potential acquisition targets from the S&P 500 based on a user-provided **strategic objective** using:
- **Semantic alignment** (SentenceTransformer embeddings + cosine similarity on company descriptions)
- **Financial size compatibility** (market cap vs. anchor company or cohort median)
- **News sentiment** (Google News RSS + VADER)
- **Simple financial health** (uses yfinance where available)

> ⚠️ This is a **prototype** for academic demonstration (not investment advice). Data sources are public and best-effort.

---

## Quick start (local)

1. **Clone** this repo and enter the folder.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## One‑click deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), select your repo.
3. App file path: `app.py`.
4. Python version: `3.10+`.
5. No secrets are required.
6. First run will warm caches (model + NLTK lexicon).

## How it works

- **Universe**: S&P 500 constituents (scraped from Wikipedia) with tickers and sectors.
- **Descriptions**: Fetched from Wikipedia first, then falls back to `yfinance` summaries if needed.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (small, fast).
- **Sentiment**: Top Google News RSS headlines for each company scored with **VADER**.
- **Market cap**: `yfinance.Ticker.fast_info` where available.
- **Ranking**: Weighted sum of standardized sub-scores:
  - Alignment (0–1)
  - Sentiment (normalized to 0–1 from VADER compound)
  - Size compatibility (closer market cap → higher score)
  - Financial health (basic signals from yfinance if present; otherwise neutral)

## Notes & Limits

- The first run downloads the embedding model (~100MB) and the VADER lexicon.
- Some tickers may have missing fields; the app handles this gracefully and assigns neutral scores.
- To broaden beyond S&P 500, you can upload a CSV of `ticker,name,sector` in the sidebar.

## License

MIT — see `LICENSE`.
