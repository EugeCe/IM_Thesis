import pandas as pd
import numpy as np
import requests
import wikipediaapi
import yfinance as yf
import feedparser
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from dateutil import parser as dateparser
from .utils import retry

# Cache the NLTK lexicon
_nltk_ready = False
def ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        _nltk_ready = True

def get_sp500_constituents() -> pd.DataFrame:
    """Return S&P 500 constituents (Ticker, Name, Sector)."""
    # Wikipedia table
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0].rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].astype(str)
    return df[["ticker", "name", "sector"]]

@retry((requests.exceptions.RequestException, ), tries=3, delay=0.3)
def wiki_summary(company_name: str, max_chars: int = 1000) -> str:
    """Fetch a concise Wikipedia summary for a company."""
    wiki = wikipediaapi.Wikipedia("en")
    page = wiki.page(company_name)
    if page.exists():
        txt = page.summary
        return txt[:max_chars]
    return ""

def yf_summary(ticker: str, max_chars: int = 1000) -> str:
    """Fallback to yfinance longBusinessSummary when available."""
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        if isinstance(info, dict):
            s = info.get("longBusinessSummary") or ""
            return (s or "")[:max_chars]
    except Exception:
        pass
    return ""

def company_description(name: str, ticker: str) -> str:
    """Get a company description with Wikipedia first, yfinance as fallback."""
    desc = wiki_summary(name)
    if len(desc) < 120:  # likely empty or too short
        desc = yf_summary(ticker)
    return desc

def get_market_cap(ticker: str):
    try:
        t = yf.Ticker(ticker)
        finfo = t.fast_info
        mc = getattr(finfo, "market_cap", None)
        if mc in (None, 0):
            mc = finfo.get("market_cap", None) if hasattr(finfo, "get") else None
        return float(mc) if mc else np.nan
    except Exception:
        return np.nan

def simple_financial_health(ticker: str):
    """Return a naive financial health score in [0,1] using a few available hints."""
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        # Heuristics with fallbacks
        pe = info.get("trailingPE") or info.get("forwardPE")
        debt_to_equity = info.get("debtToEquity")
        profit_margin = info.get("profitMargins") or info.get("profitMargin")
        # Normalize to 0..1 crudely with caps
        score = 0.5
        # Profit margin: good if >= 10%
        if profit_margin is not None:
            score += np.clip((profit_margin - 0.05) / 0.20, -0.5, 0.5)
        # P/E: penalize if extremely high (>50)
        if pe is not None:
            score += np.clip(0.2 - (max(pe, 0) / 250), -0.2, 0.2)
        # Debt-to-equity: lower is better
        if debt_to_equity is not None:
            score += np.clip(0.2 - (max(debt_to_equity, 0) / 400), -0.2, 0.2)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.5  # neutral

def news_sentiment(company_name: str, max_items: int = 12):
    """Average VADER sentiment compound score from Google News RSS headlines."""
    ensure_nltk()
    sia = SentimentIntensityAnalyzer()
    query = requests.utils.quote(company_name + " stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        items = feed.entries[:max_items]
        if not items:
            return 0.0
        scores = []
        for it in items:
            headline = it.title
            score = sia.polarity_scores(headline)["compound"]
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def zscore(series: pd.Series):
    s = series.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def minmax(series: pd.Series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)
