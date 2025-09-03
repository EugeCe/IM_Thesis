import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed(texts):
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def alignment_scores(objective_text: str, descriptions: list[str]) -> np.ndarray:
    model = get_model()
    v_obj = embed([objective_text])
    v_desc = embed(descriptions)
    sims = cosine_similarity(v_desc, v_obj).ravel()  # [-1..1]
    # Map from [-1,1] to [0,1]
    return (sims + 1.0) / 2.0

def size_compatibility(mcaps: np.ndarray, anchor_cap: float | None) -> np.ndarray:
    """Higher score when market cap is close to anchor (or cohort median)."""
    if anchor_cap is None or not np.isfinite(anchor_cap):
        # Use median of cohort as reference
        anchor_cap = np.nanmedian(mcaps[np.isfinite(mcaps)])
    if not np.isfinite(anchor_cap) or anchor_cap <= 0:
        # fall back to neutral if we can't compute
        return np.ones_like(mcaps) * 0.5
    diffs = np.abs(np.log(mcaps + 1e-9) - np.log(anchor_cap + 1e-9))
    # Smaller diff => higher score
    inv = 1.0 / (1.0 + diffs)  # in (0,1]
    # Normalize 0..1 within cohort
    s = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    return s

def combine_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    w_align = weights.get("alignment", 0.5)
    w_sent = weights.get("sentiment", 0.2)
    w_size = weights.get("size", 0.2)
    w_fin = weights.get("financial", 0.1)
    total = (
        w_align * df["alignment"] +
        w_sent  * df["sentiment_norm"] +
        w_size  * df["size_compat"] +
        w_fin   * df["financial_health"]
    )
    df["score"] = total
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
