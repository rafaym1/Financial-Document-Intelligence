from fdi.embeddings import client
from fdi.rate_limit import voyage_limiter

RERANK_MODEL = "rerank-2.5"


def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Re-score candidate chunks against the query with a cross-encoder reranker, returning the top_k most relevant.

    Unlike embedding similarity (which compares two independently-computed vectors), a reranker reads the
    query and each candidate together, so it can tell "topically related" apart from "actually answers this".
    """
    documents = [c["text"] for c in candidates]
    voyage_limiter.acquire()
    result = client.rerank(query, documents, model=RERANK_MODEL, top_k=top_k)
    return [{**candidates[r.index], "score": r.relevance_score} for r in result.results]
