import os

import anthropic
from dotenv import load_dotenv

from fdi.embeddings import embed_query
from fdi.memo import verify_section
from fdi.reranker import rerank
from fdi.vector_store import VectorStore
from fdi.verification import verify_numbers_in_context

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)
MODEL = "claude-sonnet-5"


def _build_context(results: list[dict]) -> str:
    """Join retrieved chunks into one context string, each tagged with the source it came from."""
    parts = [f"[{r['category']} / {r['source_path']}]\n{r['text']}" for r in results]
    return "\n\n".join(parts)


def answer_question(question: str, store: VectorStore, candidate_k: int = 30, top_k: int = 10) -> tuple[str, list[dict]]:
    """Retrieve a wide candidate pool by embedding similarity, rerank it for actual relevance, then ask Claude
    to answer using only the reranked context."""
    query_embedding = embed_query(question)
    candidates = store.search(query_embedding, top_k=candidate_k)
    results = rerank(question, candidates, top_k=top_k)
    context = _build_context(results)

    prompt = (
        "Answer the question using only the context below, drawn from a company's VDR "
        "(virtual data room) filings. If the context doesn't contain the answer, say so "
        "plainly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = next(block.text for block in response.content if block.type == "text")
    return answer, results


def verify_answer(answer: str, results: list[dict]) -> tuple[list, list[str]]:
    """Ground-check an answer against the context it was retrieved from, reusing the memo pipeline's checks."""
    context = _build_context(results)
    unverified_claims = verify_section(answer, context)
    unmatched_numbers = verify_numbers_in_context(answer, context)
    return unverified_claims, unmatched_numbers
