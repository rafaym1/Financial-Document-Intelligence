import json
import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from fdi.chunking import chunk_text
from fdi.schema import Document

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)
MODEL = "claude-sonnet-5"


class FinancialHighlight(BaseModel):
    metric: str
    value: str
    fiscal_period: str
    source_excerpt: str


class FinancialHighlightList(BaseModel):
    highlights: list[FinancialHighlight]


def extract_financial_highlights(document: Document) -> list[FinancialHighlight]:
    """Ask Claude to pull key financial line items out of a filing, each grounded in a verbatim source excerpt."""
    prompt = (
        "Extract the key financial highlights (e.g. revenue, net income, total assets, "
        "total debt) from this financial filing. For each one, give the exact fiscal period "
        "and a short verbatim excerpt from the text that supports the figure.\n\n"
        f"{document.text}"
    )
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=FinancialHighlightList,
    )
    return response.parsed_output.highlights


def extract_financial_highlights_chunked(document: Document) -> list[FinancialHighlight]:
    """Extract financial highlights chunk-by-chunk instead of in one call, avoiding the 'lost in the middle' effect on very long documents."""
    all_highlights: list[FinancialHighlight] = []
    for chunk in chunk_text(document.text):
        chunk_document = document.model_copy(update={"text": chunk})
        all_highlights.extend(extract_financial_highlights(chunk_document))
    return merge_highlights(_dedupe_highlights(all_highlights))


def _dedupe_highlights(highlights: list[FinancialHighlight]) -> list[FinancialHighlight]:
    """Collapse exact-phrasing duplicates re-extracted from overlapping chunks, keeping the most detailed source excerpt."""
    best: dict[tuple[str, str], FinancialHighlight] = {}
    for highlight in highlights:
        key = (highlight.metric.strip().lower(), highlight.fiscal_period.strip().lower())
        if key not in best or len(highlight.source_excerpt) > len(best[key].source_excerpt):
            best[key] = highlight
    return list(best.values())


def merge_highlights(highlights: list[FinancialHighlight], model: str = MODEL) -> list[FinancialHighlight]:
    """Ask Claude to merge near-duplicate facts (same underlying metric and period, phrased differently) into one canonical list."""
    prompt = (
        "The following financial facts were extracted independently from overlapping chunks of the same "
        "document, so some are duplicates referring to the same underlying metric and period, just phrased "
        "differently (different wording, date formats, or rounding). Merge them into one entry per distinct "
        "(metric, fiscal period) combination, preferring the most precise value and most detailed source "
        "excerpt when merging duplicates.\n\n"
        f"{json.dumps([h.model_dump() for h in highlights], indent=2)}"
    )
    response = client.with_options(timeout=900.0).messages.parse(
        model=model,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
        output_format=FinancialHighlightList,
    )
    return response.parsed_output.highlights
