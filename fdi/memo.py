import json
import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from fdi.knowledge_base import load_category_facts
from fdi.verification import verify_numbers_in_context

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)
MODEL = "claude-sonnet-5"

TONE_INSTRUCTIONS = (
    "Write in the persuasive, confident, data-driven tone of a sell-side investment bank's "
    "Confidential Information Memorandum. Use concrete numbers to support every claim. "
    "Do not invent facts -- use only the context provided."
)

SECTIONS = [
    ("Executive Summary", "A concise, compelling overview of the investment opportunity: what the company does, its scale, and why it is attractive."),
    ("Company Overview", "The company's history, business model, operating segments, and scale of operations."),
    ("Market Overview and Company Positioning", "The industry the company operates in and how it is positioned relative to competitors."),
    ("Product Overview", "The company's core products/services and what differentiates them."),
    ("Clients, Sales & Marketing Overview", "How the company acquires and retains customers, and its marketing approach."),
    ("Growth Strategy", "The company's strategy and concrete plans for future growth."),
    ("Financial Overview", "A summary of the company's key financial results and trends, citing specific figures."),
]


def generate_section(name: str, instructions: str, context: str) -> str:
    """Draft one memo section in a single Claude call, grounded only in the given context."""
    prompt = (
        f"{TONE_INSTRUCTIONS}\n\n"
        f"Write the '{name}' section of a Confidential Information Memorandum for Planet Fitness, Inc.\n"
        f"Section focus: {instructions}\n\n"
        f"Context:\n{context}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


class UnverifiedClaim(BaseModel):
    claim: str
    reason: str


class VerificationResult(BaseModel):
    unverified_claims: list[UnverifiedClaim]


def verify_section(section_text: str, context: str) -> list[UnverifiedClaim]:
    """Check every specific factual claim in a memo section against the context it was supposed to be grounded in."""
    prompt = (
        "You are fact-checking a section of a financial memo. Below is the section text, "
        "followed by the source context it was supposed to be grounded in. List every specific "
        "factual claim in the section (a number, date, statistic, or named fact) that is NOT "
        "directly supported by the source context -- do not flag general business language, only "
        "concrete factual claims. If everything is supported, return an empty list.\n\n"
        f"Section text:\n{section_text}\n\n"
        f"Source context:\n{context}"
    )
    response = client.with_options(timeout=900.0).messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=VerificationResult,
    )
    return response.parsed_output.unverified_claims


def generate_memo(business_context: str) -> str:
    """Generate a full memo, one section at a time, grounded in the extracted knowledge base and business context."""
    facts = load_category_facts("3 - Accounts")
    facts_context = json.dumps(facts, indent=2)

    parts = ["# Planet Fitness, Inc. -- Confidential Information Memorandum\n"]
    for name, instructions in SECTIONS:
        context = f"Structured facts:\n{facts_context}\n\nBusiness context:\n{business_context}"
        section_text = generate_section(name, instructions, context)
        parts.append(f"## {name}\n\n{section_text}\n")

        unverified = verify_section(section_text, context)
        if unverified:
            print(f"  [{name}] {len(unverified)} unverified claim(s) (semantic check):")
            for issue in unverified:
                print(f"    - {issue.claim} ({issue.reason})")

        unmatched_numbers = verify_numbers_in_context(section_text, context)
        if unmatched_numbers:
            print(f"  [{name}] {len(unmatched_numbers)} number(s) not found in source (strict check):")
            for number in unmatched_numbers:
                print(f"    - {number}")
    return "\n".join(parts)
