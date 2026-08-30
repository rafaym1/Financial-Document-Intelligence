import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash-lite"
# Verification is a judgment task, not generation -- a lite-tier model proved unreliable at it here
# (flagging claims that exactly match the source as "unmatched"), mirroring what this project's
# extraction pipeline already found: merging/verifying facts needs a more capable model even when
# drafting text doesn't.
VERIFY_MODEL = "gemini-3.6-flash"


def generate_answer(question: str, context: str) -> str:
    """Answer a question using only the given context, drawn from a company's VDR filings."""
    prompt = (
        "Answer the question using only the context below, drawn from a company's VDR "
        "(virtual data room) filings. If the context doesn't contain the answer, say so "
        "plainly instead of guessing. Give a complete answer: include specific figures, dates, "
        "titles, and any relevant detail or context available in the source rather than the "
        "shortest possible answer -- but don't pad with information the context doesn't support.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


class UnverifiedClaim(BaseModel):
    claim: str
    reason: str


class VerificationResult(BaseModel):
    unverified_claims: list[UnverifiedClaim]


def verify_claims(answer: str, context: str) -> list[UnverifiedClaim]:
    """Check every specific factual claim in the answer against the context it was supposed to be grounded in."""
    prompt = (
        "You are fact-checking an answer to a question about a financial filing. Below is the "
        "answer text, followed by the source context it was supposed to be grounded in. List "
        "every specific factual claim in the answer (a number, date, statistic, or named fact) "
        "that is NOT directly supported by the source context -- do not flag general phrasing, "
        "only concrete factual claims. If everything is supported, return an empty list.\n\n"
        f"Answer:\n{answer}\n\n"
        f"Source context:\n{context}"
    )
    response = client.models.generate_content(
        model=VERIFY_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationResult,
        ),
    )
    return response.parsed.unverified_claims
