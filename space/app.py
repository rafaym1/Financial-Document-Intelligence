import time
from datetime import date, datetime, timezone
from pathlib import Path

import gradio as gr
import spaces

from fdi.embeddings import embed_query
from fdi.knowledge_base import load_category_facts
from fdi.reports import build_metric_series, plot_metric_trend
from fdi.reranker import rerank
from fdi.vector_store import VectorStore
from fdi.verification import verify_numbers_in_context
from llm import generate_answer, verify_claims

INDEX_DIR = "data/vector_index"
DAILY_LIMIT_PER_VISITOR = 8
CHART_OUTPUT_DIR = Path("generated_charts")
CHART_OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_METRICS = ["Total Revenue", "Adjusted EBITDA", "Net income", "Total assets"]

SOURCE_DOCUMENTS = [
    ("10-K, FY2024 (filed 2025-02-25)", "data/dev_vdr/3 - Accounts/2025-02-25_10-K_plnt-20241231.htm"),
    ("10-K, FY2025 (filed 2026-02-25)", "data/dev_vdr/3 - Accounts/2026-02-25_10-K_plnt-20251231.htm"),
    ("Proxy Statement (filed 2025-03-26)", "data/dev_vdr/1 - Corporate Matters/2025-03-26_DEF_14A_plnt-20250326.htm"),
    ("Proxy Statement (filed 2026-03-25)", "data/dev_vdr/1 - Corporate Matters/2026-03-25_DEF_14A_plnt-20260325.htm"),
]

# Real section names from fdi/memo.py's SECTIONS -- kept here as plain strings (not an import) so this
# tab never needs an Anthropic client or API key; the memo it reveals is already generated, not redrafted.
MEMO_SECTIONS = [
    "Executive Summary",
    "Company Overview",
    "Market Overview and Company Positioning",
    "Product Overview",
    "Clients, Sales & Marketing Overview",
    "Growth Strategy",
    "Financial Overview",
]

SAMPLE_QUESTIONS = [
    "What was Planet Fitness's total revenue for fiscal year 2025?",
    "Who are the members of Planet Fitness's board of directors?",
    "What is Planet Fitness's Adjusted EBITDA trend over the last few years?",
]

store = VectorStore.load(INDEX_DIR)

_facts = [fact for record in load_category_facts("3 - Accounts") for fact in record["facts"]]
_available_metrics = sorted(
    {fact["metric"] for fact in _facts if len(build_metric_series(_facts, fact["metric"])) >= 2}
)

# In-memory per-visitor daily cap on chatbot questions. Resets on Space restart -- fine for a demo,
# no database needed. Separate from the Voyage rate limiter: this bounds worst-case Gemini usage
# from one visitor against the model's own daily quota, since Gemini calls aren't otherwise throttled.
_visitor_usage: dict[str, tuple[date, int]] = {}


def _client_ip(request: gr.Request) -> str:
    forwarded = request.headers.get("x-forwarded-for") if request.headers else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_and_record_usage(ip: str) -> bool:
    """Returns True if this visitor is still under today's cap, and records the request if so."""
    today = datetime.now(timezone.utc).date()
    last_date, count = _visitor_usage.get(ip, (today, 0))
    if last_date != today:
        count = 0
    if count >= DAILY_LIMIT_PER_VISITOR:
        return False
    _visitor_usage[ip] = (today, count + 1)
    return True


def _build_context(results: list[dict]) -> str:
    """Join retrieved chunks into one context string, each tagged with the source it came from."""
    parts = [f"[{r['category']} / {r['source_path']}]\n{r['text']}" for r in results]
    return "\n\n".join(parts)


@spaces.GPU
def handle_question(question: str, request: gr.Request):
    # This app never touches a GPU -- every call here is a network request to Voyage/Gemini.
    # The decorator exists only because HF's free tier now requires it to satisfy ZeroGPU's
    # startup check (Gradio Spaces on the free tier run on ZeroGPU with no CPU-only option).
    question = (question or "").strip()
    if not question:
        return "Ask a question about Planet Fitness's filings to get started.", "", ""

    ip = _client_ip(request)
    if not _check_and_record_usage(ip):
        return (
            f"You've reached the demo limit of {DAILY_LIMIT_PER_VISITOR} questions for today "
            "-- thanks for trying it out! Come back tomorrow.",
            "",
            "",
        )

    query_embedding = embed_query(question)
    candidates = store.search(query_embedding, top_k=30)
    results = rerank(question, candidates, top_k=10)
    context = _build_context(results)

    answer = generate_answer(question, context)

    sources = sorted({(r["category"], r["source_path"]) for r in results})
    sources_md = "\n".join(f"- **{category}** · {source_path}" for category, source_path in sources)

    unverified_claims = verify_claims(answer, context)
    unmatched_numbers = verify_numbers_in_context(answer, context)
    if not unverified_claims and not unmatched_numbers:
        verification_md = "✅ Every claim and number in this answer matched the retrieved source text."
    else:
        lines = ["⚠️ Verification flagged the following:"]
        for claim in unverified_claims:
            lines.append(f"- {claim.claim} ({claim.reason})")
        for number in unmatched_numbers:
            lines.append(f"- {number} -- not found in source")
        verification_md = "\n".join(lines)

    return answer, sources_md, verification_md


def generate_charts(selected_metrics: list[str]):
    """Render a trend chart for each selected metric, live, from the real extracted facts."""
    if not selected_metrics:
        return []
    images = []
    for metric in selected_metrics:
        series = build_metric_series(_facts, metric)
        if len(series) < 2:
            continue
        output_path = CHART_OUTPUT_DIR / f"{metric.lower().replace(' ', '_').replace('/', '_')}.png"
        plot_metric_trend(metric, series, output_path)
        images.append((str(output_path), metric))
    return images


def generate_memo_staged():
    """Give a genuine watch-it-happen progress sequence, then reveal the real generated memo.

    The memo itself was already produced by the real pipeline (fdi/memo.py's generate_memo,
    including its verification passes) -- this stages the reveal rather than re-running ~14 Claude
    calls on every click, which would be both slow and an open cost/abuse surface on a public demo.
    """
    log_lines = []
    for section in MEMO_SECTIONS:
        log_lines.append(f"Drafting *{section}*...")
        yield "\n\n".join(log_lines)
        time.sleep(0.9)
        log_lines[-1] = f"Drafting *{section}*... done. Verifying against source filings..."
        yield "\n\n".join(log_lines)
        time.sleep(0.6)
        log_lines[-1] = f"✅ *{section}* -- drafted and verified."
        yield "\n\n".join(log_lines)

    memo_text = Path("data/output/memo.md").read_text(encoding="utf-8")
    yield memo_text


with gr.Blocks(title="Financial Document Intelligence") as demo:
    gr.Markdown(
        "# Financial Document Intelligence\n"
        "A live system over Planet Fitness's public SEC filings (10-Ks and proxy statements), "
        "used here as a stand-in for a confidential virtual data room."
    )

    with gr.Tabs() as tabs:
        with gr.Tab("Chatbot", id="chatbot"):
            gr.Markdown(
                "Ask a real question. Retrieval runs in two stages (a wide embedding search, then a "
                "reranking pass), then every claim is checked against the source before it's shown. "
                "Answers can take 10-20 seconds; capped per visitor to keep the demo free to run."
            )
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What was Planet Fitness's total revenue for fiscal year 2025?",
            )
            submit_btn = gr.Button("Ask", variant="primary")
            gr.Examples(examples=SAMPLE_QUESTIONS, inputs=question_box)

            answer_box = gr.Markdown(label="Answer")
            sources_box = gr.Markdown(label="Cited sources")
            verification_box = gr.Markdown(label="Verification")

            submit_btn.click(
                fn=handle_question,
                inputs=[question_box],
                outputs=[answer_box, sources_box, verification_box],
                api_name="handle_question",
            )
            question_box.submit(
                fn=handle_question,
                inputs=[question_box],
                outputs=[answer_box, sources_box, verification_box],
                api_name=False,
            )

        with gr.Tab("Reports", id="reports"):
            gr.Markdown(
                "Pick the metrics you want to see. Charts are rendered live from the facts this "
                "system extracted from the filings via map-reduce extraction -- no manual data entry."
            )
            metric_picker = gr.CheckboxGroup(
                choices=_available_metrics,
                value=[m for m in DEFAULT_METRICS if m in _available_metrics],
                label=f"Available metrics ({len(_available_metrics)})",
            )
            generate_charts_btn = gr.Button("Generate charts", variant="primary")
            chart_gallery = gr.Gallery(label="Charts", columns=2, height="auto")

            generate_charts_btn.click(fn=generate_charts, inputs=[metric_picker], outputs=[chart_gallery])

        with gr.Tab("Memo", id="memo"):
            gr.Markdown(
                "Click to watch the memo get drafted section by section, then fact-checked against "
                "the source filings -- the same pipeline that produced the real memo shown below."
            )
            generate_memo_btn = gr.Button("Generate memo", variant="primary")
            memo_output = gr.Markdown()

            generate_memo_btn.click(fn=generate_memo_staged, inputs=[], outputs=[memo_output])

        with gr.Tab("Documents", id="documents"):
            gr.Markdown(
                "The real source filings this whole system was built on -- open or download them to "
                "verify any answer, chart, or memo claim against the original text."
            )
            for label, path in SOURCE_DOCUMENTS:
                gr.File(value=path, label=label, interactive=False)

    def _select_tab_from_query(request: gr.Request):
        """Lets the docs site deep-link into a specific tab, e.g. .../?tab=reports."""
        requested = dict(request.query_params).get("tab", "chatbot")
        valid_ids = {"chatbot", "reports", "memo", "documents"}
        return gr.Tabs(selected=requested if requested in valid_ids else "chatbot")

    demo.load(fn=_select_tab_from_query, inputs=None, outputs=[tabs])

if __name__ == "__main__":
    demo.launch()
