"""LLM-based paper classification using OpenAI (gpt-4o-mini by default)."""
from __future__ import annotations
import json
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4o-mini'
BATCH_SIZE = 15

_SYSTEM = (
    "You are a research librarian specialising in AI-for-science literature. "
    "Respond only with valid JSON — no markdown fences, no commentary."
)

_USER_TMPL = """\
For each paper below, classify three fields:
- paper_type: "review" for review/survey/perspective/overview articles; "article" for original research
- on_topic: true only if the paper's PRIMARY contribution is using AI/ML/LLM/autonomous-agent methods \
to conduct scientific research (chemistry, materials science, biology, drug discovery, etc.); \
false otherwise (instrument papers, pure chemistry/biology without AI, exobiology missions, \
space-hardware papers that only mention "autonomous" for robotic navigation)
- venue_normalized: the standard journal/conference name if you can confidently identify it \
(e.g. "Cell Physical Science", "Nature Communications"); null if uncertain

Papers:
{papers_json}

Return a JSON array:
[{{"id": 0, "paper_type": "article", "on_topic": true, "venue_normalized": null}}, ...]"""


def _call_openai(api_key: str, papers_batch: list[dict], model: str) -> tuple[list[dict], int, int]:
    """Returns (results, input_tokens, output_tokens)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    items = [
        {
            "id": i,
            "title": p["title"],
            "abstract": (p.get("abstract") or "")[:300],
            "venue": p.get("venue") or "",
        }
        for i, p in enumerate(papers_batch)
    ]
    prompt = _USER_TMPL.format(papers_json=json.dumps(items, ensure_ascii=False))
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0

    raw = json.loads(response.choices[0].message.content)
    if isinstance(raw, list):
        return raw, in_tok, out_tok
    for v in raw.values():
        if isinstance(v, list):
            return v, in_tok, out_tok
    return [], in_tok, out_tok


# gpt-4o-mini pricing (per 1M tokens, as of 2025)
_PRICE_IN  = {'gpt-4o-mini': 0.15, 'gpt-4o': 2.50}
_PRICE_OUT = {'gpt-4o-mini': 0.60, 'gpt-4o': 10.0}


def enrich_papers_llm(papers: list, api_key: str, model: str = DEFAULT_MODEL,
                       force: bool = False, max_papers: int = 0) -> int:
    """Classify unclassified on-topic papers.

    Args:
        max_papers: if > 0, cap the number of papers classified this run (safety limit)
    Returns:
        count of classified papers
    """
    from .scoring import is_on_topic
    candidates = [p for p in papers if is_on_topic(p) and (force or p.llm_on_topic is None)]
    if not candidates:
        logger.info("LLM enrichment: all on-topic papers already classified")
        return 0

    if max_papers > 0:
        candidates = candidates[:max_papers]

    logger.info("LLM enrichment: classifying %d papers with %s", len(candidates), model)
    classified = 0
    total_in, total_out = 0, 0

    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]
        batch_dicts = [{"title": p.title, "abstract": p.abstract, "venue": p.venue}
                       for p in batch]
        try:
            results, in_tok, out_tok = _call_openai(api_key, batch_dicts, model)
            total_in += in_tok
            total_out += out_tok
            for r in results:
                idx = r.get("id", -1)
                if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                    continue
                p = batch[idx]
                p.paper_type = r.get("paper_type")
                p.llm_on_topic = r.get("on_topic")
                if r.get("venue_normalized"):
                    p.venue_llm = r["venue_normalized"]
                classified += 1
        except Exception as e:
            logger.warning("LLM enrichment batch %d failed: %s",
                           batch_start // BATCH_SIZE + 1, e)

        if batch_start + BATCH_SIZE < len(candidates):
            time.sleep(0.5)

    price_in  = _PRICE_IN.get(model, 0.15)
    price_out = _PRICE_OUT.get(model, 0.60)
    cost = (total_in * price_in + total_out * price_out) / 1_000_000
    logger.info("LLM enrichment: %d input tokens, %d output tokens, est. $%.4f",
                total_in, total_out, cost)
    return classified
