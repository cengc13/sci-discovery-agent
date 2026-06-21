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
For each paper below, classify four fields:
- paper_type: "review" for review/survey/perspective/overview articles; "article" for original research
- on_topic: true only if the paper's PRIMARY contribution is using AI/ML/LLM/autonomous-agent methods \
to conduct scientific research (chemistry, materials science, biology, drug discovery, etc.); \
false otherwise (instrument papers, pure chemistry/biology without AI, exobiology missions, \
space-hardware papers that only mention "autonomous" for robotic navigation)
- relevance: an integer 0-100 (use the FULL range and avoid clustering at round multiples of 5 — \
give each paper a precise, discriminating score) rating how central the paper is to: \
AI AGENTS / autonomous systems / LLM agents that CONDUCT chemistry and materials research. \
The decisive factor is an explicit AGENT or AUTONOMOUS DECISION-MAKING LOOP — an AI/LLM agent, \
multi-agent system, or self-driving/autonomous laboratory that plans, acts, observes and iterates \
to drive the science — NOT merely the use of AI/ML. Score with this scale: \
95-100 = landmark, complete agentic/autonomous systems that autonomously run chemistry or materials \
research end-to-end (self-driving labs closing the loop on real experiments; flagship LLM/multi-agent \
chemists; autonomous computational or atomistic research agents); \
85-94 = strong, clearly agentic/autonomous systems for chemistry or materials where an explicit agent \
or autonomous loop IS the core contribution, even if simulation-only or narrower in scope; \
70-84 = agentic/autonomous AI where chemistry/materials is the application but the agentic element is \
partial or early-stage, OR agentic/autonomous AI for ADJACENT science (drug discovery, biology); \
50-69 = AI/ML for chemistry or materials WITHOUT an agent or autonomous loop (generative models, \
property/structure predictors, molecular-dynamics or score-based methods), or general AI-for-science \
not specific to chemistry/materials; \
30-49 = tangential (AI with only incidental chemistry/materials, or chemistry/materials with only \
incidental AI); 0-29 = off-topic. \
A paper with NO agent or autonomous component must not exceed 69. If on_topic is false, relevance < 40. \
Reserve 95+ for the most central and complete works; most genuinely agentic papers belong in 70-90.
- venue_normalized: the standard journal/conference name if you can confidently identify it \
(e.g. "Cell Physical Science", "Nature Communications"); null if uncertain

Papers:
{papers_json}

Return a JSON array:
[{{"id": 0, "paper_type": "article", "on_topic": true, "relevance": 85, "venue_normalized": null}}, ...]"""


def _call_openai(api_key: str, papers_batch: list[dict], model: str) -> tuple[list[dict], int, int]:
    """Returns (results, input_tokens, output_tokens)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    items = [
        {
            "id": i,
            "title": p["title"],
            "abstract": (p.get("abstract") or "")[:700],
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
    candidates = [p for p in papers if is_on_topic(p)
                  and (force or p.llm_on_topic is None or p.relevance is None)]
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
                rel = r.get("relevance")
                if isinstance(rel, (int, float)):
                    p.relevance = max(0, min(100, int(rel)))
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


_VERIFY_PROMPT = """\
Is this GitHub repository plausibly the official code release for the paper?
Answer "yes" if the repository name matches the paper's system/method name, OR \
the description/README describes the same method. Answer "no" only if it is \
clearly about something unrelated. Reply with a single word — "yes" or "no".

Paper: {title}
Abstract: {abstract}

Repository: {full_name}
Description: {description}
Topics: {topics}
README (excerpt): {readme}"""


def verify_code_url(api_key: str, model: str,
                    paper_title: str, paper_abstract: str,
                    repo: dict) -> bool:
    """Return True if the LLM believes repo is the code release for the paper."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = _VERIFY_PROMPT.format(
        title=paper_title,
        abstract=(paper_abstract or '')[:250],
        full_name=repo.get('full_name', ''),
        description=repo.get('description') or 'none',
        topics=', '.join(repo.get('topics') or []) or 'none',
        readme=(repo.get('readme') or 'none')[:600],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().lower().startswith('yes')
    except Exception as e:
        logger.warning("Code URL verification failed: %s", e)
        return False
