from __future__ import annotations
import re as _re
from datetime import date
from .models import Paper

# Paper must contain at least one AI/agent term …
AI_AGENT_TERMS = [
    # Agent / LLM core
    'agentic', 'multi-agent', 'multiagent',
    'llm', 'large language model', 'language model',
    'gpt', 'chatgpt', 'foundation model', 'generative ai',
    'copilot', 'co-pilot', 'co-scientist',
    # Autonomous lab / robotic science
    'autonomous',  # covers self-driving labs, autonomous experiments, etc.
    'self-driving lab', 'self-driving laboratory',
    'robotic',     # covers robotic chemist, robotic synthesis, robotic platform
    'robot chemist', 'robot scientist',
    'ai agent', 'ai-driven',
    # Tool use / closed-loop
    'tool-augmented', 'tool use', 'tool-use', 'tool calling',
    'closed-loop',
    # Broad AI/ML
    'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'reinforcement learning',
]

# … AND at least one science-domain term
SCIENCE_TERMS = [
    # Chemistry
    'chemistry', 'chemical', 'synthesis', 'molecule', 'reaction',
    'catalyst', 'reagent', 'organic', 'inorganic', 'biochem',
    # Materials
    'material', 'crystal', 'alloy', 'polymer', 'battery', 'semiconductor',
    'perovskite', 'nanomaterial', 'composite',
    # Biology / medicine ('gene' omitted — substring of 'generative'/'general')
    'biology', 'protein', 'drug', 'cell', 'enzyme', 'genomic',
    'genetic', 'genome', 'biomedical', 'pharmacol', 'medicin',
    'disease', 'cancer', 'neuroscien',
    # Scientific context (keeps survey / review papers about AI for science)
    'scientific', 'science', 'experiment', 'laboratory',
    'discovery', 'hypothesis',
]

TOP_VENUES = {
    'nature', 'science', 'cell', 'jacs', 'journal of the american chemical society',
    'angewandte chemie', 'advanced materials', 'acs nano', 'nature chemistry',
    'nature materials', 'nature biotechnology', 'nature methods', 'nature communications',
    'chemical science', 'acs central science', 'matter', 'joule',
    'neurips', 'advances in neural information processing',
    'icml', 'iclr', 'aaai', 'acl', 'emnlp',
    'pnas', 'proceedings of the national academy',
    'journal of chemical information', 'jcim',
}


def importance_score(paper: Paper) -> float:
    """Higher = more important. Components:
      - citations_per_year: age-normalised citation rate
      - venue_bonus: top journal/conference
      - recency_boost: papers < 60 days old get a boost so they surface even without citations yet
    """
    today = date.today()
    citations = paper.citation_count or 0

    if paper.published_date:
        age_days = max((today - paper.published_date).days, 1)
        age_years = age_days / 365.25
    else:
        age_days = 365
        age_years = 1.0

    citations_per_year = citations / max(age_years, 0.25)

    venue_lower = (paper.venue or '').lower()
    venue_bonus = 40.0 if any(v in venue_lower for v in TOP_VENUES) else 0.0

    # Papers < 60 days old: linearly decaying boost up to +30 at day 0
    recency_boost = max(0.0, (60 - age_days) * 0.5) if age_days < 60 else 0.0

    return citations_per_year + venue_bonus + recency_boost


# Strong agentic/autonomous signals — their presence pushes relevance up
_STRONG_TERMS = [
    'agentic', 'multi-agent', 'multiagent', 'ai agent', 'llm agent',
    'self-driving lab', 'self-driving laboratory', 'autonomous laboratory',
    'autonomous experiment', 'robot chemist', 'robot scientist', 'co-scientist',
    'closed-loop', 'tool-augmented',
]
_CHEM_MAT_TERMS = [
    'chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst',
    'reagent', 'material', 'crystal', 'alloy', 'polymer', 'battery',
    'semiconductor', 'perovskite', 'nanomaterial',
    # computational chemistry / materials (atomistic simulation subfield)
    'atomistic', 'molecular dynamics', 'density functional', 'first-principles',
    'computational chemistry', 'computational materials',
]


def _heuristic_relevance(paper: Paper) -> int:
    """Cheap 0-100 relevance estimate used until the LLM assigns a graded score.

    Rewards agentic/autonomous signals and chemistry/materials grounding, with
    title hits weighted above abstract hits. Deliberately conservative so that
    LLM scores (when present) dominate ordering.
    """
    if not is_on_topic(paper):
        return 0
    title = paper.title.lower()
    abstract = (paper.abstract or '').lower()
    score = 40  # on-topic floor
    score += 20 * sum(1 for t in _STRONG_TERMS if t in title)
    score += 8 * sum(1 for t in _STRONG_TERMS if t in abstract and t not in title)
    score += 10 * sum(1 for t in _CHEM_MAT_TERMS if t in title)
    score += 3 * sum(1 for t in _CHEM_MAT_TERMS if t in abstract and t not in title)
    return max(0, min(100, score))


def relevance_value(paper: Paper) -> int:
    """Primary ranking signal: LLM-assigned relevance (0-100), with a keyword
    heuristic fallback for papers not yet classified by the LLM."""
    if paper.relevance is not None:
        return paper.relevance
    return _heuristic_relevance(paper)


def is_on_topic(paper: Paper) -> bool:
    """Return True if the paper is about agentic/autonomous AI applied to science.

    Requires both an AI/agent signal and a science-domain signal so that
    general LLM surveys, AGI debates, and commerce/security papers are excluded.
    """
    text = (paper.title + ' ' + (paper.abstract or '')).lower()
    return (
        any(term in text for term in AI_AGENT_TERMS) and
        any(term in text for term in SCIENCE_TERMS)
    )


# Known predatory / pay-to-publish outlets whose papers the LLM scores highly on
# title wording alone. Matched as case-insensitive venue substrings.
_PREDATORY_VENUE_TERMS = [
    'journal of information systems engineering',
    'international journal of innovative research',
    'universal library of innovative research',
]
# Predatory-publisher DOI prefixes (registrant codes).
_PREDATORY_DOI_PREFIXES = (
    '10.52783/',   # JISEM
    '10.37082/',   # IJIRMPS
    '10.70315/',   # Universal Library of Innovative Research
    '10.55524/',   # IJIREM
)
# Conference/meeting abstracts (not full papers), e.g. "Abstract 31: ...".
_ABSTRACT_TITLE_RE = _re.compile(r'^\s*abstract\s+\d+\s*:', _re.IGNORECASE)


def is_reputable_venue(paper: Paper) -> bool:
    """False for clearly predatory venues and meeting abstracts, True otherwise.

    Deliberately conservative — only drops known pay-to-publish outlets and
    numbered conference abstracts, so legitimate preprints and journals pass.
    """
    if _ABSTRACT_TITLE_RE.match(paper.title or ''):
        return False
    v = (paper.venue or '').lower()
    if any(term in v for term in _PREDATORY_VENUE_TERMS):
        return False
    doi = (paper.doi or '').lower()
    if doi.startswith(_PREDATORY_DOI_PREFIXES):
        return False
    return True


def is_recent(paper: Paper, days: int) -> bool:
    if not paper.published_date:
        return False
    return (date.today() - paper.published_date).days <= days


_PREPRINT_VENUES = {'', 'arxiv', 'arxiv.org', 'biorxiv', 'chemrxiv', 'medrxiv'}


def is_published(paper: Paper) -> bool:
    """True for peer-reviewed publications; False for preprints (arXiv, bioRxiv, chemRxiv)."""
    if paper.source == 'openreview':
        return True
    if paper.pubmed_id:
        return True
    v = (paper.venue or '').strip().lower()
    if not v or v in _PREPRINT_VENUES or 'preprint' in v or v.startswith('arxiv'):
        # Check DOI as last resort — real publishers have non-arXiv DOIs
        if paper.doi and not paper.doi.startswith('10.48550'):
            # bioRxiv preprint DOIs match 10.1101/YYYY.MM.DD.*
            if not _re.match(r'10\.1101/\d{4}\.\d{2}\.\d{2}', paper.doi):
                return True
        return False
    return True
