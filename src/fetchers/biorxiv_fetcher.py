from __future__ import annotations
import re
import time
import logging
import requests
from datetime import date
from ..models import Paper

logger = logging.getLogger(__name__)

# chemRxiv DOI prefix (registered with CrossRef)
CROSSREF_API = 'https://api.crossref.org/works'
CHEMRXIV_PREFIX = '10.26434'

# Strip JATS XML tags from CrossRef abstracts
_JATS_TAG = re.compile(r'<[^>]+>')

CHEMRXIV_QUERIES = [
    'AI agent autonomous chemistry materials discovery',
    'LLM tool-augmented chemistry synthesis planning',
    'self-driving laboratory robotic chemistry autonomous experiment',
    'closed-loop autonomous chemical synthesis materials',
    'agentic AI chemistry materials scientific discovery',
]

DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'battery', 'perovskite'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'genomic', 'biolog', 'biochem', 'enzyme'],
}

AGENT_TERMS = [
    'ai agent', 'llm agent', 'agentic', 'multi-agent', 'autonomous agent',
    'self-driving', 'autonomous lab', 'autonomous experiment', 'robotic chemistry',
    'tool-augmented', 'tool use', 'closed-loop', 'automated workflow',
    'autonomous workflow', 'autonomous discovery',
]

SCIENCE_TERMS = [
    'chemistry', 'chemical', 'materials', 'drug discovery', 'drug design',
    'molecular design', 'synthesis planning', 'retrosynthesis',
    'scientific discovery',
]


def _matches(title: str, abstract: str) -> bool:
    text = (title + ' ' + (abstract or '')).lower()
    return any(t in text for t in AGENT_TERMS) and any(t in text for t in SCIENCE_TERMS)


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + (abstract or '')).lower()
    domains = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['chemistry']


def _clean_abstract(raw: str) -> str:
    return _JATS_TAG.sub('', raw).strip()


def _parse_date(item: dict) -> date | None:
    pub = item.get('published') or item.get('published-print') or item.get('published-online')
    if not pub:
        return None
    parts = pub.get('date-parts', [[]])[0]
    try:
        y = parts[0]
        m = parts[1] if len(parts) > 1 else 1
        d = parts[2] if len(parts) > 2 else 1
        return date(y, m, d)
    except (IndexError, TypeError, ValueError):
        return None


class BiorxivFetcher:
    """chemRxiv-only fetcher via CrossRef API (DOI prefix 10.26434)."""

    def __init__(self, config: dict):
        import os
        self.config = config
        self.max_results = config.get('max_results', 200)
        self.mailto = config.get('mailto') or os.environ.get('PUBMED_EMAIL', 'user@example.com')
        self._headers = {
            'User-Agent': f'paper-fetcher/1.0 (mailto:{self.mailto})',
        }

    def fetch(self, since: date, until: date) -> list[Paper]:
        return self._fetch_chemrxiv(since, until)

    def _fetch_chemrxiv(self, since: date, until: date) -> list[Paper]:
        seen_dois: set[str] = set()
        papers: list[Paper] = []

        for query in CHEMRXIV_QUERIES:
            batch = self._crossref_search(query, since, until)
            for item in batch:
                doi = (item.get('DOI') or '').lower()
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)

                title = ' '.join(item.get('title') or [])
                abstract = _clean_abstract(item.get('abstract') or '')
                if not title or not _matches(title, abstract):
                    continue

                pub_date = _parse_date(item)
                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (item.get('author') or [])
                ]

                papers.append(Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    source='chemrxiv',
                    published_date=pub_date,
                    doi=item.get('DOI'),
                    url=f"https://doi.org/{item['DOI']}" if item.get('DOI') else None,
                    venue='chemRxiv',
                    domains=_infer_domains(title, abstract),
                ))

            time.sleep(0.5)

        logger.info(f'chemRxiv (CrossRef): {len(papers)} papers')
        return papers

    def _crossref_search(self, query: str, since: date, until: date) -> list[dict]:
        try:
            r = requests.get(
                CROSSREF_API,
                headers=self._headers,
                params={
                    'filter': f'prefix:{CHEMRXIV_PREFIX},from-pub-date:{since},until-pub-date:{until}',
                    'query': query,
                    'rows': min(self.max_results, 100),
                    'select': 'title,abstract,author,published,DOI,URL',
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get('message', {}).get('items', [])
        except Exception as e:
            logger.warning(f'chemRxiv CrossRef search failed for "{query}": {e}')
            return []
