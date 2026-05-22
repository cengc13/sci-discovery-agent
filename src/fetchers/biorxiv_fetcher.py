from __future__ import annotations
import time
import logging
import requests
from datetime import date
from ..models import Paper

logger = logging.getLogger(__name__)

BIORXIV_API = 'https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json'
CHEMRXIV_API = 'https://chemrxiv.org/engage/chemrxiv/public-api/v1/items'

# A paper must match at least one AGENT term AND one SCIENCE term to be included.
AGENT_TERMS = [
    'ai agent', 'llm agent', 'agentic', 'multi-agent', 'autonomous agent',
    'self-driving', 'autonomous lab', 'autonomous experiment', 'robotic chemistry',
    'tool-augmented', 'tool use', 'closed-loop', 'automated workflow',
    'autonomous workflow', 'autonomous discovery',
]

SCIENCE_TERMS = [
    'chemistry', 'chemical synthesis', 'materials', 'drug discovery', 'drug design',
    'protein design', 'molecular design', 'synthesis planning', 'retrosynthesis',
    'scientific discovery', 'biology', 'genomics',
]

DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'battery'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'genomic', 'biolog', 'biochem', 'enzyme'],
}

CHEMRXIV_QUERIES = [
    'AI agent autonomous chemistry materials discovery',
    'LLM tool-augmented chemistry synthesis planning',
    'self-driving laboratory robotic chemistry autonomous experiment',
    'closed-loop autonomous chemical synthesis materials',
]


def _matches(title: str, abstract: str) -> bool:
    text = (title + ' ' + (abstract or '')).lower()
    has_agent = any(t in text for t in AGENT_TERMS)
    has_science = any(t in text for t in SCIENCE_TERMS)
    return has_agent and has_science


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + (abstract or '')).lower()
    domains = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['biology']


class BiorxivFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.max_results = config.get('max_results', 200)
        self.include_chemrxiv = config.get('include_chemrxiv', True)

    def fetch(self, since: date, until: date) -> list[Paper]:
        papers: list[Paper] = []
        papers.extend(self._fetch_server('biorxiv', since, until))
        papers.extend(self._fetch_server('medrxiv', since, until))
        if self.include_chemrxiv:
            papers.extend(self._fetch_chemrxiv(since, until))
        return papers

    def _fetch_server(self, server: str, since: date, until: date) -> list[Paper]:
        papers: list[Paper] = []
        cursor = 0
        start_str = since.strftime('%Y-%m-%d')
        end_str = until.strftime('%Y-%m-%d')

        while True:
            url = BIORXIV_API.format(server=server, start=start_str, end=end_str, cursor=cursor)
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                collection = data.get('collection', [])
                if not collection:
                    break

                for item in collection:
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')
                    if not _matches(title, abstract):
                        continue

                    doi = item.get('doi') or ''
                    biorxiv_id = doi.split('/')[-1] if doi else None
                    pub_date_str = item.get('date', '')
                    try:
                        pub_date: date | None = date.fromisoformat(pub_date_str) if pub_date_str else None
                    except ValueError:
                        pub_date = None

                    authors_raw = item.get('authors', '')
                    authors = [a.strip() for a in authors_raw.split(';')] if authors_raw else []

                    papers.append(Paper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        source=server,
                        published_date=pub_date,
                        doi=doi or None,
                        biorxiv_id=biorxiv_id,
                        url=f"https://www.biorxiv.org/content/{doi}v1" if doi else None,
                        venue=server,
                        domains=_infer_domains(title, abstract),
                    ))

                messages = data.get('messages', [{}])
                total = messages[0].get('total', 0) if messages else 0
                cursor += len(collection)
                if cursor >= total or cursor >= self.max_results * 3:
                    break
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"{server} fetch failed at cursor {cursor}: {e}")
                break

        return papers

    def _fetch_chemrxiv(self, since: date, until: date) -> list[Paper]:
        papers: list[Paper] = []

        for term in CHEMRXIV_QUERIES:
            try:
                r = requests.get(
                    CHEMRXIV_API,
                    params={'term': term, 'sort': 'published_date_desc', 'limit': 50, 'skip': 0},
                    timeout=30,
                )
                r.raise_for_status()
                items = r.json().get('itemHits', [])

                for wrapper in items:
                    item = wrapper.get('item', {})
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')

                    pub_date_str = (item.get('publishedDate') or '')[:10]
                    try:
                        pub_date: date | None = date.fromisoformat(pub_date_str) if pub_date_str else None
                    except ValueError:
                        pub_date = None

                    if pub_date and (pub_date < since or pub_date > until):
                        continue

                    doi = item.get('doi') or ''
                    authors = [
                        f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
                        for a in (item.get('authors') or [])
                    ]

                    papers.append(Paper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        source='chemrxiv',
                        published_date=pub_date,
                        doi=doi or None,
                        url=f"https://doi.org/{doi}" if doi else None,
                        venue='chemRxiv',
                        domains=_infer_domains(title, abstract),
                    ))

                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"chemRxiv fetch failed for '{term}': {e}")

        return papers
