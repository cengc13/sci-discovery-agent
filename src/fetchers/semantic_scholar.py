from __future__ import annotations
import time
import logging
import requests
from datetime import date
from ..models import Paper

logger = logging.getLogger(__name__)

BASE_URL = 'https://api.semanticscholar.org/graph/v1'
FIELDS = 'title,authors,abstract,year,externalIds,publicationDate,venue,citationCount,openAccessPdf'

QUERIES = [
    'AI agent autonomous chemistry materials biology scientific discovery',
    'LLM agent tool-augmented chemistry biology drug discovery',
    'self-driving laboratory autonomous experiment chemistry materials',
    'large language model autonomous planning workflow chemistry biology',
    'multi-agent system scientific discovery chemistry biology',
    'closed-loop autonomous experiment materials chemistry drug discovery',
    'agentic AI scientific research chemistry materials biology',
]

DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'battery', 'perovskite'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'biolog', 'biochem', 'enzyme', 'genomic'],
}


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + (abstract or '')).lower()
    domains = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['AI/science']


class SemanticScholarFetcher:
    def __init__(self, config: dict):
        self.config = config
        import os
        self.api_key = config.get('api_key') or os.environ.get('S2_API_KEY', '')
        self.max_results = config.get('max_results', 100)
        self.max_queries = config.get('max_queries', len(QUERIES))
        self.delay = 1.1  # S2 free key = 1 req/s; be safe

    def fetch(self, since: date | None = None, until: date | None = None) -> list[Paper]:
        seen_ids: set[str] = set()
        papers: list[Paper] = []
        headers = {'x-api-key': self.api_key} if self.api_key else {}

        for i, query in enumerate(QUERIES[:self.max_queries]):
            if i > 0:
                time.sleep(self.delay)
            logger.info(f"S2: {query[:70]}...")
            offset = 0
            while offset < self.max_results:
                try:
                    r = requests.get(
                        f'{BASE_URL}/paper/search',
                        headers=headers,
                        params={
                            'query': query,
                            'fields': FIELDS,
                            'limit': min(100, self.max_results - offset),
                            'offset': offset,
                        },
                        timeout=30,
                    )
                    if r.status_code == 429:
                        retry_after = int(r.headers.get('Retry-After', 10))
                        logger.info(f"S2 rate-limited; waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    r.raise_for_status()
                    batch = r.json().get('data', [])
                    if not batch:
                        break

                    for item in batch:
                        pid = item.get('paperId', '')
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)

                        pub_date = self._parse_date(item)
                        if since is not None and pub_date and pub_date < since:
                            continue
                        if until is not None and pub_date and pub_date > until:
                            continue

                        ext = item.get('externalIds') or {}
                        title = item.get('title') or ''
                        abstract = item.get('abstract') or ''
                        arxiv_id = ext.get('ArXiv')
                        doi = ext.get('DOI')
                        pubmed_id = ext.get('PubMed')

                        oa = item.get('openAccessPdf') or {}
                        url = oa.get('url')
                        if arxiv_id and not url:
                            url = f'https://arxiv.org/abs/{arxiv_id}'
                        elif doi and not url:
                            url = f'https://doi.org/{doi}'

                        papers.append(Paper(
                            title=title,
                            authors=[a['name'] for a in (item.get('authors') or [])],
                            abstract=abstract,
                            source='semantic_scholar',
                            published_date=pub_date,
                            doi=doi,
                            arxiv_id=arxiv_id,
                            pubmed_id=pubmed_id,
                            url=url,
                            venue=item.get('venue') or '',
                            citation_count=item.get('citationCount'),
                            domains=_infer_domains(title, abstract),
                        ))

                    offset += len(batch)
                    if len(batch) < 100:
                        break
                    time.sleep(self.delay)
                except Exception as e:
                    logger.warning(f"S2 query failed at offset {offset}: {e}")
                    break

        return papers

    def _parse_date(self, item: dict) -> date | None:
        pd = item.get('publicationDate')
        if pd:
            try:
                return date.fromisoformat(pd)
            except ValueError:
                pass
        year = item.get('year')
        if year:
            try:
                return date(int(year), 1, 1)
            except (ValueError, TypeError):
                pass
        return None
