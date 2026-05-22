from __future__ import annotations
import logging
import re
import requests
import time
from datetime import date
from ..models import Paper
from ..scoring import is_on_topic

logger = logging.getLogger(__name__)

BASE_URL = 'https://api2.openreview.net'

_GH_PAT = re.compile(r'https?://github\.com/[\w\-][^/\s\)>\]]+/[\w\-][^\s\)>\]]*')

_DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'battery', 'perovskite'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'biolog', 'biochem', 'enzyme', 'genomic'],
}

# Venue IDs for accepted papers at target conferences
_TARGET_VENUES = [
    ('ICLR.cc/2025/Conference', 'ICLR 2025', 2025),
    ('ICLR.cc/2024/Conference', 'ICLR 2024', 2024),
    ('NeurIPS.cc/2024/Conference', 'NeurIPS 2024', 2024),
    ('ICML.cc/2024/Conference', 'ICML 2024', 2024),
]


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + (abstract or '')).lower()
    domains = [d for d, kws in _DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['AI/science']


def _extract_github(abstract: str) -> str | None:
    if not abstract:
        return None
    m = _GH_PAT.search(abstract)
    return m.group(0).rstrip('.,;:)') if m else None


class OpenReviewFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.max_per_venue = config.get('max_per_venue', 4000)
        self.delay = config.get('delay', 0.5)
        self.venue_delay = config.get('venue_delay', 3.0)

    def fetch(self, since: date | None = None, until: date | None = None) -> list[Paper]:
        papers: list[Paper] = []
        seen: set[str] = set()

        for i, (venueid, venue_name, year) in enumerate(_TARGET_VENUES):
            if i > 0:
                time.sleep(self.venue_delay)
            logger.info(f'OpenReview: scanning {venue_name}...')
            count = self._scan_venue(venueid, venue_name, year, seen, papers)
            logger.info(f'OpenReview: {venue_name} → {count} relevant papers')

        logger.info(f'OpenReview: {len(papers)} total relevant papers')
        return papers

    def _scan_venue(self, venueid: str, venue_name: str, year: int,
                    seen: set[str], papers: list[Paper]) -> int:
        found = 0
        offset = 0
        checked = 0

        while checked < self.max_per_venue:
            batch = self._fetch_batch(venueid, limit=100, offset=offset)
            if not batch:
                break
            for note in batch:
                checked += 1
                note_id = note.get('id', '')
                if note_id in seen:
                    continue
                p = self._note_to_paper(note, venue_name, year)
                if p is None or not is_on_topic(p):
                    continue
                seen.add(note_id)
                papers.append(p)
                found += 1
            offset += len(batch)
            if len(batch) < 100:
                break
            time.sleep(self.delay)

        return found

    def _fetch_batch(self, venueid: str, limit: int, offset: int) -> list[dict]:
        backoff = 10
        for attempt in range(3):
            try:
                r = requests.get(
                    f'{BASE_URL}/notes',
                    params={'content.venueid': venueid, 'limit': limit, 'offset': offset},
                    timeout=30,
                )
                if r.status_code == 429:
                    wait = int(r.headers.get('Retry-After', backoff))
                    logger.info(f'OpenReview rate-limited; waiting {wait}s')
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    continue
                r.raise_for_status()
                return r.json().get('notes', [])
            except requests.HTTPError as e:
                logger.warning(f'OpenReview batch failed (offset={offset}): {e}')
                return []
            except Exception as e:
                logger.warning(f'OpenReview batch error (offset={offset}): {e}')
                if attempt < 2:
                    time.sleep(backoff)
                    backoff *= 2
        return []

    def _note_to_paper(self, note: dict, venue_name: str, year: int) -> Paper | None:
        c = note.get('content', {})

        def _val(field: str) -> str:
            v = c.get(field, '')
            return v.get('value', '') if isinstance(v, dict) else (v or '')

        title = _val('title')
        if not title:
            return None

        abstract = _val('abstract')
        authors_raw = c.get('authors', {})
        if isinstance(authors_raw, dict):
            authors_raw = authors_raw.get('value', [])
        authors = authors_raw if isinstance(authors_raw, list) else []

        # Use cdate (ms epoch) for published_date; fall back to conference year
        pub_date: date = date(year, 1, 1)
        cdate = note.get('cdate')
        if cdate:
            try:
                pub_date = date.fromtimestamp(cdate / 1000)
            except (OSError, ValueError, OverflowError):
                pass

        note_id = note.get('id', '')
        url = f'https://openreview.net/forum?id={note_id}' if note_id else None

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            source='openreview',
            published_date=pub_date,
            url=url,
            venue=venue_name,
            domains=_infer_domains(title, abstract),
            code_url=_extract_github(abstract),
        )
