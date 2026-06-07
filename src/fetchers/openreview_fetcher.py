from __future__ import annotations
import json
import logging
import random
import re
import requests
import time
from datetime import date
from pathlib import Path
from ..models import Paper
from ..scoring import is_on_topic

logger = logging.getLogger(__name__)

BASE_URL = 'https://api2.openreview.net'
STATE_FILE = Path('data/openreview_state.json')

_GH_PAT = re.compile(r'https?://github\.com/[\w\-][^/\s\)>\]]+/[\w\-][^\s\)>\]]*')

_DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'battery', 'perovskite'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'biolog', 'biochem', 'enzyme', 'genomic'],
}

# Scan the current and recent conference seasons so new years do not go stale.
_CONFERENCES = ('ICLR', 'NeurIPS', 'ICML')


def _target_venues() -> list[tuple[str, str, int]]:
    # Start from prior year — current-year conferences may not yet have published.
    current_year = date.today().year
    venues: list[tuple[str, str, int]] = []
    for year in range(current_year - 1, current_year - 3, -1):
        for conf in _CONFERENCES:
            venues.append((f'{conf}.cc/{year}/Conference', f'{conf} {year}', year))
    return venues


def _jitter(base: float, frac: float = 0.2) -> float:
    """Return base ± frac of base (uniform), always positive."""
    return max(0.0, base * (1 + random.uniform(-frac, frac)))


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + (abstract or '')).lower()
    domains = [d for d, kws in _DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['AI/science']


def _extract_github(abstract: str) -> str | None:
    if not abstract:
        return None
    m = _GH_PAT.search(abstract)
    return m.group(0).rstrip('.,;:)') if m else None


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


class OpenReviewFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.max_per_venue = config.get('max_per_venue', 4000)
        self.delay = config.get('delay', 0.5)          # between paginated requests
        self.venue_delay = config.get('venue_delay', 3.0)  # between conference venues
        self.force_rescan = config.get('force_rescan', False)

    def fetch(self, since: date | None = None, until: date | None = None) -> list[Paper]:
        state = _load_state()
        papers: list[Paper] = []
        seen: set[str] = set()

        for i, (venueid, venue_name, year) in enumerate(_target_venues()):
            if not self.force_rescan and state.get(venueid, {}).get('completed'):
                logger.info(f'OpenReview: {venue_name} already scanned — skipping '
                            f'(found {state[venueid].get("papers_found", "?")} papers on '
                            f'{state[venueid].get("completed_at", "?")})')
                continue

            if i > 0:
                time.sleep(_jitter(self.venue_delay))

            logger.info(f'OpenReview: scanning {venue_name}...')
            count, exhausted = self._scan_venue(venueid, venue_name, year, seen, papers)
            logger.info(f'OpenReview: {venue_name} → {count} relevant papers')

            if exhausted:
                state[venueid] = {
                    'completed': True,
                    'papers_found': count,
                    'completed_at': str(date.today()),
                }
                _save_state(state)

        logger.info(f'OpenReview: {len(papers)} total relevant papers')
        return papers

    def _scan_venue(self, venueid: str, venue_name: str, year: int,
                    seen: set[str], papers: list[Paper]) -> tuple[int, bool]:
        """Returns (papers_found, exhausted). exhausted=True means we reached natural end of venue."""
        found = 0
        offset = 0
        checked = 0

        while checked < self.max_per_venue:
            batch = self._fetch_batch(venueid, limit=100, offset=offset)
            if batch is None:
                # Fetch failed (rate-limited / network error) — don't mark complete
                return found, False
            if len(batch) == 0:
                # Legitimate end of pagination
                return found, True
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
                # Last partial page → venue fully scanned
                return found, True
            time.sleep(_jitter(self.delay))

        # Hit max_per_venue cap — mark complete so we don't rescan every run
        return found, True

    def _fetch_batch(self, venueid: str, limit: int, offset: int) -> list[dict] | None:
        """Return notes list, empty list on legitimate end-of-results, or None on fetch failure."""
        backoff = 10.0
        for attempt in range(4):
            try:
                r = requests.get(
                    f'{BASE_URL}/notes',
                    params={'content.venueid': venueid, 'limit': limit, 'offset': offset},
                    timeout=30,
                )
                if r.status_code == 429:
                    retry_after = int(r.headers.get('Retry-After', backoff))
                    wait = _jitter(max(retry_after, backoff))
                    logger.info(f'OpenReview rate-limited; waiting {wait:.1f}s')
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    continue
                if r.status_code in (404, 410):
                    # Venue doesn't exist — cache as empty so it isn't retried every run
                    return []
                r.raise_for_status()
                return r.json().get('notes', [])
            except requests.HTTPError as e:
                logger.warning(f'OpenReview batch failed (offset={offset}): {e}')
                return None
            except Exception as e:
                logger.warning(f'OpenReview batch error (offset={offset}): {e}')
                if attempt < 3:
                    wait = _jitter(backoff)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
        # All retries exhausted (rate-limited or network error) — signal failure
        logger.warning(f'OpenReview: gave up on {venueid} at offset={offset} after 4 attempts')
        return None

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
