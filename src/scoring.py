from __future__ import annotations
from datetime import date
from .models import Paper

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


def is_recent(paper: Paper, days: int) -> bool:
    if not paper.published_date:
        return False
    return (date.today() - paper.published_date).days <= days
