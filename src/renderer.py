from __future__ import annotations
from datetime import date, datetime
from .models import Paper
from .scoring import importance_score, is_recent, is_on_topic, is_published

README_START = '<!-- PAPERS_START -->'
README_END = '<!-- PAPERS_END -->'

_REVIEW_TITLE_SIGNALS = [
    'review', 'survey', 'perspective', 'overview', 'roadmap',
    'tutorial', 'meta-analysis', 'systematic', 'a comprehensive',
    'progress in', 'advances in', 'trends in',
    'benchmark', 'benchmarking', 'informatics', 'landscape',
    'state of the art', 'state-of-the-art',
]
_REVIEW_VENUE_SIGNALS = [
    'reviews', 'review journal', 'annual review', 'perspectives',
]

_VENUE_ABBREVS: dict[str, str] = {
    # Nature portfolio — specific sub-journals must appear before 'nature'
    'nature machine intelligence': 'Nat. Mach. Intell.',
    'nature communications': 'Nat. Commun.',
    'nature computational science': 'Nat. Comput. Sci.',
    'nature reviews materials': 'Nat. Rev. Mater.',
    'nature reviews chemistry': 'Nat. Rev. Chem.',
    'nature reviews drug discovery': 'Nat. Rev. Drug Discov.',
    'nature biotechnology': 'Nat. Biotechnol.',
    'nature chemical engineering': 'Nat. Chem. Eng.',
    'nature chemical biology': 'Nat. Chem. Biol.',
    'nature chemistry': 'Nat. Chem.',
    'nature catalysis': 'Nat. Catal.',
    'nature electronics': 'Nat. Electron.',
    'nature energy': 'Nat. Energy',
    'nature nanotechnology': 'Nat. Nanotechnol.',
    'nature materials': 'Nat. Mater.',
    'nature methods': 'Nat. Methods.',
    'nature physics': 'Nat. Phys.',
    'nature synthesis': 'Nat. Synth.',
    'nature': 'Nature',
    # Science family
    'science advances': 'Sci. Adv.',
    'science robotics': 'Sci. Robot.',
    'science translational medicine': 'Sci. Transl. Med.',
    'science': 'Science',
    # Cell Press — specific before 'cell'
    'cell chemical biology': 'Cell Chem. Biol.',
    'cell reports physical science': 'Cell Rep. Phys. Sci.',
    'cell reports methods': 'Cell Rep. Methods',
    'cell physical science': 'Cell Phys. Sci.',
    'cell systems': 'Cell Syst.',
    'cell': 'Cell',
    # ACS
    'acs central science': 'ACS Cent. Sci.',
    'acs measurement science au': 'ACS Meas. Sci. Au',
    'acs nano': 'ACS Nano',
    'journal of the american chemical society': 'JACS',
    'journal of chemical information and modeling': 'JCIM',
    'chemical reviews': 'Chem. Rev.',
    'chemical science': 'Chem. Sci.',
    # Other chemistry/materials
    'advanced materials': 'Adv. Mater.',
    'advances in materials': 'Adv. Mater.',
    'advanced science': 'Adv. Sci.',
    'angewandte chemie': 'Angew. Chem.',
    'the innovation': 'The Innovation',
    'proceedings of the national academy of sciences': 'PNAS',
    'pnas': 'PNAS',
    'matter': 'Matter',
    'joule': 'Joule',
    # IEEE
    'ieee transactions on automation science and engineering': 'IEEE TASE',
    'ieee transactions on neural networks and learning systems': 'IEEE TNNLS',
    'ieee transactions on pattern analysis and machine intelligence': 'IEEE TPAMI',
    # ML conferences
    'iclr': 'ICLR',
    'neurips': 'NeurIPS',
    'icml': 'ICML',
    # Preprints
    'arxiv.org': 'arXiv',
    'arxiv': 'arXiv',
    'biorxiv': 'bioRxiv',
    'chemrxiv': 'chemRxiv',
    # Other
    'frontiers in artificial intelligence': 'Front. Artif. Intell.',
}

_DOMAIN_PRIORITY = ['materials', 'chemistry', 'biology']
_DOMAIN_DISPLAY = {
    'materials': 'materials science',
    'chemistry': 'chemistry',
    'biology': 'biology',
    'AI/science': 'AI/science',
}


def _is_review(paper: Paper) -> bool:
    if paper.paper_type is not None:
        return paper.paper_type == 'review'
    title_lower = paper.title.lower()
    if any(s in title_lower for s in _REVIEW_TITLE_SIGNALS):
        return True
    venue_lower = (paper.venue or '').lower()
    return any(v in venue_lower for v in _REVIEW_VENUE_SIGNALS)


def _is_chem_or_materials(paper: Paper) -> bool:
    """Keep only papers with chemistry or materials as a domain."""
    return 'chemistry' in paper.domains or 'materials' in paper.domains


def _short_venue(paper_or_venue) -> str:
    if isinstance(paper_or_venue, str):
        venue_llm, venue_raw = None, paper_or_venue
    else:
        venue_llm = paper_or_venue.venue_llm
        venue_raw = paper_or_venue.venue or ''

    # Prefer LLM-corrected name; fall back to raw venue
    venue = venue_llm or venue_raw
    if not venue:
        return '—'
    v = venue.split('(')[0].strip()
    v_lower = v.lower()
    # Sort longest-first so 'nature chemical engineering' beats 'nature'
    for key in sorted(_VENUE_ABBREVS, key=len, reverse=True):
        if v_lower.startswith(key):
            return _VENUE_ABBREVS[key]
    # For LLM-provided names with no abbreviation, return as-is (already clean)
    fallback = venue_llm if venue_llm else v
    return (fallback[:22] + '…') if len(fallback) > 22 else fallback


def _primary_domain(paper: Paper) -> str:
    for d in _DOMAIN_PRIORITY:
        if d in paper.domains:
            return _DOMAIN_DISPLAY[d]
    if paper.domains:
        return _DOMAIN_DISPLAY.get(paper.domains[0], paper.domains[0])
    return '—'


def _domain_sort_key(paper: Paper) -> int:
    for i, d in enumerate(_DOMAIN_PRIORITY):
        if d in paper.domains:
            return i
    return len(_DOMAIN_PRIORITY)


def _md_row(*cells: str) -> str:
    return '| ' + ' | '.join(cells) + ' |'


def _title_link(paper: Paper) -> str:
    url = paper.primary_url
    title = paper.title.replace('|', '\\|')
    short = (title[:80] + '…') if len(title) > 80 else title
    return f'[{short}]({url})' if url else short


def _fmt_citations(n: int | None) -> str:
    if n is None:
        return '—'
    if n >= 1000:
        return f'{n / 1000:.1f}k'
    return str(n)


def _code_cell(paper: Paper) -> str:
    return f'[Code]({paper.code_url})' if paper.code_url else '-'


def _pub_year(paper: Paper) -> int:
    return paper.published_date.year if paper.published_date else 0


def _top_table_articles(papers: list[tuple]) -> list[str]:
    """Title | Year | Venue | Code."""
    if not papers:
        return []
    cols = ['Title', 'Year', 'Venue', 'Code']
    sep = ['---'] * len(cols)
    lines = [_md_row(*cols), _md_row(*sep)]
    for paper, _ in papers:
        year = str(paper.published_date.year) if paper.published_date else '?'
        lines.append(_md_row(_title_link(paper), year, _short_venue(paper), _code_cell(paper)))
    return lines


def _top_table_reviews(papers: list[tuple]) -> list[str]:
    """Title | Year | Venue | Citations."""
    if not papers:
        return []
    cols = ['Title', 'Year', 'Venue', 'Citations']
    sep = ['---', '---', '---', '---:']
    lines = [_md_row(*cols), _md_row(*sep)]
    for paper, _ in papers:
        year = str(paper.published_date.year) if paper.published_date else '?'
        lines.append(_md_row(
            _title_link(paper), year, _short_venue(paper),
            _fmt_citations(paper.citation_count),
        ))
    return lines


def _recent_table_articles(papers: list[tuple]) -> list[str]:
    """Title | Date | Source | Code."""
    if not papers:
        return []
    cols = ['Title', 'Date', 'Source', 'Code']
    sep = ['---'] * len(cols)
    lines = [_md_row(*cols), _md_row(*sep)]
    for paper, _ in papers:
        d = str(paper.published_date) if paper.published_date else '?'
        lines.append(_md_row(_title_link(paper), d, paper.source, _code_cell(paper)))
    return lines


def _recent_table_reviews(papers: list[tuple]) -> list[str]:
    """Title | Date | Source | Citations."""
    if not papers:
        return []
    cols = ['Title', 'Date', 'Source', 'Citations']
    sep = ['---', '---', '---', '---:']
    lines = [_md_row(*cols), _md_row(*sep)]
    for paper, _ in papers:
        d = str(paper.published_date) if paper.published_date else '?'
        lines.append(_md_row(
            _title_link(paper), d, paper.source,
            _fmt_citations(paper.citation_count),
        ))
    return lines


def _top6_reviews(reviews: list[tuple]) -> list[tuple]:
    """3 most cited + 3 most recent, deduped, max 6."""
    by_cited  = sorted(reviews, key=lambda x: -(x[0].citation_count or 0))[:3]
    by_recent = sorted(reviews, key=lambda x: -_pub_year(x[0]))[:3]
    seen = {id(p) for p, _ in by_cited}
    extra = [(p, s) for p, s in by_recent if id(p) not in seen]
    return by_cited + extra


def render_markdown(papers: list[Paper], recent_days: int = 90, top_n: int = 20) -> str:
    on_topic = [p for p in papers if is_on_topic(p) and _is_chem_or_materials(p)
                and p.llm_on_topic is not False]
    scored = [(p, importance_score(p)) for p in on_topic]

    seminal_all = [(p, s) for p, s in scored if (p.citation_count or 0) >= 5]
    recent_all  = [(p, s) for p, s in scored if is_recent(p, recent_days)]

    sem_articles_raw = [(p, s) for p, s in seminal_all if not _is_review(p) and is_published(p)]
    sem_reviews_raw  = [(p, s) for p, s in seminal_all if     _is_review(p)]
    rec_articles_raw = [(p, s) for p, s in recent_all  if not _is_review(p)]
    rec_reviews_raw  = [(p, s) for p, s in recent_all  if     _is_review(p)]

    # Articles: year desc → domain priority
    sem_articles = sorted(sem_articles_raw, key=lambda x: (-_pub_year(x[0]), _domain_sort_key(x[0])))[:top_n]
    # Reviews: 3 most cited + 3 most recent
    sem_reviews  = _top6_reviews(sem_reviews_raw)

    rec_articles = sorted(rec_articles_raw, key=lambda x: (-_pub_year(x[0]), _domain_sort_key(x[0])))[:top_n]
    rec_reviews  = sorted(rec_reviews_raw,  key=lambda x: (-_pub_year(x[0]), -(x[0].citation_count or 0)))[:6]

    lines: list[str] = [
        README_START,
        '',
        f'*Auto-updated {datetime.now().strftime("%Y-%m-%d")} · '
        f'Sources: arXiv · Semantic Scholar · chemRxiv · OpenReview (ICLR/NeurIPS/ICML)*',
        '',
    ]

    lines += ['### Top Papers (citation-ranked)', '']
    if sem_articles:
        lines += ['#### Articles', '']
        lines += _top_table_articles(sem_articles)
    if sem_reviews:
        lines += ['', '#### Reviews & Surveys', '']
        lines += _top_table_reviews(sem_reviews)

    lines += ['']

    lines += [f'### Recent Highlights (last {recent_days} days)', '']
    if rec_articles:
        lines += ['#### Articles', '']
        lines += _recent_table_articles(rec_articles)
    if rec_reviews:
        lines += ['', '#### Reviews & Surveys', '']
        lines += _recent_table_reviews(rec_reviews)

    lines += ['', README_END]
    return '\n'.join(lines)


def get_display_papers(papers: list[Paper], recent_days: int = 90, top_n: int = 20) -> list[Paper]:
    """Return the unique set of papers that would appear in the README tables."""
    on_topic = [p for p in papers if is_on_topic(p) and _is_chem_or_materials(p)
                and p.llm_on_topic is not False]
    scored = [(p, importance_score(p)) for p in on_topic]

    seminal_all = [(p, s) for p, s in scored if (p.citation_count or 0) >= 5]
    recent_all  = [(p, s) for p, s in scored if is_recent(p, recent_days)]

    sem_articles = sorted([(p, s) for p, s in seminal_all if not _is_review(p) and is_published(p)],
                          key=lambda x: (-_pub_year(x[0]), _domain_sort_key(x[0])))[:top_n]
    sem_reviews  = _top6_reviews([(p, s) for p, s in seminal_all if _is_review(p)])
    rec_articles = sorted([(p, s) for p, s in recent_all if not _is_review(p)],
                          key=lambda x: (-_pub_year(x[0]), _domain_sort_key(x[0])))[:top_n]
    rec_reviews  = sorted([(p, s) for p, s in recent_all if _is_review(p)],
                          key=lambda x: (-_pub_year(x[0]), -(x[0].citation_count or 0)))[:6]

    seen: set[int] = set()
    result: list[Paper] = []
    for p, _ in sem_articles + sem_reviews + rec_articles + rec_reviews:
        if id(p) not in seen:
            seen.add(id(p))
            result.append(p)
    return result


def inject_into_readme(md_block: str, readme_path: str = 'README.md'):
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        content = f'{README_START}\n{README_END}\n'

    if README_START in content and README_END in content:
        before = content[:content.index(README_START)]
        after = content[content.index(README_END) + len(README_END):]
        new_content = before + md_block + after
    else:
        new_content = content.rstrip() + '\n\n' + md_block + '\n'

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
