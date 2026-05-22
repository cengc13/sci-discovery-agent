from __future__ import annotations
from datetime import date, datetime
from .models import Paper
from .scoring import importance_score, is_recent, is_on_topic

README_START = '<!-- PAPERS_START -->'
README_END = '<!-- PAPERS_END -->'

# Title / venue keywords that indicate a review or survey paper
_REVIEW_TITLE_SIGNALS = [
    'review', 'survey', 'perspective', 'overview', 'roadmap',
    'tutorial', 'meta-analysis', 'systematic', 'a comprehensive',
    'progress in', 'advances in', 'trends in',
]
_REVIEW_VENUE_SIGNALS = [
    'reviews', 'review journal', 'annual review', 'perspectives',
]


def _is_review(paper: Paper) -> bool:
    title_lower = paper.title.lower()
    if any(s in title_lower for s in _REVIEW_TITLE_SIGNALS):
        return True
    venue_lower = (paper.venue or '').lower()
    return any(v in venue_lower for v in _REVIEW_VENUE_SIGNALS)


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


def _top_table(papers: list[tuple], header_cols: list[str], row_fn) -> list[str]:
    if not papers:
        return []
    sep = ['---'] * len(header_cols)
    sep[-1] = '---:'  # right-align citations
    lines = [_md_row(*header_cols), _md_row(*sep)]
    for paper, _ in papers:
        lines.append(row_fn(paper))
    return lines


def render_markdown(papers: list[Paper], recent_days: int = 90, top_n: int = 20) -> str:
    on_topic = [p for p in papers if is_on_topic(p)]
    scored = [(p, importance_score(p)) for p in on_topic]
    scored.sort(key=lambda x: -x[1])

    seminal = [(p, s) for p, s in scored if (p.citation_count or 0) >= 5][:top_n]
    recent = [(p, s) for p, s in scored if is_recent(p, recent_days)][:top_n]

    def top_row(paper: Paper) -> str:
        year = str(paper.published_date.year) if paper.published_date else '?'
        venue = (paper.venue or '').split('(')[0].strip()[:30] or '—'
        domains = ', '.join(paper.domains) if paper.domains else '—'
        return _md_row(_title_link(paper), year, venue, domains,
                       _fmt_citations(paper.citation_count))

    def recent_row(paper: Paper) -> str:
        d = str(paper.published_date) if paper.published_date else '?'
        domains = ', '.join(paper.domains) if paper.domains else '—'
        return _md_row(_title_link(paper), d, paper.source, domains,
                       _fmt_citations(paper.citation_count))

    TOP_COLS = ['Title', 'Year', 'Venue', 'Domain', 'Citations']
    RECENT_COLS = ['Title', 'Date', 'Source', 'Domain', 'Citations']

    lines: list[str] = [
        README_START,
        '',
        f'*Auto-updated {datetime.now().strftime("%Y-%m-%d")} · '
        f'Sources: arXiv · Semantic Scholar · PubMed · bioRxiv · chemRxiv*',
        '',
    ]

    # ── Top Papers ────────────────────────────────────────────────────────────
    sem_articles = [(p, s) for p, s in seminal if not _is_review(p)]
    sem_reviews  = [(p, s) for p, s in seminal if     _is_review(p)]

    lines += ['### Top Papers (citation-ranked)', '']
    if sem_articles:
        lines += ['#### Articles', '']
        lines += _top_table(sem_articles, TOP_COLS, top_row)
    if sem_reviews:
        lines += ['', '#### Reviews & Surveys', '']
        lines += _top_table(sem_reviews, TOP_COLS, top_row)

    lines += ['']

    # ── Recent Highlights ─────────────────────────────────────────────────────
    rec_articles = [(p, s) for p, s in recent if not _is_review(p)]
    rec_reviews  = [(p, s) for p, s in recent if     _is_review(p)]

    lines += [f'### Recent Highlights (last {recent_days} days)', '']
    if rec_articles:
        lines += ['#### Articles', '']
        lines += _top_table(rec_articles, RECENT_COLS, recent_row)
    if rec_reviews:
        lines += ['', '#### Reviews & Surveys', '']
        lines += _top_table(rec_reviews, RECENT_COLS, recent_row)

    lines += ['', README_END]
    return '\n'.join(lines)


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
