from __future__ import annotations
from datetime import date, datetime
from .models import Paper
from .scoring import importance_score, is_recent

README_START = '<!-- PAPERS_START -->'
README_END = '<!-- PAPERS_END -->'


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


def _fmt_authors(paper: Paper) -> str:
    if not paper.authors:
        return '—'
    first = paper.authors[0].split()[-1] if paper.authors else ''  # last name of first author
    suffix = ' et al.' if len(paper.authors) > 1 else ''
    return first + suffix


def render_markdown(papers: list[Paper], recent_days: int = 90, top_n: int = 20) -> str:
    scored = [(p, importance_score(p)) for p in papers]
    scored.sort(key=lambda x: -x[1])

    seminal = [(p, s) for p, s in scored if (p.citation_count or 0) >= 5][:top_n]
    recent_cutoff = recent_days
    recent = [(p, s) for p, s in scored if is_recent(p, recent_cutoff)][:top_n]

    lines: list[str] = [
        README_START,
        '',
        f'*Auto-updated {datetime.now().strftime("%Y-%m-%d")} · '
        f'Sources: arXiv · Semantic Scholar · PubMed · bioRxiv · chemRxiv*',
        '',
    ]

    # --- Seminal papers ---
    lines += [
        f'### Top Papers (citation-ranked)',
        '',
        _md_row('Title', 'First Author', 'Year', 'Venue', 'Domain', 'Citations'),
        _md_row('---', '---', ':---:', '---', '---', '---:'),
    ]
    for paper, _ in seminal:
        year = str(paper.published_date.year) if paper.published_date else '?'
        venue = (paper.venue or '').split('(')[0].strip()[:30] or '—'
        domains = ', '.join(paper.domains) if paper.domains else '—'
        lines.append(_md_row(
            _title_link(paper),
            _fmt_authors(paper),
            year,
            venue,
            domains,
            _fmt_citations(paper.citation_count),
        ))

    lines += ['']

    # --- Recent highlights ---
    lines += [
        f'### Recent Highlights (last {recent_cutoff} days)',
        '',
        _md_row('Title', 'First Author', 'Date', 'Source', 'Domain', 'Citations'),
        _md_row('---', '---', ':---:', '---', '---', '---:'),
    ]
    for paper, _ in recent:
        d = str(paper.published_date) if paper.published_date else '?'
        domains = ', '.join(paper.domains) if paper.domains else '—'
        lines.append(_md_row(
            _title_link(paper),
            _fmt_authors(paper),
            d,
            paper.source,
            domains,
            _fmt_citations(paper.citation_count),
        ))

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
