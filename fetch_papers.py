#!/usr/bin/env python3
"""Fetch and rank agentic-AI-for-science papers; emit README-ready markdown."""
from __future__ import annotations
import json
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import click
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = Path('data/papers.json')


# ── serialisation helpers ────────────────────────────────────────────────────

def _paper_to_dict(p) -> dict:
    return {
        'title': p.title,
        'authors': p.authors,
        'abstract': p.abstract[:800] if p.abstract else '',
        'source': p.source,
        'published_date': str(p.published_date) if p.published_date else None,
        'doi': p.doi,
        'arxiv_id': p.arxiv_id,
        'pubmed_id': p.pubmed_id,
        'biorxiv_id': p.biorxiv_id,
        'url': p.url,
        'venue': p.venue,
        'citation_count': p.citation_count,
        'domains': p.domains,
        'code_url': p.code_url,
    }


def _dict_to_paper(d: dict):
    from src.models import Paper
    from datetime import date as dt
    pub = dt.fromisoformat(d['published_date']) if d.get('published_date') else None
    return Paper(
        title=d['title'],
        authors=d.get('authors', []),
        abstract=d.get('abstract', ''),
        source=d.get('source', ''),
        published_date=pub,
        doi=d.get('doi'),
        arxiv_id=d.get('arxiv_id'),
        pubmed_id=d.get('pubmed_id'),
        biorxiv_id=d.get('biorxiv_id'),
        url=d.get('url'),
        venue=d.get('venue'),
        citation_count=d.get('citation_count'),
        domains=d.get('domains', []),
        code_url=d.get('code_url'),
    )


def load_cache() -> list:
    if DATA_FILE.exists():
        try:
            return [_dict_to_paper(d) for d in json.loads(DATA_FILE.read_text())]
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    return []


def save_cache(papers: list):
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps([_paper_to_dict(p) for p in papers], indent=2))


def _extract_code_urls(papers: list) -> int:
    """Backfill code_url from GitHub links found in paper abstracts."""
    import re
    gh_pat = re.compile(r'https?://github\.com/[\w\-][^/\s\)>\]]+/[\w\-][^\s\)>\]]*')
    count = 0
    for p in papers:
        if p.code_url:
            continue
        m = gh_pat.search(p.abstract or '')
        if m:
            p.code_url = m.group(0).rstrip('.,;:)')
            count += 1
    return count


def merge(existing: list, new_papers: list) -> list:
    """Merge new papers into existing, dedup by external ID then normalised title.

    When a new paper matches an existing one by arxiv_id/doi/pubmed_id,
    we update the existing entry's citation count rather than adding a duplicate.
    Off-topic papers are filtered before merging.
    """
    from src.scoring import is_on_topic
    new_papers = [p for p in new_papers if is_on_topic(p)]

    def title_key(p):
        t = p.title.lower()
        # Strip short codename prefix ("ChemCrow: Augmenting..." → "augmenting...")
        # so preprint and published titles with the same subtitle are treated as one paper.
        if ': ' in t:
            prefix, suffix = t.split(': ', 1)
            if len(prefix.split()) <= 2:
                t = suffix
        return re.sub(r'\W+', ' ', t).strip()

    def norm_arxiv(arxiv_id: str) -> str:
        # Strip version suffix so '2302.07842v1' and '2302.07842' match
        return re.sub(r'v\d+$', '', arxiv_id.lower())

    # Build lookup maps for external IDs in the existing list
    arxiv_map: dict[str, object] = {}
    doi_map: dict[str, object] = {}
    pubmed_map: dict[str, object] = {}
    for p in existing:
        if p.arxiv_id:
            arxiv_map[norm_arxiv(p.arxiv_id)] = p
        if p.doi:
            doi_map[p.doi.lower()] = p
        if p.pubmed_id:
            pubmed_map[p.pubmed_id] = p

    title_map: dict[str, object] = {title_key(p): p for p in existing}
    added = 0

    for p in new_papers:
        # Try external-ID match first — enrich citation count if higher
        existing_paper = None
        if p.arxiv_id and norm_arxiv(p.arxiv_id) in arxiv_map:
            existing_paper = arxiv_map[norm_arxiv(p.arxiv_id)]
        elif p.doi and p.doi.lower() in doi_map:
            existing_paper = doi_map[p.doi.lower()]
        elif p.pubmed_id and p.pubmed_id in pubmed_map:
            existing_paper = pubmed_map[p.pubmed_id]

        if existing_paper is not None:
            if (p.citation_count or 0) > (existing_paper.citation_count or 0):
                existing_paper.citation_count = p.citation_count
            if not existing_paper.venue and p.venue:
                existing_paper.venue = p.venue
            if not existing_paper.url and p.url:
                existing_paper.url = p.url
            continue

        # Fall back to title-based dedup
        k = title_key(p)
        if k not in title_map:
            title_map[k] = p
            existing.append(p)
            if p.arxiv_id:
                arxiv_map[norm_arxiv(p.arxiv_id)] = p
            if p.doi:
                doi_map[p.doi.lower()] = p
            if p.pubmed_id:
                pubmed_map[p.pubmed_id] = p
            added += 1
        else:
            # Title match: prefer higher citation count, fill missing fields
            prev = title_map[k]
            if (p.citation_count or 0) > (prev.citation_count or 0):
                prev.citation_count = p.citation_count
            if not prev.venue and p.venue:
                prev.venue = p.venue
            if not prev.url and p.url:
                prev.url = p.url

    logger.info(f"Merged {added} new; {len(existing)} unique total")
    return existing


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option('--days', default=90, show_default=True,
              help='Window for "recent highlights" table and incremental fetch')
@click.option('--top-n', default=20, show_default=True, help='Max rows per table')
@click.option('--config', default='config.yaml', show_default=True)
@click.option('--output', default='PAPERS.md', show_default=True,
              help='Output file (use - for stdout)')
@click.option('--update-readme', is_flag=True,
              help='Inject the block into README.md between <!-- PAPERS_START/END --> markers')
@click.option('--fetch-seminal', is_flag=True, default=True, show_default=True,
              help='Also query Semantic Scholar without a date filter to surface high-citation papers')
@click.option('--no-fetch', is_flag=True,
              help='Skip fetching; just re-render from the cached data/papers.json')
@click.option('--enrich/--no-enrich', default=True, show_default=True,
              help='After fetching, batch-enrich citation counts via Semantic Scholar')
@click.option('--source', multiple=True, metavar='NAME',
              help='Restrict fetch to: arxiv, semantic_scholar, pubmed, biorxiv (repeatable)')
@click.option('-v', '--verbose', is_flag=True)
def main(days, top_n, config, output, update_readme, fetch_seminal, no_fetch, enrich, source, verbose):
    """Fetch, rank and render agentic-AI-for-science papers to markdown."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = yaml.safe_load(open(config))
    sources_cfg = cfg.get('sources', {})
    active = set(source) if source else {'arxiv', 'semantic_scholar', 'chemrxiv', 'openreview'}

    cached = load_cache()

    if not no_fetch:
        from src.fetchers.arxiv_fetcher import ArxivFetcher
        from src.fetchers.semantic_scholar import SemanticScholarFetcher
        from src.fetchers.biorxiv_fetcher import BiorxivFetcher
        from src.fetchers.openreview_fetcher import OpenReviewFetcher

        until_date = date.today()
        since_date = until_date - timedelta(days=days)
        fetcher_map = [
            ('arxiv',            'arXiv',      ArxivFetcher,           sources_cfg.get('arxiv', {})),
            ('semantic_scholar', 'Semantic Scholar', SemanticScholarFetcher, sources_cfg.get('semantic_scholar', {})),
            ('chemrxiv',         'chemRxiv',   BiorxivFetcher,         sources_cfg.get('chemrxiv', {})),
            ('openreview',       'OpenReview', OpenReviewFetcher,      sources_cfg.get('openreview', {})),
        ]
        new_papers = []
        for key, label, Cls, src_cfg in fetcher_map:
            if key not in active or not src_cfg.get('enabled', True):
                continue
            click.echo(f'[{label}] fetching last {days} days...')
            try:
                fetcher = Cls(src_cfg)
                if key == 'semantic_scholar':
                    batch = fetcher.fetch(since=since_date, until=until_date)
                else:
                    batch = fetcher.fetch(since_date, until_date)
                click.echo(f'[{label}] {len(batch)} papers')
                new_papers.extend(batch)
            except Exception as e:
                click.echo(f'[{label}] ERROR: {e}', err=True)

        # Seminal fetch: S2 with no date filter, take whatever is highly cited
        if fetch_seminal and 'semantic_scholar' in active:
            import time as _time
            _time.sleep(1.1)  # gap between dated fetch and seminal fetch
            s2_cfg = sources_cfg.get('semantic_scholar', {})
            s2_cfg_seminal = {**s2_cfg, 'max_results': 200}
            click.echo('[Semantic Scholar] fetching seminal (no date filter)...')
            try:
                seminal_papers = SemanticScholarFetcher(s2_cfg_seminal).fetch()
                click.echo(f'[Semantic Scholar] {len(seminal_papers)} seminal candidates')
                new_papers.extend(seminal_papers)
            except Exception as e:
                click.echo(f'[Semantic Scholar seminal] ERROR: {e}', err=True)

        cached = merge(cached, new_papers)
        _extract_code_urls(cached)
        save_cache(cached)

    # Citation enrichment runs independently of --no-fetch
    if enrich:
        import time as _time
        from src.fetchers.semantic_scholar import SemanticScholarFetcher
        s2_cfg = sources_cfg.get('semantic_scholar', {})
        if s2_cfg.get('enabled', True):
            click.echo('[Semantic Scholar] enriching citation counts...')
            if not no_fetch and 'semantic_scholar' in active:
                _time.sleep(1.1)  # respect 1 req/s after prior S2 search calls
            enricher = SemanticScholarFetcher(s2_cfg)
            n = enricher.enrich_citations(cached)
            click.echo(f'[Semantic Scholar] updated {n} citation counts')
            save_cache(cached)

    from src.renderer import render_markdown, inject_into_readme
    md = render_markdown(cached, recent_days=days, top_n=top_n)

    if update_readme:
        inject_into_readme(md)
        click.echo('README.md updated.')
    elif output == '-':
        print(md)
    else:
        Path(output).write_text(md, encoding='utf-8')
        click.echo(f'Written to {output}')


if __name__ == '__main__':
    main()
