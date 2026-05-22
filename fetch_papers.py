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
        'abstract': p.abstract[:400] if p.abstract else '',
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


def merge(existing: list, new_papers: list) -> list:
    """Merge new papers into existing, dedup by normalised title."""
    def key(p):
        return re.sub(r'\W+', ' ', p.title.lower()).strip()

    seen = {key(p) for p in existing}
    added = 0
    for p in new_papers:
        k = key(p)
        if k not in seen:
            seen.add(k)
            existing.append(p)
            added += 1

    # Prefer higher citation count when same paper comes from multiple sources
    title_map: dict[str, object] = {}
    for p in existing:
        k = key(p)
        if k not in title_map:
            title_map[k] = p
        else:
            prev = title_map[k]
            if (p.citation_count or 0) > (prev.citation_count or 0):
                title_map[k] = p

    logger.info(f"Merged {added} new papers; total unique: {len(title_map)}")
    return list(title_map.values())


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
@click.option('--source', multiple=True, metavar='NAME',
              help='Restrict fetch to: arxiv, semantic_scholar, pubmed, biorxiv (repeatable)')
@click.option('-v', '--verbose', is_flag=True)
def main(days, top_n, config, output, update_readme, fetch_seminal, no_fetch, source, verbose):
    """Fetch, rank and render agentic-AI-for-science papers to markdown."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = yaml.safe_load(open(config))
    sources_cfg = cfg.get('sources', {})
    active = set(source) if source else {'arxiv', 'semantic_scholar', 'pubmed', 'biorxiv'}

    cached = load_cache()

    if not no_fetch:
        from src.fetchers.arxiv_fetcher import ArxivFetcher
        from src.fetchers.semantic_scholar import SemanticScholarFetcher
        from src.fetchers.pubmed_fetcher import PubmedFetcher
        from src.fetchers.biorxiv_fetcher import BiorxivFetcher

        until_date = date.today()
        since_date = until_date - timedelta(days=days)
        fetcher_map = [
            ('arxiv',            'arXiv',           ArxivFetcher,           sources_cfg.get('arxiv', {})),
            ('semantic_scholar', 'Semantic Scholar', SemanticScholarFetcher, sources_cfg.get('semantic_scholar', {})),
            ('pubmed',           'PubMed',           PubmedFetcher,          sources_cfg.get('pubmed', {})),
            ('biorxiv',          'bioRxiv/chemRxiv', BiorxivFetcher,         sources_cfg.get('biorxiv', {})),
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
