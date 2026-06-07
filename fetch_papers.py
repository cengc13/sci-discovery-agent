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
        'paper_type': p.paper_type,
        'llm_on_topic': p.llm_on_topic,
        'venue_llm': p.venue_llm,
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
        paper_type=d.get('paper_type'),
        llm_on_topic=d.get('llm_on_topic'),
        venue_llm=d.get('venue_llm'),
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


_CODE_URL_BLOCKLIST = {
    # arXiv's own "report missing HTML" link — appears on every HTML-unavailable page
    'github.com/arxiv/html_feedback',
}


def _is_blocked_code_url(url: str) -> bool:
    return any(b in url.lower() for b in _CODE_URL_BLOCKLIST)


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
            url = m.group(0).rstrip('.,;:)')
            if not _is_blocked_code_url(url):
                p.code_url = url
                count += 1
    return count


def _merge_paper_metadata(dst, src) -> None:
    """Copy missing metadata from src into dst and keep the strongest signals."""
    if (src.citation_count or 0) > (dst.citation_count or 0):
        dst.citation_count = src.citation_count
    if not dst.venue and src.venue:
        dst.venue = src.venue
    if not dst.url and src.url:
        dst.url = src.url
    if not dst.code_url and src.code_url:
        dst.code_url = src.code_url
    if dst.paper_type is None and src.paper_type is not None:
        dst.paper_type = src.paper_type
    if dst.llm_on_topic is None and src.llm_on_topic is not None:
        dst.llm_on_topic = src.llm_on_topic
    if not dst.venue_llm and src.venue_llm:
        dst.venue_llm = src.venue_llm
    if src.domains:
        dst.domains = list(dict.fromkeys(dst.domains + src.domains))
    if not dst.abstract and src.abstract:
        dst.abstract = src.abstract
    if not dst.authors and src.authors:
        dst.authors = src.authors


_STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'or', 'with',
    'via', 'from', 'by', 'at', 'as', 'is', 'are', 'we', 'our', 'this',
    'that', 'large', 'using', 'based', 'new', 'approach', 'method', 'paper',
    'model', 'models', 'learning', 'deep', 'language', 'toward', 'towards',
}

_ACRONYM_PAT  = re.compile(r'\b([A-Z][A-Z0-9]{1,7})\b')
_CAMEL_PAT    = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b')   # SciToolAgent, MatClaw
_HYBRID_PAT   = re.compile(r'\b([A-Z][a-z]{1,6}[A-Z]{2,8})\b')        # TopoMAS, ChemROT


def _extract_tool_name(title: str, abstract: str) -> str | None:
    # Quoted name: “DIVE”, “ChemAgent”
    quoted = re.search(u'[“”‘’\'”]([A-Za-z][A-Za-z0-9\\-]{1,15})[“”‘’\'”]', title)
    if quoted:
        return quoted.group(1)
    short_title = title.split(':')[0]
    # All-caps acronym: QUASAR, ARES
    acr = _ACRONYM_PAT.findall(short_title)
    if acr:
        return acr[0]
    # CamelCase: SciToolAgent, MatClaw
    cc = _CAMEL_PAT.findall(short_title)
    if cc:
        return cc[0]
    # Hybrid CamelCase+caps: TopoMAS, ChemROT
    hy = _HYBRID_PAT.findall(short_title)
    if hy:
        return hy[0]
    return None


def _title_keywords(title: str) -> list[str]:
    words = re.sub(r'[^a-z0-9 ]', ' ', title.lower()).split()
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS][:6]


def _enrich_code_urls_github(papers: list, github_token: str | None = None,
                              delay: float = 6.5,
                              llm_api_key: str | None = None,
                              llm_model: str = 'gpt-4o-mini') -> int:
    """Search GitHub for repos matching tool name + paper keywords.

    Only run on papers already selected for display (~50 papers).
    Uses unauthenticated search (10 req/min) by default; pass github_token
    for 30 req/min. Delay default ~6.5s keeps well under unauthenticated limit.
    When llm_api_key is provided, candidate matches are verified by LLM before
    being accepted, eliminating false positives from keyword overlap.
    """
    import time
    import random
    import requests

    candidates = [p for p in papers if not p.code_url]
    logger.info(f"GitHub code search: {len(candidates)} candidates")

    headers = {'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28'}
    if github_token:
        headers['Authorization'] = f'Bearer {github_token}'

    count = 0
    for p in candidates:
        tool = _extract_tool_name(p.title, p.abstract or '')
        if not tool:
            continue

        keywords = _title_keywords(p.title)
        query = f'{tool} in:name,description {" ".join(keywords[:3])}'

        try:
            r = requests.get('https://api.github.com/search/repositories',
                             params={'q': query, 'sort': 'updated', 'per_page': 30},
                             headers=headers, timeout=10)
            if r.status_code == 403 or r.status_code == 429:
                logger.warning('GitHub search rate-limited; stopping early')
                break
            r.raise_for_status()
            items = r.json().get('items', [])
        except Exception as e:
            logger.debug(f"GitHub search failed for '{query}': {e}")
            time.sleep(delay)
            continue

        paper_kws = set(_title_keywords(p.title))
        best_score, best_repo = 0, None
        for repo in items:
            repo_name = repo.get('name', '').lower()
            repo_text = ' '.join(filter(None, [
                repo_name, repo.get('description') or '',
                ' '.join(repo.get('topics') or []),
            ])).lower()
            exact_name = repo_name == tool.lower()
            tool_hit   = tool.lower() in repo_text
            kw_hits    = sum(1 for kw in paper_kws if kw in repo_text)
            # Score: exact name match = 4, tool anywhere = 2, each keyword = 1
            score = (4 if exact_name else 2 if tool_hit else 0) + kw_hits
            if score > best_score:
                best_score, best_repo = score, repo
        # Accept: exact name + any keyword (score≥5), or strong keyword overlap (score≥4)
        if best_repo and best_score >= 4:
            if llm_api_key:
                from src.llm_enricher import verify_code_url
                if not verify_code_url(llm_api_key, llm_model,
                                       p.title, p.abstract or '', best_repo):
                    logger.info(f"  LLM rejected match (score={best_score}): "
                                f"{p.title[:45]} → {best_repo['html_url']}")
                    jitter = delay * (1 + random.uniform(-0.15, 0.15))
                    time.sleep(jitter)
                    continue
            p.code_url = best_repo['html_url']
            count += 1
            logger.info(f"  GitHub match (score={best_score}): {p.title[:50]} → {p.code_url}")

        jitter = delay * (1 + random.uniform(-0.15, 0.15))
        time.sleep(jitter)

    return count


def _enrich_code_urls_arxiv(papers: list, delay: float = 2.0) -> int:
    """Fetch arXiv HTML for papers missing code_url and search for GitHub links.

    Targets the Data/Code Availability sections that aren't in the abstract.
    Only attempts papers with an arxiv_id; skips if HTML version doesn't exist.
    """
    import re
    import time
    import random
    import requests

    gh_pat = re.compile(r'https?://github\.com/[\w\-][^/\s\)>\]"<]+/[\w\-][^\s\)>\]"<]*')
    candidates = [p for p in papers if p.arxiv_id and not p.code_url]
    logger.info(f"arXiv HTML code enrichment: {len(candidates)} candidates")
    count = 0

    for p in candidates:
        arxiv_id = re.sub(r'v\d+$', '', p.arxiv_id)
        url = f'https://arxiv.org/html/{arxiv_id}'
        try:
            r = requests.get(url, timeout=15,
                             headers={'User-Agent': 'paper-fetcher/1.0 (code-enrichment)'})
            if r.status_code == 404:
                continue
            if r.status_code == 429:
                logger.warning("arXiv HTML rate-limited; stopping code enrichment early")
                break
            r.raise_for_status()
            m = gh_pat.search(r.text)
            if m:
                url = m.group(0).rstrip('.,;:)"\'')
                if not _is_blocked_code_url(url):
                    p.code_url = url
                    count += 1
                    logger.info(f"  found code: {p.title[:60]} → {p.code_url}")
        except Exception as e:
            logger.debug(f"arXiv HTML fetch failed for {arxiv_id}: {e}")
        jitter = delay * (1 + random.uniform(-0.2, 0.2))
        time.sleep(jitter)

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
            _merge_paper_metadata(existing_paper, p)
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
            _merge_paper_metadata(prev, p)

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
@click.option('--fetch-seminal/--no-fetch-seminal', default=True, show_default=True,
              help='Also query Semantic Scholar without a date filter to surface high-citation papers')
@click.option('--no-fetch', is_flag=True,
              help='Skip fetching; just re-render from the cached data/papers.json')
@click.option('--enrich/--no-enrich', default=True, show_default=True,
              help='After fetching, batch-enrich citation counts via Semantic Scholar')
@click.option('--enrich-code/--no-enrich-code', default=False, show_default=True,
              help='Fetch arXiv HTML to find GitHub links in Data/Code Availability sections')
@click.option('--source', multiple=True, metavar='NAME',
              help='Restrict fetch to: arxiv, semantic_scholar, pubmed, chemrxiv/biorxiv (repeatable)')
@click.option('--enrich-llm/--no-enrich-llm', default=False, show_default=True,
              help='Use Gemini Flash to classify paper type and verify on-topic status')
@click.option('--llm-force', is_flag=True,
              help='Re-classify all papers, even those already classified by LLM')
@click.option('--llm-max-papers', default=0, show_default=True,
              help='Cap papers classified per run (0 = no limit; safety guard for large backlogs)')
@click.option('-v', '--verbose', is_flag=True)
def main(days, top_n, config, output, update_readme, fetch_seminal, no_fetch, enrich, enrich_code, source, enrich_llm, llm_force, llm_max_papers, verbose):
    """Fetch, rank and render agentic-AI-for-science papers to markdown."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = yaml.safe_load(open(config))
    sources_cfg = cfg.get('sources', {})
    source_aliases = {'biorxiv': 'chemrxiv', 'chemrxiv': 'chemrxiv'}
    active = {source_aliases.get(s, s) for s in source} if source else {'arxiv', 'semantic_scholar', 'chemrxiv', 'openreview'}

    cached = load_cache()

    if not no_fetch:
        from src.fetchers.arxiv_fetcher import ArxivFetcher
        from src.fetchers.semantic_scholar import SemanticScholarFetcher
        from src.fetchers.biorxiv_fetcher import BiorxivFetcher
        from src.fetchers.openreview_fetcher import OpenReviewFetcher

        until_date = date.today()
        since_date = until_date - timedelta(days=days)
        chemrxiv_cfg = sources_cfg.get('chemrxiv', sources_cfg.get('biorxiv', {}))
        fetcher_map = [
            ('arxiv',            'arXiv',      ArxivFetcher,           sources_cfg.get('arxiv', {})),
            ('semantic_scholar', 'Semantic Scholar', SemanticScholarFetcher, sources_cfg.get('semantic_scholar', {})),
            ('chemrxiv',         'chemRxiv',   BiorxivFetcher,         chemrxiv_cfg),
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

    if enrich_llm:
        import os
        llm_cfg = cfg.get('llm', {})
        openai_key = os.environ.get('OPENAI_API_KEY') or llm_cfg.get('openai_api_key', '')
        if not openai_key:
            click.echo('[LLM] ERROR: no OpenAI API key — set OPENAI_API_KEY or llm.openai_api_key in config', err=True)
        else:
            from src.llm_enricher import enrich_papers_llm, DEFAULT_MODEL
            model = llm_cfg.get('model', DEFAULT_MODEL)
            click.echo(f'[LLM] classifying papers with {model}...')
            n = enrich_papers_llm(cached, api_key=openai_key, model=model,
                                   force=llm_force, max_papers=llm_max_papers)
            click.echo(f'[LLM] classified {n} papers')
            save_cache(cached)

    if enrich_code:
        from src.renderer import get_display_papers
        display = get_display_papers(cached, recent_days=days, top_n=top_n)
        display_no_code = [p for p in display if not p.code_url]
        click.echo(f'[code enrichment] {len(display_no_code)} display papers missing code URL')

        click.echo('[arXiv HTML] checking Data/Code Availability sections...')
        n = _enrich_code_urls_arxiv(display_no_code)
        click.echo(f'[arXiv HTML] found {n} new code links')

        display_no_code = [p for p in display if not p.code_url]
        if display_no_code:
            import os
            gh_token = os.environ.get('GITHUB_TOKEN')
            llm_cfg = cfg.get('llm', {})
            openai_key = os.environ.get('OPENAI_API_KEY') or llm_cfg.get('openai_api_key', '')
            from src.llm_enricher import DEFAULT_MODEL
            llm_model = llm_cfg.get('model', DEFAULT_MODEL)
            click.echo(f'[GitHub search] searching {len(display_no_code)} remaining papers'
                       + (' (authenticated)' if gh_token else ' (unauthenticated, ~6s/paper)')
                       + (' + LLM verify' if openai_key else ''))
            n2 = _enrich_code_urls_github(display_no_code, github_token=gh_token,
                                           llm_api_key=openai_key or None,
                                           llm_model=llm_model)
            click.echo(f'[GitHub search] found {n2} new code links')

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
