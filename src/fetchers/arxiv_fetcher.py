from __future__ import annotations
import logging
import re
from datetime import date
from ..models import Paper

_GH_PAT = re.compile(r'https?://github\.com/[\w\-][^/\s\)>\]]+/[\w\-][^\s\)>\]]*')

logger = logging.getLogger(__name__)

# Queries targeting title + abstract. Each broadens coverage from different angles.
SCIENCE_TERMS = '(ti:chemistry OR ti:materials OR ti:biology OR ti:"drug discovery" OR ti:protein OR ti:molecule OR ti:"scientific discovery" OR ti:synthesis)'

QUERIES = [
    # Explicit agent framing
    f'(ti:"AI agent" OR ti:"LLM agent" OR ti:"agentic AI" OR ti:"multi-agent") AND {SCIENCE_TERMS}',

    # Autonomous systems / self-driving
    'ti:"self-driving lab" OR ti:"self-driving laboratory" OR ti:"autonomous laboratory" OR ti:"robotic chemistry" OR ti:"autonomous chemical" OR ti:"autonomous experiment"',

    # Tool-augmented LLMs for science (e.g. ChemCrow)
    f'(ti:"tool-augmented" OR ti:"tool use" OR ti:"tool-using" OR ti:"chemistry tools") AND (ti:"language model" OR ti:LLM)',

    # LLM/autonomous + planning/workflow in science
    f'(ti:"large language model" OR ti:LLM) AND (ti:autonomous OR ti:planning OR ti:workflow OR ti:agent OR ti:tool) AND {SCIENCE_TERMS}',

    # Autonomous + scientific discovery tasks
    f'(ti:autonomous OR ti:automated) AND (ti:"scientific discovery" OR ti:"materials discovery" OR ti:"drug discovery" OR ti:"synthesis planning" OR ti:"chemical synthesis")',

    # Closed-loop experimental AI (self-driving labs framing)
    '(ti:"closed-loop" OR ti:"closed loop") AND (ti:chemistry OR ti:materials OR ti:synthesis OR ti:drug OR ti:experiment OR ti:"scientific")',
]

DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst', 'organic', 'reagent'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor', 'solid-state', 'battery', 'perovskite'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'genomic', 'biolog', 'biochem', 'enzyme', 'peptide'],
}


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + abstract).lower()
    domains = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['AI/science']


class ArxivFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.max_results = config.get('max_results', 150)

    def fetch(self, since: date, until: date) -> list[Paper]:
        try:
            import arxiv
        except ImportError:
            logger.error("arxiv package not installed. Run: pip install arxiv")
            return []

        client = arxiv.Client(page_size=100, delay_seconds=1, num_retries=3)
        seen_ids: set[str] = set()
        papers: list[Paper] = []

        for query in QUERIES:
            logger.info(f"arXiv: {query[:70]}...")
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            try:
                for result in client.results(search):
                    pub = result.published.date()
                    if pub < since or pub > until:
                        continue
                    arxiv_id = result.get_short_id()
                    if arxiv_id in seen_ids:
                        continue
                    seen_ids.add(arxiv_id)

                    comment = (result.comment or '').replace('\n', ' ')
                    gh_match = _GH_PAT.search(comment) or _GH_PAT.search(result.summary)
                    paper = Paper(
                        title=result.title.replace('\n', ' '),
                        authors=[a.name for a in result.authors],
                        abstract=result.summary.replace('\n', ' '),
                        source='arxiv',
                        published_date=pub,
                        doi=result.doi or None,
                        arxiv_id=arxiv_id,
                        url=result.entry_id,
                        venue='arXiv',
                        domains=_infer_domains(result.title, result.summary),
                        code_url=gh_match.group(0).rstrip('.,;:)') if gh_match else None,
                    )
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"arXiv query failed: {e}")

        return papers
