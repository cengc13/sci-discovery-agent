from __future__ import annotations
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import date
from ..models import Paper

logger = logging.getLogger(__name__)

ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

QUERIES = [
    '(AI agent[Title] OR LLM agent[Title] OR agentic AI[Title] OR multi-agent[Title]) AND (chemistry[Title] OR materials[Title] OR biology[Title] OR drug discovery[Title])',
    '(self-driving lab[Title] OR autonomous laboratory[Title] OR robotic chemistry[Title] OR autonomous experiment[Title])',
    '(large language model[Title] OR LLM[Title]) AND (autonomous[Title] OR agent[Title] OR planning[Title] OR workflow[Title]) AND (chemistry[Title] OR biology[Title] OR materials[Title])',
    '(autonomous[Title] OR automated[Title]) AND (scientific discovery[Title] OR drug discovery[Title] OR materials discovery[Title] OR synthesis planning[Title])',
    '(closed-loop[Title]) AND (chemistry[Title] OR materials[Title] OR drug[Title] OR synthesis[Title] OR experiment[Title])',
]

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}

DOMAIN_KEYWORDS = {
    'chemistry': ['chemistry', 'chemical', 'synthesis', 'molecule', 'reaction', 'catalyst'],
    'materials': ['material', 'crystal', 'alloy', 'polymer', 'semiconductor'],
    'biology': ['biology', 'protein', 'drug', 'cell', 'gene', 'biolog', 'biochem', 'enzyme'],
}


def _infer_domains(title: str, abstract: str) -> list[str]:
    text = (title + ' ' + abstract).lower()
    domains = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    return domains or ['biology']


class PubmedFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.email = config.get('email', 'user@example.com')
        self.api_key = config.get('api_key', '')
        self.max_results = config.get('max_results', 100)

    def fetch(self, since: date, until: date) -> list[Paper]:
        seen_pmids: set[str] = set()
        all_pmids: list[str] = []

        for query in QUERIES:
            logger.info(f"PubMed: {query[:70]}...")
            for pmid in self._search(query, since, until):
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    all_pmids.append(pmid)

        papers: list[Paper] = []
        for i in range(0, len(all_pmids), 20):
            batch = all_pmids[i:i + 20]
            papers.extend(self._fetch_details(batch))
            time.sleep(0.35)

        return papers

    def _base_params(self) -> dict:
        params = {'email': self.email}
        if self.api_key:
            params['api_key'] = self.api_key
        return params

    def _search(self, query: str, since: date, until: date) -> list[str]:
        params = {
            **self._base_params(),
            'db': 'pubmed',
            'term': query,
            'retmax': self.max_results,
            'retmode': 'json',
            'datetype': 'pdat',
            'mindate': since.strftime('%Y/%m/%d'),
            'maxdate': until.strftime('%Y/%m/%d'),
        }
        try:
            r = requests.get(ESEARCH_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()['esearchresult']['idlist']
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return []

    def _fetch_details(self, pmids: list[str]) -> list[Paper]:
        params = {
            **self._base_params(),
            'db': 'pubmed',
            'id': ','.join(pmids),
            'rettype': 'xml',
            'retmode': 'xml',
        }
        try:
            r = requests.get(EFETCH_URL, params=params, timeout=60)
            r.raise_for_status()
            return self._parse_xml(r.text)
        except Exception as e:
            logger.warning(f"PubMed fetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[Paper]:
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"PubMed XML parse error: {e}")
            return []
        for article in root.findall('.//PubmedArticle'):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
        return papers

    def _parse_article(self, article: ET.Element) -> Paper | None:
        mc = article.find('MedlineCitation')
        if mc is None:
            return None

        pmid_el = mc.find('PMID')
        pmid = pmid_el.text if pmid_el is not None else None

        art = mc.find('Article')
        if art is None:
            return None

        title_el = art.find('ArticleTitle')
        title = ''.join(title_el.itertext()) if title_el is not None else ''
        if not title:
            return None

        abstract_parts = art.findall('.//AbstractText')
        abstract = ' '.join(''.join(p.itertext()) for p in abstract_parts)

        authors = []
        for author in art.findall('.//Author'):
            last = author.findtext('LastName', '')
            fore = author.findtext('ForeName', '')
            if last:
                authors.append(f"{fore} {last}".strip())

        venue = ''
        pub_date = None
        journal = art.find('Journal')
        if journal is not None:
            venue = journal.findtext('Title') or journal.findtext('ISOAbbreviation') or ''
            ji = journal.find('JournalIssue/PubDate')
            if ji is not None:
                year = ji.findtext('Year')
                month = ji.findtext('Month', '1')
                day = ji.findtext('Day', '1')
                if year:
                    try:
                        month_num = int(month) if month.isdigit() else MONTH_MAP.get(month[:3], 1)
                        day_num = int(day) if day.isdigit() else 1
                        pub_date = date(int(year), month_num, day_num)
                    except (ValueError, TypeError):
                        try:
                            pub_date = date(int(year), 1, 1)
                        except ValueError:
                            pass

        doi = None
        pd_section = article.find('PubmedData')
        if pd_section is not None:
            for aid in pd_section.findall('.//ArticleId'):
                if aid.get('IdType') == 'doi':
                    doi = aid.text

        url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else None

        return Paper(
            title=title.strip(),
            authors=authors,
            abstract=abstract.strip(),
            source='pubmed',
            published_date=pub_date,
            doi=doi,
            pubmed_id=pmid,
            url=url,
            venue=venue,
            domains=_infer_domains(title, abstract),
        )
