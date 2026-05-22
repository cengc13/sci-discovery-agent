from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import hashlib
import re


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    source: str
    published_date: Optional[date] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    biorxiv_id: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    domains: list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.pubmed_id:
            return f"pubmed:{self.pubmed_id}"
        if self.biorxiv_id:
            return f"biorxiv:{self.biorxiv_id}"
        normalized = re.sub(r'\W+', ' ', self.title.lower()).strip()
        return f"title:{hashlib.md5(normalized.encode()).hexdigest()[:12]}"

    @property
    def slug(self) -> str:
        s = self.title.lower()
        s = re.sub(r'[^a-z0-9\s-]', '', s)
        s = re.sub(r'\s+', '-', s.strip())
        s = s[:65].rstrip('-')
        if self.arxiv_id:
            s = f"{s}-{self.arxiv_id.replace('/', '-')}"
        elif self.doi:
            s = f"{s}-{self.doi.split('/')[-1][:12]}"
        return s

    @property
    def primary_url(self) -> str:
        if self.url:
            return self.url
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pubmed_id:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pubmed_id}/"
        return ""
