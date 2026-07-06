"""Per-provider normalizers: raw ``search_response`` -> List[EvidenceDoc].

The raw results live at a different path per provider, with different field names. These
paths were verified against ``handlers/*_handler.py`` at ``tavily-ai/tavily-search-evals``.
This module is the ONLY place provider-specific extraction lives; nothing downstream knows
a provider's identity.
"""

import logging
from typing import Any, List

from arena.adapters.base import EvidenceDoc

logger = logging.getLogger(__name__)


def _response_data(raw: Any) -> Any:
    """The base handlers wrap the provider payload under ``search_response`` (None on error)."""
    if not isinstance(raw, dict):
        return None
    return raw.get("search_response")


def normalize_tavily(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    docs = []
    for res in data.get("results", []) or []:
        url = res.get("url", "")
        content = res.get("content", "")
        if url and content:
            docs.append(EvidenceDoc(url=url, title=res.get("title", ""), content=content,
                                    score=res.get("score")))
    return docs


def normalize_exa(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    docs = []
    for res in data.get("results", []) or []:
        url = res.get("url", "")
        content = res.get("highlights", "")
        if isinstance(content, list):
            content = " ".join(content)
        content = str(content) if content else ""
        if url and content:
            docs.append(EvidenceDoc(url=url, title=res.get("title", ""), content=content,
                                    score=res.get("score"),
                                    published_date=res.get("publishedDate")))
    return docs


def normalize_brave(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    results = (data.get("web") or {}).get("results", []) if isinstance(data.get("web"), dict) else []
    docs = []
    for res in results or []:
        url = res.get("url", "")
        title = res.get("title", "")
        description = res.get("description", "")
        content = f"{title}\n{description}" if title and description else title or description
        if url and content:
            docs.append(EvidenceDoc(url=url, title=title, content=content,
                                    published_date=res.get("page_age")))
    return docs


def normalize_serper(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    docs = []
    for res in data.get("organic", []) or []:
        url = res.get("link", "")
        title = res.get("title", "")
        snippet = res.get("snippet", "")
        content = f"{title}\n{snippet}" if title and snippet else title or snippet
        if url and content:
            docs.append(EvidenceDoc(url=url, title=title, content=content,
                                    published_date=res.get("date")))
    return docs


def normalize_perplexity_search(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    docs = []
    for res in data.get("results", []) or []:
        url = res.get("url", "")
        content = res.get("snippet", "")
        if url and content:
            docs.append(EvidenceDoc(url=url, title=res.get("title", ""), content=content,
                                    published_date=res.get("date")))
    return docs


def normalize_firecrawl(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    # v2: {"data": {"web": [...]}}; fall back to a flat {"results"/"data": [...]} if present.
    payload = data.get("data")
    if isinstance(payload, dict):
        results = payload.get("web", []) or []
    elif isinstance(payload, list):
        results = payload
    else:
        results = data.get("results", []) or []
    docs = []
    for res in results or []:
        url = res.get("url", "")
        content = res.get("markdown") or res.get("description") or res.get("snippet") or ""
        if url and content:
            docs.append(EvidenceDoc(url=url, title=res.get("title", ""), content=content))
    return docs


def normalize_linkup(raw: Any) -> List[EvidenceDoc]:
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if results is None and isinstance(data.get("searchResults"), dict):
        results = data["searchResults"].get("results")
    docs = []
    for res in results or []:
        if res.get("type") == "image":  # skip image results — no textual content
            continue
        url = res.get("url", "")
        content = res.get("content") or res.get("snippet") or ""
        if url and content:
            docs.append(EvidenceDoc(url=url, title=res.get("name", ""), content=content))
    return docs


def normalize_claude_search(raw: Any) -> List[EvidenceDoc]:
    # Claude's web_search results carry only {url, title, page_age}; the result body is
    # encrypted (not human-readable), so content falls back to the title.
    data = _response_data(raw)
    if not isinstance(data, dict):
        return []
    docs = []
    for res in data.get("results", []) or []:
        url = res.get("url", "")
        title = res.get("title", "")
        if url and title:
            docs.append(EvidenceDoc(url=url, title=title, content=title,
                                    published_date=res.get("page_age")))
    return docs


NORMALIZERS = {
    "tavily": normalize_tavily,
    "exa": normalize_exa,
    "brave": normalize_brave,
    "serper": normalize_serper,
    "perplexity_search": normalize_perplexity_search,
    "firecrawl": normalize_firecrawl,
    "linkup": normalize_linkup,
    "claude_search": normalize_claude_search,
}
