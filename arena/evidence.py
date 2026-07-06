"""Evidence handling shared by the reader and judge.

Two jobs, both load-bearing for neutrality and safety:
1. **Common budget cap** — truncate every provider's evidence to the same per-query token
   budget so a provider cannot win merely by returning more text (the verbosity confound).
   This is truncation of what the API already returned, not fetching — never a crawler.
2. **Inert rendering** — wrap each document in a delimiter carrying a per-run random nonce
   so untrusted web content cannot close the tag and forge instructions (prompt injection).
"""

import logging
from typing import List

from arena.adapters.base import EvidenceDoc
from arena.tokens import calculate_token_consumption

logger = logging.getLogger(__name__)

INJECTION_MARKERS = ("ignore previous", "ignore all", "ignore the above", "instructions",
                     "as an ai", "system prompt", "disregard")


def cap_evidence(docs: List[EvidenceDoc], budget_tokens: int,
                 token_model: str = "gpt-4.1") -> List[EvidenceDoc]:
    """Truncate ``docs`` so their combined content is <= ``budget_tokens``.

    Fills the budget document by document; the doc that crosses the budget is char-truncated
    proportionally. A provider that returns less than the budget keeps that (real) shortfall.
    """
    if budget_tokens <= 0 or not docs:
        return docs
    capped: List[EvidenceDoc] = []
    used = 0
    for d in docs:
        if used >= budget_tokens:
            break
        toks = calculate_token_consumption(d.content, token_model)
        if used + toks <= budget_tokens:
            capped.append(d)
            used += toks
        else:
            remaining = budget_tokens - used
            # Rough char-per-token proportional cut (token counting is model-specific; this
            # keeps the cap deterministic without re-encoding substrings).
            if toks > 0:
                keep_chars = max(1, int(len(d.content) * remaining / toks))
                capped.append(EvidenceDoc(url=d.url, title=d.title,
                                          content=d.content[:keep_chars],
                                          score=d.score, published_date=d.published_date))
            used = budget_tokens
            break
    return capped


def render_evidence(docs: List[EvidenceDoc], nonce: str) -> str:
    """Render docs as nonce-fenced inert data blocks. No provider identity appears."""
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(
            f'<evidence id="{i}" nonce="{nonce}">\n'
            f"url: {d.url}\n"
            f"{d.content}\n"
            f"</evidence>"
        )
    return "\n".join(lines)


def looks_injected(text: str) -> bool:
    """Cheap detector: does this text mention prompt-injection markers? (Flag, don't block.)"""
    low = (text or "").lower()
    return any(m in low for m in INJECTION_MARKERS)
