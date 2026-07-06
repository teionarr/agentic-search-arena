"""Token counting — a tiny tiktoken wrapper.

Behaviourally identical to ``utils.token_utils`` (same cl100k fallback, same rough-estimate
on error), reimplemented here because importing ``utils.token_utils`` runs ``utils/__init__``
which eagerly pulls ``langchain_openai``/``quotientai``. Decoupling keeps the arena core and
its Tier-A tests runnable without the base's heavy dependencies (same rationale as paths.py).
"""

import logging
from typing import List, Tuple

import tiktoken

logger = logging.getLogger(__name__)


def calculate_token_consumption(text: str, model: str = "gpt-4.1") -> int:
    """Number of tokens in ``text`` (best-effort; ~len/4 on any failure)."""
    try:
        if "gpt-4.1" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4.1")
        elif "gpt-4.1-mini" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def get_token_stats(token_counts: List[int]) -> Tuple[int, float]:
    if token_counts:
        total = sum(token_counts)
        return total, total / len(token_counts)
    return 0, 0
