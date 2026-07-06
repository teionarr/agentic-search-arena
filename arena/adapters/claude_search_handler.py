"""Claude web-search handler — a native-answer provider not shipped by the base repo.

Uses the Anthropic ``web_search`` server tool (a frontier baseline, §4/§12). Unlike the
retrieval-only providers, Claude returns a provider-*synthesized* answer plus the results it
searched, so this handler carries an ``answer`` (the native-answer path, ``needs_synthesis``
is False for its adapter) alongside the normalized results.

Kept in the arena package so the base repo stays untouched. Follows the base handler contract
(async ``search`` returning ``{answer, search_response, provider_latency}``, sentinel-not-raise).

The Messages response is a list of content blocks: ``text`` blocks (the synthesized answer)
and ``web_search_tool_result`` blocks whose ``.content`` is a list of ``web_search_result``
items ``{url, title, page_age, encrypted_content}``. The result content is encrypted (not
human-readable), so evidence content falls back to the result title.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CLAUDE_SEARCH_MODEL = "claude-sonnet-4-6"


def _block_type(block: Any) -> Optional[str]:
    return getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")


def _block_field(block: Any, field: str) -> Any:
    return block.get(field) if isinstance(block, dict) else getattr(block, field, None)


def _serialize(response: Any) -> Dict[str, Any]:
    """Flatten the Messages response into a plain dict the normalizer can read.

    Produces ``{"answer": <synthesized text>, "results": [{url,title,page_age}]}`` — no
    provider identity, no encrypted payload, so it is safe to persist in ``raw``.
    """
    content = _block_field(response, "content") or []
    answer_parts, results = [], []
    for block in content:
        btype = _block_type(block)
        if btype == "text":
            text = _block_field(block, "text")
            if text:
                answer_parts.append(text)
        elif btype == "web_search_tool_result":
            for res in _block_field(block, "content") or []:
                if _block_type(res) != "web_search_result":
                    continue  # skip web_search_tool_result_error entries
                url = _block_field(res, "url")
                if url:
                    results.append({"url": url, "title": _block_field(res, "title") or "",
                                    "page_age": _block_field(res, "page_age")})
    return {"answer": "".join(answer_parts).strip(), "results": results}


class ClaudeSearchHandler:
    """Handles interactions with the Anthropic web_search server tool."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1",
                 client: Any = None):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key and client is None:
            raise ValueError("ANTHROPIC_API_KEY not provided to initialize Claude search handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.model = self.search_params.get("model", CLAUDE_SEARCH_MODEL)
        self.max_uses = self.search_params.get("max_uses", 5)
        self.is_llm_response = True
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: no key needed just to import arena
            self._client = anthropic.Anthropic(timeout=60.0, max_retries=0)
        return self._client

    async def search(self, query: str) -> Dict[str, Any]:
        tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_uses}
        try:
            start_time = time.time()
            response = self._get_client().messages.create(
                model=self.model, max_tokens=1024,
                messages=[{"role": "user", "content": query}], tools=[tool])
            end_time = time.time()
            flat = _serialize(response)
            return {"answer": flat["answer"], "search_response": flat,
                    "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Claude search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
