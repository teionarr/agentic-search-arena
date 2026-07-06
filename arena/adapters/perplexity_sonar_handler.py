"""Perplexity Sonar handler — the second native-answer provider (after Claude web search).

Sonar is Perplexity's ANSWER API (POST /chat/completions, model ``sonar``): it returns a
provider-*synthesized* answer plus the ``search_results``/``citations`` it drew on, so this
handler carries an ``answer`` (the native-answer path, ``needs_synthesis`` is False for its
adapter) alongside the normalized results.

The base repo ships ``handlers/perplexity_handler.py`` for this same API, but it appends a
formatted "Sources:" list to the answer (which would leak citation URLs into the judged
native answer) and reports no ``provider_latency``, so the arena keeps its own thin handler.
Follows the base handler contract (async ``search`` returning
``{answer, search_response, provider_latency}``, sentinel-not-raise). Only depends on aiohttp.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
SONAR_MODEL = "sonar"


def _extract_answer(response_data: Dict[str, Any]) -> str:
    """The synthesized answer is the message content of the chat completion choice(s)."""
    parts = []
    for choice in response_data.get("choices", []) or []:
        content = (choice.get("message") or {}).get("content", "")
        if content:
            parts.append(content)
    return "".join(parts).strip()


class PerplexitySonarHandler:
    """Handles interactions with the Perplexity Sonar chat/completions API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not provided to initialize Perplexity Sonar handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.model = self.search_params.get("model", SONAR_MODEL)
        self.is_llm_response = True

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "messages": [{"role": "user", "content": query}]}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                start_time = time.time()
                async with session.post(PERPLEXITY_API_URL, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in Perplexity Sonar search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": _extract_answer(response_data),
                            "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Perplexity Sonar search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
