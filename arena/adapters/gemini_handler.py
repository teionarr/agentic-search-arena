"""Google Gemini grounding handler — a new provider not shipped by the base repo.

Kept in the arena package (not ``handlers/``) so the base repo stays untouched. Follows the
base handler contract: async ``search(query) -> {answer, search_response, provider_latency}``
with the sentinel-not-raise error idiom. Only depends on aiohttp.

Gemini grounding (POST .../v1beta/models/{model}:generateContent with the ``google_search``
tool) authenticates via the ``x-goog-api-key`` header. It returns a generated answer plus
``candidates[0].groundingMetadata`` carrying ``groundingChunks`` (the supporting web sources)
and ``groundingSupports`` (which answer segments each chunk supports) — the arena normalizes
those chunks into evidence ``results[]``. Key is read from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiHandler:
    """Handles interactions with the Google Gemini grounding (Google Search) API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY not provided to initialize Gemini handler")
        self.token_model = token_model
        self.search_params = dict(search_params or {})
        self.model = self.search_params.pop("model", DEFAULT_MODEL)
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        data = {
            "contents": [{"parts": [{"text": query}]}],
            "tools": [{"google_search": {}}],
            **self.search_params,
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in Gemini search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": "", "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Gemini search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
