"""Firecrawl search handler — a new provider not shipped by the base repo.

Kept in the arena package (not ``handlers/``) so the base repo stays untouched. Follows the
base handler contract: async ``search(query) -> {answer, search_response, provider_latency}``
with the sentinel-not-raise error idiom. Only depends on aiohttp.

Firecrawl v2 search (POST /v2/search) returns ``{"data": {"web": [{url,title,description}]}}``
— SERP-style snippets by default, comparable to Brave/Serper.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

FIRECRAWL_API_URL = "https://api.firecrawl.dev/v2/search"


class FirecrawlHandler:
    """Handles interactions with the Firecrawl Search API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not provided to initialize Firecrawl handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"query": query, **self.search_params}
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(FIRECRAWL_API_URL, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in Firecrawl search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": "", "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Firecrawl search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
