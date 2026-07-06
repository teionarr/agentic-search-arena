"""Parallel.ai Search API handler — a new provider not shipped by the base repo.

Kept in the arena package (not ``handlers/``) so the base repo stays untouched. Follows the
base handler contract: async ``search(query) -> {answer, search_response, provider_latency}``
with the sentinel-not-raise error idiom. Only depends on aiohttp.

Parallel search (POST https://api.parallel.ai/v1/search) authenticates via the ``x-api-key``
header. The request takes an ``objective`` plus ``search_queries`` (a list); we pass the raw
query as both so it works with just a query string. It returns
``{"search_id": ..., "results": [{url,title,publish_date,excerpts}], "session_id": ...}``.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

PARALLEL_API_URL = "https://api.parallel.ai/v1/search"


class ParallelHandler:
    """Handles interactions with the Parallel Search API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("PARALLEL_API_KEY")
        if not self.api_key:
            raise ValueError("PARALLEL_API_KEY not provided to initialize Parallel handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        # Parallel needs search_queries (list) + objective; derive both from the raw query.
        data = {"objective": query, "search_queries": [query], **self.search_params}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.post(PARALLEL_API_URL, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in Parallel search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": "", "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Parallel search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
