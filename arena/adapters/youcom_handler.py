"""You.com Search API handler — a new provider not shipped by the base repo.

Kept in the arena package (not ``handlers/``) so the base repo stays untouched. Follows the
base handler contract: async ``search(query) -> {answer, search_response, provider_latency}``
with the sentinel-not-raise error idiom. Only depends on aiohttp.

You.com search (GET https://ydc-index.io/v1/search) authenticates via the ``X-API-Key``
header and takes ``query`` as a URL query parameter. It returns
``{"results": {"web": [{url,title,description,snippets,page_age}], "news": [...]}, "metadata": {...}}``.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

YOUCOM_API_URL = "https://ydc-index.io/v1/search"


class YouComHandler:
    """Handles interactions with the You.com Search API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("YOU_API_KEY")
        if not self.api_key:
            raise ValueError("YOU_API_KEY not provided to initialize You.com handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"X-API-Key": self.api_key}
        params = {"query": query, **self.search_params}  # You.com passes query as a GET param
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get(YOUCOM_API_URL, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in You.com search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": "", "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in You.com search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
