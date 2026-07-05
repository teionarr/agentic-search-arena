"""Linkup search handler — a new provider not shipped by the base repo.

Kept in the arena package so the base repo stays untouched. Follows the base handler contract
(async ``search`` returning ``{answer, search_response, provider_latency}``, sentinel-not-raise).
Only depends on aiohttp.

Linkup search (POST /v1/search, ``outputType=searchResults``) returns
``{"results": [{type, name, url, content}]}``. Note the query field is ``q`` (not ``query``).
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

LINKUP_API_URL = "https://api.linkup.so/v1/search"


class LinkupHandler:
    """Handles interactions with the Linkup Search API."""

    def __init__(self, search_params: Optional[Dict[str, Any]] = None, token_model: str = "gpt-4.1"):
        self.api_key = os.getenv("LINKUP_API_KEY")
        if not self.api_key:
            raise ValueError("LINKUP_API_KEY not provided to initialize Linkup handler")
        self.token_model = token_model
        self.search_params = search_params or {}
        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"q": query, **self.search_params}  # Linkup uses 'q', not 'query'
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(LINKUP_API_URL, json=data, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error in Linkup search: HTTP {response.status}")
                        return {"answer": "", "search_response": None, "provider_latency": None}
                    response_data = await response.json()
                    end_time = time.time()
                    return {"answer": "", "search_response": response_data,
                            "provider_latency": end_time - start_time}
        except Exception as e:
            logger.error(f"Error in Linkup search: {str(e)}")
            return {"answer": "", "search_response": None, "provider_latency": None}
