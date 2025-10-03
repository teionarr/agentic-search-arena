import os
import time
import logging
import aiohttp
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from handlers.base_handler import ProviderHandler
from utils.token_utils import calculate_token_consumption, get_token_stats
from utils.utils import EvaluationType

load_dotenv()

logger = logging.getLogger(__name__)
PERPLEXITY_API_URL = "https://api.perplexity.ai"


class PerplexitySearchHandler(ProviderHandler):
    """Handles interactions with the Perplexity Search API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
            token_model: str = "gpt-4.1",
    ):
        """
        Initialize the PerplexitySearchHandler.

        Args:
            search_params: Default search parameters to use for all searches
            token_model: Model to use for token consumption calculation
        """
        super().__init__(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            api_url=PERPLEXITY_API_URL,
            search_params=search_params
        )
        self.token_model = token_model

        self.is_llm_response = False

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a Perplexity search using async HTTP request.

        Args:
            query: The query to search for

        Returns:
            Dictionary containing a results array with search results
        """
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "query": query,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                        f"{self.api_url}/search",
                        json=payload,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Perplexity search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "search_response": None,
                            "provider_latency": None
                        }

                    response_data = await response.json()
                    end_time = time.time()
                    logger.info("Received response from Perplexity Search API")

                    return {
                        "search_response": response_data,
                        "provider_latency": end_time - start_time
                    }

        except Exception as e:
            logger.error(f"Error in Perplexity search: {str(e)}")
            return {
                "search_response": None,
                "provider_latency": None
            }

    async def post_process(self, search_response: dict, evaluation_type: EvaluationType = EvaluationType.SIMPLEQA) -> Tuple[str, int, int]:
        """
        Post process search response.

        Args:
            search_response: Dictionary containing the search response
            evaluation_type: Type of evaluation 

        Returns:
            Tuple of (processed response ready for LLM prompt, token count, token average)
        """
        if "search_response" not in search_response or search_response["search_response"] is None:
            return "", 0, 0

        search_results = []
        token_counts = []

        # Extract search results
        response_data = search_response["search_response"]
        if "results" in response_data:
            search_results, token_counts = self._format_search_response(response_data, evaluation_type)

        token_count, token_avg = get_token_stats(token_counts)

        return search_results, token_count, token_avg

    def _format_search_response(self, response_data: dict, evaluation_type: EvaluationType) -> Tuple[list, list]:
        """
        Extract search response.

        Args:
            response_data: Dictionary containing the search response data
            evaluation_type: Type of evaluation

        Returns:
            Tuple of (formatted results, token counts)
        """
        search_results = []
        token_counts = []

        try:
            if evaluation_type == EvaluationType.SIMPLEQA:
                for res in response_data["results"]:
                    url = res.get("url", "No URL")
                    content = res.get("snippet", "")

                    if url and content:
                        token_counts.append(calculate_token_consumption(content, self.token_model))
                        search_results.append((url, content))
                formatted_results = self._format_search_results_for_prompt(search_results)
            elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
                search_results = response_data["results"]
                formatted_results = [str(search_result) for search_result in search_results]
                token_counts = [calculate_token_consumption(document, self.token_model) for document in formatted_results]

        except Exception as e:
            logger.error(f"Error processing Perplexity Search response: {str(e)}")
            return [], []

        return formatted_results, token_counts