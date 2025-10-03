import os
import time
import aiohttp
import logging
from typing import Dict, Any, Optional, Tuple

from handlers.base_handler import ProviderHandler
from utils.token_utils import calculate_token_consumption, get_token_stats
from utils.utils import EvaluationType

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1"


class BraveHandler(ProviderHandler):
    """Handles interactions with the Brave Search API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
            token_model: str = "gpt-4.1",
    ):
        """
        Initialize the BraveHandler.

        Args:
            search_params: Default search parameters to use for all searches
            token_model: Model to use for token consumption calculation
        """
        super().__init__(
            api_key=os.getenv("BRAVE_API_KEY"),
            api_url=BRAVE_API_URL,
            search_params=search_params
        )
        self.token_model = token_model
        self.is_llm_response = False

    async def search(self, question: str) -> Dict[str, Any]:
        """Run a Brave search using async HTTP request.

        Args:
            question: The question to search for

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }

        params = {
            "q": question,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                        f"{self.api_url}/web/search",
                        params=params,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error in Brave search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None,
                            "provider_latency": None
                        }

                    response_data = await response.json()
                    end_time = time.time()
                    logger.info("Received response from Brave Search API")

                    return {
                        "answer": "",
                        "search_response": response_data,
                        "provider_latency": end_time - start_time
                    }

        except Exception as e:
            logger.error(f"Error in Brave search: {str(e)}")
            return {
                "answer": "",
                "search_response": None,
                "provider_latency": None
            }

    async def post_process(self, search_response: dict, evaluation_type: EvaluationType = EvaluationType.SIMPLEQA) -> Tuple[str, int, int]:
        """
        Post process search response.

        Args:
            search_response: Dictionary containing the search response

        Returns:
            Tuple of (processed response ready for LLM prompt, token count, token average)
        """
        if "search_response" not in search_response or search_response["search_response"] is None:
            return "", 0, 0

        search_results = []
        token_counts = []

        # Extract web results
        response_data = search_response["search_response"]
        if "web" in response_data and "results" in response_data["web"]: 
            search_results, token_counts = self._format_search_response(response_data, evaluation_type) 

        token_count, token_avg = get_token_stats(token_counts)

        return search_results, token_count, token_avg   

    def _format_search_response(self, response_data: dict, evaluation_type: EvaluationType) -> Tuple[list, list]:
        """
        Extract search response.
        """
        search_results = []
        token_counts = []

        # Extract web results
        if evaluation_type == EvaluationType.SIMPLEQA:
            for result in response_data["web"]["results"]:
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("description", "")
                content = f"{title}\n{description}" if title and description else title or description
                if url and content:
                    token_counts.append(calculate_token_consumption(content, self.token_model))
                    search_results.append((url, content))
            formatted_results = self._format_search_results_for_prompt(search_results)
        elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
            web_results = response_data['web']['results']
            formatted_results = [str(web_result) for web_result in web_results] 
            token_counts = [calculate_token_consumption(document, self.token_model) for document in formatted_results]

        return formatted_results, token_counts