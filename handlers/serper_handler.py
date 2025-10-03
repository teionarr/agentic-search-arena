import os
import logging
import aiohttp
from typing import Dict, Any, Optional, Tuple
import time
from utils.token_utils import calculate_token_consumption, get_token_stats
from utils.utils import EvaluationType

from handlers.base_handler import ProviderHandler

logger = logging.getLogger(__name__)


SERPER_API_URL = "https://google.serper.dev"


class SerperHandler(ProviderHandler):
    """Handles interactions with the Serper API."""

    def __init__(
            self,
            search_params: Optional[Dict[str, Any]] = None,
            token_model: str = "gpt-4.1",
    ):
        """
        Initialize the SerperHandler.

        Args:
            search_params: Default search parameters to use for all searches
            token_model: Model to use for token consumption calculation
        """
        super().__init__(
            api_key=os.getenv("SERPER_API_KEY"),
            api_url=SERPER_API_URL,
            search_params=search_params
        )
        self.token_model = token_model
        self.is_llm_response = False

    async def search(self, question: str) -> Dict[str, Any]:
        """Run a Serper search using async HTTP request.

        Args:
            question: The question to search for

        Returns:
            Dictionary containing 'answer' and 'search_response'
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "q": question,
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
                        logger.error(f"Error in Serper search: HTTP {response.status}")
                        error_text = await response.text()
                        logger.error(f"Response: {error_text}")
                        return {
                            "answer": "",
                            "search_response": None,
                            "provider_latency": None
                        }

                    response_data = await response.json()
                    end_time = time.time()
                    logger.info("Received response from Serper API")

                    return {
                        "answer": "",
                        "search_response": response_data,
                        "provider_latency": end_time - start_time
                    }

        except Exception as e:
            logger.error(f"Error in Serper search: {str(e)}")
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
        if "organic" in response_data:
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

        # Extract organic results
        if evaluation_type == EvaluationType.SIMPLEQA:
            for result in response_data["organic"]:
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                content = f"{title}\n{snippet}" if title and snippet else title or snippet
                if url and content:
                    token_counts.append(calculate_token_consumption(content, self.token_model))
                    search_results.append((url, content))
            formatted_results = self._format_search_results_for_prompt(search_results)
        elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
            organic_results = response_data["organic"]
            formatted_results = [str(organic_result) for organic_result in organic_results]
            token_counts = [calculate_token_consumption(document, self.token_model) for document in formatted_results]

        return formatted_results, token_counts