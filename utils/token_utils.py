"""
Token calculation utilities.
"""

import tiktoken
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_token_consumption(text: str, model: str = "gpt-4.1") -> int:
    """
    Calculate token consumption for a given text string.

    Args:
        text: The text to count tokens for
        model: The model name to use for encoding (defaults to gpt-4.1)

    Returns:
        int: Number of tokens in the text
    """
    try:
        # Get the appropriate encoding for the model
        if "gpt-4.1" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4.1")
        elif "gpt-4.1-mini" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        # Encode the text and return the number of tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        logger.warning(f"Error calculating token consumption for {text[:10]}...")
        logger.info(f"Returning rough estimate of length divided by 4: {len(text) // 4}")
        return len(text) // 4

def get_token_stats(token_counts: list) -> Tuple[int, int]:
    """
    Get token stats from a list of token counts.
    """
    if len(token_counts) > 0:
        token_count = sum(token_counts)
        token_avg = token_count / len(token_counts)
    else:
        token_count = 0
        token_avg = 0
    return token_count, token_avg