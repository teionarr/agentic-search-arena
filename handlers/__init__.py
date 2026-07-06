from .tavily_handler import TavilyHandler
from .exa_handler import ExaHandler
try:
    from .gptr_handler import GPTRHandler
except ImportError:
    # gpt_researcher is absent under the slim arena install (requirements-arena.txt);
    # the arena never uses GPTRHandler, and run_evaluation.py runs on the full install.
    GPTRHandler = None
from .perplexity_handler import PerplexityHandler
from .perplexity_search_handler import PerplexitySearchHandler
from .serper_handler import SerperHandler
from .brave_handler import BraveHandler

all = [TavilyHandler, ExaHandler, GPTRHandler, PerplexityHandler, SerperHandler, BraveHandler, PerplexitySearchHandler]
