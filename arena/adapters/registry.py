"""Provider registry — the ONLY place provider identity lives.

Mirrors the base repo's ``handler_map`` idiom: a plain dict keyed by lowercase provider
name. Each entry declares the env key(s) the provider needs, a ``default_config`` (always a
dict, never ``None`` — some handlers call ``search_params.get(...)`` in ``__init__``), and a
factory that builds the wrapped ``HandlerAdapter``.

Handler classes are imported lazily inside each factory: importing a base handler eagerly
runs ``handlers/__init__.py`` (which pulls ``gpt_researcher`` etc.), so we defer that cost
until a provider is actually built. Importing this module stays light, which keeps the
arena core and its Tier-A tests runnable without the base's heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from arena.adapters.base import HandlerAdapter
from arena.adapters import normalize


@dataclass
class ProviderSpec:
    """Registry entry for one in-scope provider."""

    required_env_keys: List[str]
    # Optional any-of gate: the provider is satisfied if AT LEAST ONE of these keys is present
    # (in addition to all of ``required_env_keys``). Used when a handler accepts alternate keys.
    any_of_env_keys: List[str] = field(default_factory=list)
    default_config: Dict[str, Any] = field(default_factory=dict)
    _factory: Callable[[Dict[str, Any], str], Any] = None
    # Minimum seconds between requests to this provider (respects per-plan rate limits).
    min_interval_s: float = 0.0
    # True for providers that return their own synthesized answer (native-answer path, §5).
    native_answer: bool = False
    # Model family, used only for the self-preference caveat (§5/§6). None = not applicable.
    family: Optional[str] = None

    def build(self, name: str, config: Dict[str, Any], token_model: str = "gpt-4.1") -> HandlerAdapter:
        handler = self._factory(config, token_model)
        adapter = HandlerAdapter(name=name, handler=handler,
                                 normalize_fn=normalize.NORMALIZERS[name],
                                 native_answer=self.native_answer)
        adapter.min_interval_s = self.min_interval_s
        return adapter


def _tavily_factory(config, token_model):
    from handlers.tavily_handler import TavilyHandler
    return TavilyHandler(config, token_model=token_model)


def _exa_factory(config, token_model):
    from handlers.exa_handler import ExaHandler
    return ExaHandler(config, token_model=token_model)


def _brave_factory(config, token_model):
    from handlers.brave_handler import BraveHandler
    return BraveHandler(config, token_model=token_model)


def _serper_factory(config, token_model):
    from handlers.serper_handler import SerperHandler
    return SerperHandler(config, token_model=token_model)


def _perplexity_search_factory(config, token_model):
    from handlers.perplexity_search_handler import PerplexitySearchHandler
    return PerplexitySearchHandler(config, token_model=token_model)


def _perplexity_factory(config, token_model):
    # Arena-native Sonar handler (the base's perplexity_handler appends a "Sources:" list to
    # the answer and reports no latency — see its module docstring); light import (aiohttp only).
    from arena.adapters.perplexity_sonar_handler import PerplexitySonarHandler
    return PerplexitySonarHandler(config, token_model=token_model)


def _firecrawl_factory(config, token_model):
    # Arena-native handler (not in the base repo); light import (aiohttp only).
    from arena.adapters.firecrawl_handler import FirecrawlHandler
    return FirecrawlHandler(config, token_model=token_model)


def _linkup_factory(config, token_model):
    from arena.adapters.linkup_handler import LinkupHandler
    return LinkupHandler(config, token_model=token_model)


def _claude_search_factory(config, token_model):
    # Arena-native native-answer handler (not in the base repo); lazily imports anthropic.
    from arena.adapters.claude_search_handler import ClaudeSearchHandler
    return ClaudeSearchHandler(config, token_model=token_model)


def _youcom_factory(config, token_model):
    from arena.adapters.youcom_handler import YouComHandler
    return YouComHandler(config, token_model=token_model)


def _parallel_factory(config, token_model):
    from arena.adapters.parallel_handler import ParallelHandler
    return ParallelHandler(config, token_model=token_model)


def _gemini_factory(config, token_model):
    from arena.adapters.gemini_handler import GeminiHandler
    return GeminiHandler(config, token_model=token_model)


# In-scope providers: document-returning plus the native-answer pair (claude_search,
# perplexity Sonar, §5). The remaining finished-answer provider (gptr) is M1. max_results /
# top-k held constant at 10 across providers (§15).
REGISTRY: Dict[str, ProviderSpec] = {
    "tavily": ProviderSpec(
        required_env_keys=["TAVILY_API_KEY"],
        default_config={"search_depth": "advanced", "max_results": 10, "include_answer": False},
        _factory=_tavily_factory,
    ),
    "exa": ProviderSpec(
        required_env_keys=["EXA_API_KEY"],
        # highlights required or Exa returns no usable content (silently empty otherwise).
        default_config={"type": "auto", "contents": {"highlights": True}, "numResults": 10},
        _factory=_exa_factory,
    ),
    "brave": ProviderSpec(
        required_env_keys=["BRAVE_API_KEY"],
        default_config={"count": 10},
        _factory=_brave_factory,
        min_interval_s=1.1,  # Brave Free plan allows ~1 request/second
    ),
    "serper": ProviderSpec(
        required_env_keys=["SERPER_API_KEY"],
        default_config={"type": "search", "num": 10},
        _factory=_serper_factory,
    ),
    "perplexity_search": ProviderSpec(
        required_env_keys=["PERPLEXITY_API_KEY"],
        default_config={"max_results": 10, "max_tokens_per_page": 512},
        _factory=_perplexity_search_factory,
    ),
    "perplexity": ProviderSpec(
        required_env_keys=["PERPLEXITY_API_KEY"],
        default_config={"model": "sonar"},
        _factory=_perplexity_factory,
        native_answer=True,   # returns its own synthesized answer (native-answer path, §5)
        # family stays None: family exists only for the Claude self-preference caveat (§5/§6);
        # a non-Claude native provider shares the native path but not the caveat.
    ),
    "firecrawl": ProviderSpec(
        required_env_keys=["FIRECRAWL_API_KEY"],
        default_config={"limit": 10},
        _factory=_firecrawl_factory,
    ),
    "linkup": ProviderSpec(
        required_env_keys=["LINKUP_API_KEY"],
        default_config={"depth": "standard", "outputType": "searchResults"},
        _factory=_linkup_factory,
    ),
    "claude_search": ProviderSpec(
        required_env_keys=["ANTHROPIC_API_KEY"],
        default_config={"max_uses": 5},
        _factory=_claude_search_factory,
        native_answer=True,   # returns its own synthesized answer (native-answer path, §5)
        family="claude",      # frontier baseline; triggers the self-preference caveat (§5/§6)
    ),
    "youcom": ProviderSpec(
        required_env_keys=["YOU_API_KEY"],
        default_config={"count": 10},
        _factory=_youcom_factory,
    ),
    "parallel": ProviderSpec(
        required_env_keys=["PARALLEL_API_KEY"],
        # 'advanced' is Parallel's documented default (higher-quality retrieval + compression).
        default_config={"mode": "advanced"},
        _factory=_parallel_factory,
    ),
    "gemini": ProviderSpec(
        # GeminiHandler accepts either key; gate on the any-of set so a deployment with only
        # GOOGLE_API_KEY is still included (matches the handler's GEMINI_API_KEY-then-GOOGLE lookup).
        required_env_keys=[],
        any_of_env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        default_config={"model": "gemini-2.5-flash"},
        _factory=_gemini_factory,
    ),
}


def native_answer_providers() -> List[str]:
    """Providers that return their own synthesized answer (native-answer path, §5)."""
    return [name for name, spec in REGISTRY.items() if spec.native_answer]


def claude_family_providers() -> List[str]:
    """Providers in the Claude model family — used for the self-preference caveat (§5/§6)."""
    return [name for name, spec in REGISTRY.items() if spec.family == "claude"]


def provider_names() -> List[str]:
    return list(REGISTRY.keys())
