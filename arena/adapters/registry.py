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
from typing import Any, Callable, Dict, List, Optional, Union

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
    # Billable units ONE search request consumes per the vendor's PUBLIC pricing page: a
    # constant, or a callable(config) -> Optional[float] when the count depends on the request
    # config (e.g. Tavily's search_depth). None — or a callable returning None — means per-call
    # billing is token-based or not deterministic from the config, so the adapter reports no
    # units and the provider's cost column stays blank (§8.2 honesty rule: never fabricated).
    cost_units_per_call: Union[float, Callable[[Dict[str, Any]], Optional[float]], None] = None

    def build(self, name: str, config: Dict[str, Any], token_model: str = "gpt-4.1") -> HandlerAdapter:
        handler = self._factory(config, token_model)
        adapter = HandlerAdapter(name=name, handler=handler,
                                 normalize_fn=normalize.NORMALIZERS[name],
                                 native_answer=self.native_answer)
        adapter.min_interval_s = self.min_interval_s
        adapter.cost_units_per_call = resolve_cost_units(self, config)
        return adapter


def resolve_cost_units(spec: ProviderSpec, config: Dict[str, Any]) -> Optional[float]:
    """Per-call billable units for ``config`` — constant or config-dependent (§8.2)."""
    units = spec.cost_units_per_call
    return units(config) if callable(units) else units


# ---- Per-call billing units (§8.2) --------------------------------------------------------
# Set ONLY where the vendor's public pricing makes the per-request unit count deterministic
# for the given config. Each constant cites the pricing page checked (2026-07-06); anything
# variable (billed on returned pages, token usage, or an unpublished tier) resolves to None.


def _tavily_units(config: Dict[str, Any]) -> Optional[float]:
    # https://docs.tavily.com/documentation/api-credits (checked 2026-07-06):
    # "basic" search = 1 API credit per request, "advanced" = 2. Other depths: unknown → None.
    return {"basic": 1.0, "advanced": 2.0}.get(config.get("search_depth", "basic"))


def _serper_units(config: Dict[str, Any]) -> Optional[float]:
    # https://serper.dev/ (checked 2026-07-06) bills per query ("Get 2,500 free queries");
    # one request with the default num=10 is one query. Larger num values may consume more
    # credits and the tier table is only shown in the dashboard, so anything above 10 → None.
    return 1.0 if config.get("num", 10) <= 10 else None


# https://docs.linkup.so/pages/documentation/development/pricing (checked 2026-07-06), in
# multiples of the $0.005 standard searchResults search (the pricing.yaml unit):
# standard searchResults $0.005 / sourcedAnswer|structured $0.006; deep $0.05 / $0.055.
_LINKUP_UNITS = {("standard", "searchResults"): 1.0, ("standard", "sourcedAnswer"): 1.2,
                 ("standard", "structured"): 1.2, ("deep", "searchResults"): 10.0,
                 ("deep", "sourcedAnswer"): 11.0, ("deep", "structured"): 11.0}


def _linkup_units(config: Dict[str, Any]) -> Optional[float]:
    return _LINKUP_UNITS.get((config.get("depth"), config.get("outputType")))


def _firecrawl_units(config: Dict[str, Any]) -> Optional[float]:
    # https://docs.firecrawl.dev/features/search (checked 2026-07-06): "The cost of a search
    # is 2 credits per 10 results, rounded up (1-10 results = 2 credits)". With limit <= 10
    # that is a flat 2 credits; above 10 it depends on how many results actually return, and
    # scrapeOptions adds per-result scraping credits — both non-deterministic → None.
    if config.get("scrapeOptions"):
        return None
    return 2.0 if config.get("limit", 5) <= 10 else None


def _parallel_units(config: Dict[str, Any]) -> Optional[float]:
    # https://parallel.ai/pricing (checked 2026-07-06): Search API $5/1k requests including
    # up to 10 results; https://docs.parallel.ai/api-reference/search/search: max_results
    # "Defaults to 10". Above 10, extra results bill $1/1k per result actually returned —
    # non-deterministic → None.
    return 1.0 if config.get("max_results", 10) <= 10 else None


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
        cost_units_per_call=_tavily_units,  # API credits, depth-dependent (1 basic / 2 advanced)
    ),
    "exa": ProviderSpec(
        required_env_keys=["EXA_API_KEY"],
        # highlights required or Exa returns no usable content (silently empty otherwise).
        default_config={"type": "auto", "contents": {"highlights": True}, "numResults": 10},
        _factory=_exa_factory,
        # No units: https://exa.ai/pricing (checked 2026-07-06) bills the search request
        # ($7/1k, up to 10 results) PLUS contents (highlights) at $1/1k pages per page
        # actually returned — the page count varies per query, so cost stays blank (§8.2).
    ),
    "brave": ProviderSpec(
        required_env_keys=["BRAVE_API_KEY"],
        default_config={"count": 10},
        _factory=_brave_factory,
        min_interval_s=1.1,  # Brave Free plan allows ~1 request/second
        # https://brave.com/search/api/ (checked 2026-07-06): Search plan bills flat
        # $5/1k requests — one request is one billable unit regardless of count.
        cost_units_per_call=1.0,
    ),
    "serper": ProviderSpec(
        required_env_keys=["SERPER_API_KEY"],
        default_config={"type": "search", "num": 10},
        _factory=_serper_factory,
        cost_units_per_call=_serper_units,  # 1 query per request at num<=10 (serper.dev)
    ),
    "perplexity_search": ProviderSpec(
        required_env_keys=["PERPLEXITY_API_KEY"],
        default_config={"max_results": 10, "max_tokens_per_page": 512},
        _factory=_perplexity_search_factory,
        # https://docs.perplexity.ai/getting-started/pricing (checked 2026-07-06): Search API
        # is $5/1k requests with "no token costs" — flat one unit per request.
        cost_units_per_call=1.0,
    ),
    "perplexity": ProviderSpec(
        required_env_keys=["PERPLEXITY_API_KEY"],
        default_config={"model": "sonar"},
        _factory=_perplexity_factory,
        native_answer=True,   # returns its own synthesized answer (native-answer path, §5)
        # family stays None: family exists only for the Claude self-preference caveat (§5/§6);
        # a non-Claude native provider shares the native path but not the caveat.
        # No units: https://docs.perplexity.ai/getting-started/pricing (checked 2026-07-06)
        # bills Sonar per token ($1/M input + $1/M output) — token-based, cost stays blank.
    ),
    "firecrawl": ProviderSpec(
        required_env_keys=["FIRECRAWL_API_KEY"],
        default_config={"limit": 10},
        _factory=_firecrawl_factory,
        cost_units_per_call=_firecrawl_units,  # 2 credits per <=10-result search (no scraping)
    ),
    "linkup": ProviderSpec(
        required_env_keys=["LINKUP_API_KEY"],
        default_config={"depth": "standard", "outputType": "searchResults"},
        _factory=_linkup_factory,
        cost_units_per_call=_linkup_units,  # per-search, depth/outputType-dependent (docs.linkup.so)
    ),
    "claude_search": ProviderSpec(
        required_env_keys=["ANTHROPIC_API_KEY"],
        default_config={"max_uses": 5},
        _factory=_claude_search_factory,
        native_answer=True,   # returns its own synthesized answer (native-answer path, §5)
        family="claude",      # frontier baseline; triggers the self-preference caveat (§5/§6)
        # No units: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        # (checked 2026-07-06) bills $10/1k searches PLUS standard token costs, and the number
        # of searches per request varies (Claude decides, capped by max_uses) — cost stays blank.
    ),
    "youcom": ProviderSpec(
        required_env_keys=["YOU_API_KEY"],
        default_config={"count": 10},
        _factory=_youcom_factory,
        # No units: You.com publishes no per-request price (you.com/docs pricing page not
        # publicly reachable, checked 2026-07-06) — unknowable per-call, cost stays blank.
    ),
    "parallel": ProviderSpec(
        required_env_keys=["PARALLEL_API_KEY"],
        # 'advanced' is Parallel's documented default (higher-quality retrieval + compression).
        default_config={"mode": "advanced"},
        _factory=_parallel_factory,
        cost_units_per_call=_parallel_units,  # 1 request at max_results<=10 (parallel.ai/pricing)
    ),
    "gemini": ProviderSpec(
        # GeminiHandler accepts either key; gate on the any-of set so a deployment with only
        # GOOGLE_API_KEY is still included (matches the handler's GEMINI_API_KEY-then-GOOGLE lookup).
        required_env_keys=[],
        any_of_env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        default_config={"model": "gemini-2.5-flash"},
        _factory=_gemini_factory,
        # No units: https://ai.google.dev/gemini-api/docs/pricing (checked 2026-07-06) bills
        # grounded requests $35/1k PLUS per-token model costs — token-based, cost stays blank.
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
