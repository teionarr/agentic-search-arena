"""Config loading (YAML/JSON, strong defaults) and the queries file loader (CSV/JSONL).

Zero-config is the default path: with no config file, the arena runs across every in-scope
provider whose key is present, using registry defaults. A config file only *overrides*.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

from arena.adapters.registry import provider_names

logger = logging.getLogger(__name__)


class ArenaType(Enum):
    """Local shim so we can reuse the base's output-dir convention without editing the
    shared ``EvaluationType`` enum in ``utils/utils.py``."""

    ARENA = "arena"


# Default per-set sample for benchmark-suite mode (§7): a few hundred; full runs opt-in.
DEFAULT_BENCHMARK_SAMPLE_SIZE = 300


@dataclass
class Query:
    """One row of the queries file."""

    query: str
    expected_answer: Optional[str] = None
    category: Optional[str] = None
    freshness_need: Optional[str] = None


@dataclass
class ArenaConfig:
    """Resolved run configuration. All fields have strong defaults (zero-config works)."""

    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> {enabled, config}
    reader_model: Optional[str] = None          # None -> judge model (llm.py default)
    judge_primary: str = "claude"
    judge_secondary: Optional[str] = None       # optional 2nd judge model id -> ensemble κ (§6.4)
    order_swap: bool = True
    exclude_on_flip: bool = True
    aggregation_method: str = "bradley_terry"   # §6.3 default; "winrate" keeps the M0 estimator
    judge_reliability_weighting: str = "auto"   # §6.3 per-judge; engages only with a per-judge signal
    evidence_budget_tokens: int = 600           # common per-provider evidence cap (§ heterogeneity)
    consensus_min_providers: int = 3            # Tier-1 (§3): min converging providers for a silver label
    max_concurrency: int = 8                     # concurrent reader/judge/search calls
    repeats: int = 1                             # ×N runs per query — providers are non-deterministic;
                                                 # single-shot numbers are noise (statistical honesty)
    save_traces: bool = False                    # persist per-query raw payloads + reader inputs
                                                 # (auditability §15); opt-in, redacted on write
    weights: Dict[str, float] = field(default_factory=dict)
    langfuse_enabled: bool = False           # M5 — optional tracing, off by default (§11)
    output_dir: str = "results"
    config_path: Optional[str] = None
    pricing_path: Optional[str] = None          # cost pricing map (§8.2); None -> configs/pricing.yaml
    # Benchmark-suite mode (M2, §7). Off by default; a sample per set unless overridden.
    benchmark_suite: bool = False
    benchmark_datasets: List[str] = field(default_factory=lambda: ["simpleqa"])
    benchmark_sample_size: int = DEFAULT_BENCHMARK_SAMPLE_SIZE
    published_claims_path: Optional[str] = None
    # Tier-3 downstream success (§3): the user's own end-task loop, run per provider.
    # ``{provider}`` in the command is substituted (also exported as ARENA_PROVIDER); exit 0
    # = success. None = Tier 3 off.
    downstream_command: Optional[str] = None
    downstream_runs: int = 5
    downstream_timeout_s: int = 300

    def __post_init__(self):
        # Fail fast on nonsensical values: a non-positive budget would silently disable the
        # evidence cap (uncapped providers skew the comparison); concurrency must be >= 1.
        if self.evidence_budget_tokens <= 0:
            raise ValueError("evidence_budget_tokens must be > 0")
        # A value < 2 (or a bool/non-int) would turn a single answer into "consensus" (§3 Tier 1).
        if (isinstance(self.consensus_min_providers, bool)
                or not isinstance(self.consensus_min_providers, int)
                or self.consensus_min_providers < 2):
            raise ValueError("consensus_min_providers must be an integer >= 2")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.downstream_runs < 1:
            raise ValueError("downstream_runs must be >= 1")


DEFAULT_CONFIG_PATH = "configs/arena.yaml"


def resolve_config_path(explicit: Optional[str]) -> Optional[str]:
    """Use the explicit --config, else the project's configs/arena.yaml if it exists.

    This means provider toggles (e.g. firecrawl/linkup disabled) are honored by every command
    without needing --config on each invocation."""
    if explicit:
        return explicit
    return DEFAULT_CONFIG_PATH if os.path.isfile(DEFAULT_CONFIG_PATH) else None


def load_config(config_path: Optional[str]) -> ArenaConfig:
    """Load an optional YAML/JSON config. Unknown provider keys are rejected with a clear error."""
    if not config_path:
        return ArenaConfig()

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        # safe_load only — never yaml.load (which can execute arbitrary objects).
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    providers = raw.get("providers", {}) or {}
    known = set(provider_names())
    unknown = set(providers) - known
    if unknown:
        raise ValueError(
            f"Unknown provider key(s) in config: {sorted(unknown)}. Known: {sorted(known)}"
        )

    judge = raw.get("judge", {}) or {}
    bench = ((raw.get("modes", {}) or {}).get("benchmark_suite", {})) or {}
    if not isinstance(bench, dict):  # allow `benchmark_suite: true` shorthand
        bench = {"enabled": bool(bench)}
    datasets = bench.get("datasets", ["simpleqa"])
    if isinstance(datasets, str):  # `datasets: simpleqa` — don't explode the string into chars
        datasets = [datasets]
    downstream = raw.get("downstream", {}) or {}
    aggregation = raw.get("aggregation", {}) or {}
    method = aggregation.get("method", "bradley_terry")
    if method not in ("bradley_terry", "winrate"):
        raise ValueError(
            f"Unknown aggregation.method: {method!r}. Known: 'bradley_terry', 'winrate'"
        )
    return ArenaConfig(
        providers=providers,
        reader_model=(raw.get("reader", {}) or {}).get("model"),
        judge_primary=judge.get("primary", "claude"),
        judge_secondary=judge.get("secondary"),
        order_swap=judge.get("order_swap", True),
        exclude_on_flip=judge.get("exclude_on_flip", True),
        aggregation_method=method,
        judge_reliability_weighting=aggregation.get("judge_reliability_weighting", "auto"),
        evidence_budget_tokens=raw.get("evidence_budget_tokens", 600),
        consensus_min_providers=raw.get("consensus_min_providers", 3),
        max_concurrency=raw.get("max_concurrency", 8),
        repeats=int(raw.get("repeats", 1)),
        save_traces=bool((raw.get("output", {}) or {}).get("save_traces", False)),
        weights=raw.get("weights", {}) or {},
        langfuse_enabled=bool((raw.get("langfuse", {}) or {}).get("enabled", False)),
        output_dir=(raw.get("output", {}) or {}).get("dir", "results"),
        config_path=config_path,
        pricing_path=(raw.get("pricing", {}) or {}).get("path"),
        benchmark_suite=bool(bench.get("enabled", False)),
        benchmark_datasets=list(datasets) or ["simpleqa"],
        benchmark_sample_size=int(bench.get("sample_size", DEFAULT_BENCHMARK_SAMPLE_SIZE)),
        published_claims_path=bench.get("published_claims_path"),
        downstream_command=downstream.get("command"),
        downstream_runs=int(downstream.get("runs", 5)),
        downstream_timeout_s=int(downstream.get("timeout_s", 300)),
    )


def load_queries(path: str) -> List[Query]:
    """Load a queries file (CSV or JSONL). Required column ``query``; optional
    ``expected_answer`` / ``category`` / ``freshness_need``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Queries file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:  # default to CSV (covers .csv and unknown extensions)
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

    queries: List[Query] = []
    for i, row in enumerate(rows):
        q = (row.get("query") or "").strip() if isinstance(row.get("query"), str) else row.get("query")
        if not q:
            raise ValueError(f"Row {i} is missing a non-empty 'query' field")
        queries.append(Query(
            query=q,
            expected_answer=row.get("expected_answer") or None,
            category=row.get("category") or None,
            freshness_need=row.get("freshness_need") or None,
        ))
    if not queries:
        raise ValueError("Queries file contained no rows")
    return queries
