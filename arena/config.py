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
    judge_secondary: Optional[str] = None       # M1 — reserved, not wired
    order_swap: bool = True
    exclude_on_flip: bool = True
    evidence_budget_tokens: int = 600           # common per-provider evidence cap (§ heterogeneity)
    max_concurrency: int = 8                     # concurrent reader/judge/search calls
    weights: Dict[str, float] = field(default_factory=dict)
    output_dir: str = "results"
    config_path: Optional[str] = None

    def __post_init__(self):
        # Fail fast on nonsensical values: a non-positive budget would silently disable the
        # evidence cap (uncapped providers skew the comparison); concurrency must be >= 1.
        if self.evidence_budget_tokens <= 0:
            raise ValueError("evidence_budget_tokens must be > 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


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
    return ArenaConfig(
        providers=providers,
        reader_model=(raw.get("reader", {}) or {}).get("model"),
        judge_primary=judge.get("primary", "claude"),
        judge_secondary=judge.get("secondary"),
        order_swap=judge.get("order_swap", True),
        exclude_on_flip=judge.get("exclude_on_flip", True),
        evidence_budget_tokens=raw.get("evidence_budget_tokens", 600),
        max_concurrency=raw.get("max_concurrency", 8),
        weights=raw.get("weights", {}) or {},
        output_dir=(raw.get("output", {}) or {}).get("dir", "results"),
        config_path=config_path,
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
