"""Benchmark-suite mode (M2, §7) — re-run public sets under ONE identical policy.

This rides on the arena machinery: one config, one reader, one judge, one grader. It does
**not** reimplement benchmark infrastructure — it writes *thin loaders* over already-vendored
data files (§8.1) and reuses ``run_calibration`` (arena/calibrate.py) for the judge-vs-gold
number. Two payoffs from one mechanism:

  (1) **Calibration report** (§6.5, output #5): judge-vs-gold agreement % per dataset — how
      often the reference-free pairwise judge agrees with ground-truth ordering.
  (2) **Marketing-claims ledger** (§7, output #6): each vendor's *published* number shown next
      to the neutral re-run under identical conditions, reporting the **delta and the trace**.
      This does **not** accuse — a gap can be legitimate (different reader/judge/config/date;
      the web drifts), so every claim carries ``as_of`` + ``source`` and every rerun is
      timestamped. If no published numbers are supplied, the neutral re-run is still produced.

Defaults to a **sample** (a few hundred per set); full runs are opt-in via ``sample_size``.
When both the arena (user queries) and a benchmark ran, the **arena-vs-benchmark cross-signal**
is emitted so the user can see whether their-workload rank and public rank point the same way.

    python run_arena.py --queries q.csv --benchmark-suite
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import yaml

from arena.calibrate import run_calibration
from arena.config import DEFAULT_BENCHMARK_SAMPLE_SIZE, ArenaConfig, Query

logger = logging.getLogger(__name__)

# Default per-set sample (§7): a few hundred; full runs are opt-in via config/flag. Single-
# sourced in arena.config so the dataclass default and the loader agree.
DEFAULT_SAMPLE_SIZE = DEFAULT_BENCHMARK_SAMPLE_SIZE

# Where each vendored dataset lives. Only SimpleQA is vendored in this repo today; FRAMES /
# FreshQA loaders exist (§14) and read a file at the mapped path if the user drops one in.
DATASET_PATHS = {
    "simpleqa": "datasets/simple_qa_test_set.csv",
    "frames": "datasets/frames_test_set.csv",
    "freshqa": "datasets/freshqa_test_set.csv",
}


# --------------------------------------------------------------------------------------------
# Connectors: thin loaders into the common ``Query`` schema (loader-only, no network).
# --------------------------------------------------------------------------------------------

def _rows_from_file(path: str) -> List[dict]:
    """Read a benchmark file as a list of dict rows. Supports CSV and JSONL by extension."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Benchmark data file not found: {path}. Vendor the dataset there or pass an "
            "explicit path.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _first(row: dict, *keys: str) -> Optional[str]:
    """First present, non-empty value among candidate column names (schema-tolerant)."""
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            v = v.strip()
        if v:
            return v
    return None


def load_simpleqa(sample_size: int, path: Optional[str] = None) -> List[Query]:
    """SimpleQA (OpenAI): ``metadata,problem,answer`` — the vendored primary accuracy set."""
    path = path or DATASET_PATHS["simpleqa"]
    out = []
    for row in _rows_from_file(path):
        q = _first(row, "problem", "question", "query")
        if not q:
            continue
        out.append(Query(query=q, expected_answer=_first(row, "answer"), category="simpleqa"))
        if len(out) >= sample_size:
            break
    return out


def load_frames(sample_size: int, path: Optional[str] = None) -> List[Query]:
    """FRAMES (Google, multi-hop): ``Prompt``/``question`` + ``Answer``/``answer`` (§8.1)."""
    path = path or DATASET_PATHS["frames"]
    out = []
    for row in _rows_from_file(path):
        q = _first(row, "Prompt", "prompt", "question", "query")
        if not q:
            continue
        out.append(Query(query=q, expected_answer=_first(row, "Answer", "answer"),
                         category="frames"))
        if len(out) >= sample_size:
            break
    return out


def load_freshqa(sample_size: int, path: Optional[str] = None) -> List[Query]:
    """FreshQA (Vu et al., freshness): ``question`` + ``answer``; carries a freshness tag (§8.1).

    Note (§8.1 caveat 2): FreshQA is increasingly memorized; connected here for continuity.
    """
    path = path or DATASET_PATHS["freshqa"]
    out = []
    for row in _rows_from_file(path):
        q = _first(row, "question", "query", "Prompt")
        if not q:
            continue
        out.append(Query(
            query=q,
            expected_answer=_first(row, "answer", "Answer"),
            category="freshqa",
            freshness_need=_first(row, "freshness_need", "fact_type") or "recent",
        ))
        if len(out) >= sample_size:
            break
    return out


LOADERS: Dict[str, Callable[..., List[Query]]] = {
    "simpleqa": load_simpleqa,
    "frames": load_frames,
    "freshqa": load_freshqa,
}


def available_datasets() -> List[str]:
    """Datasets whose vendored data file is actually present on disk (loadable now)."""
    return [d for d, p in DATASET_PATHS.items() if os.path.isfile(p)]


def load_benchmark(dataset: str, sample_size: int = DEFAULT_SAMPLE_SIZE,
                   path: Optional[str] = None) -> List[Query]:
    """Load one public set into the common ``Query`` schema, capped at ``sample_size`` (§7)."""
    key = dataset.lower()
    if key not in LOADERS:
        raise ValueError(f"Unknown benchmark dataset: {dataset}. Known: {sorted(LOADERS)}")
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    return LOADERS[key](sample_size, path)


# --------------------------------------------------------------------------------------------
# Marketing-claims ledger (§7, output #6): published-vs-rerun delta + trace, no accusation.
# --------------------------------------------------------------------------------------------

def load_published_claims(path: Optional[str]) -> Dict[str, Dict[str, dict]]:
    """Load the user-editable published-numbers table.

    Shape (``dataset -> provider -> {score, as_of, source}``). Missing file => ``{}`` (the
    neutral re-run is still produced; §7). ``score`` is the vendor's *published* accuracy in
    [0,1]. ``as_of``/``source`` timestamp and attribute the claim so a gap is never an accusation.
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    claims = (raw.get("published_claims") or raw) if isinstance(raw, dict) else {}
    out: Dict[str, Dict[str, dict]] = {}
    for dataset, provs in claims.items():
        if not isinstance(provs, dict):
            continue
        out[dataset.lower()] = {
            str(prov): {
                "score": entry.get("score"),
                "as_of": entry.get("as_of"),
                "source": entry.get("source"),
            }
            for prov, entry in provs.items() if isinstance(entry, dict)
        }
    return out


def build_ledger(dataset: str, rerun_accuracy: Dict[str, dict],
                 published: Dict[str, dict], rerun_timestamp: str) -> List[dict]:
    """One ledger row per provider: neutral re-run next to any published claim + the delta/trace.

    ``rerun_accuracy``: ``provider -> {rate, correct, total}`` from the neutral re-run.
    ``published``: ``provider -> {score, as_of, source}`` (may be empty => rerun-only rows).
    ``delta = rerun_rate - published_score`` (None when either side is absent). Every row is
    timestamped and carries the claim's ``as_of``/``source`` so the gap is auditable, not accusatory.
    """
    providers = sorted(set(rerun_accuracy) | set(published))
    rows = []
    for prov in providers:
        rr = rerun_accuracy.get(prov, {}) or {}
        pub = published.get(prov, {}) or {}
        rerun_rate = rr.get("rate")
        pub_score = pub.get("score")
        delta = (round(rerun_rate - pub_score, 4)
                 if rerun_rate is not None and pub_score is not None else None)
        rows.append({
            "dataset": dataset,
            "provider": prov,
            "rerun_rate": rerun_rate,
            "rerun_correct": rr.get("correct"),
            "rerun_total": rr.get("total"),
            "rerun_as_of": rerun_timestamp,
            "published_score": pub_score,
            "published_as_of": pub.get("as_of"),
            "published_source": pub.get("source"),
            "delta": delta,
        })
    return rows


# --------------------------------------------------------------------------------------------
# Orchestration: run each set under one policy; assemble calibration + ledger + cross-signal.
# --------------------------------------------------------------------------------------------

def _accuracy_from_metrics(metrics: dict) -> Dict[str, dict]:
    """Pull the judge-free accuracy-vs-gold column out of a run's per-provider metrics."""
    return {p: (m.get("accuracy") or {}) for p, m in metrics.items()}


def run_benchmark_suite(datasets: List[str], sample_size: int, adapters, reader_llm, judge_llm,
                        grader_llm, config: ArenaConfig,
                        published: Optional[Dict[str, Dict[str, dict]]] = None,
                        search_gatherer: Optional[Callable] = None) -> dict:
    """Re-run every requested public set across all providers under one identical policy (§7).

    Returns a per-dataset report: neutral re-run accuracy, judge-vs-gold calibration (§6.5,
    reusing ``run_calibration``), and — when published numbers are supplied — the marketing
    ledger (§7). ``run_calibration`` already returns the accuracy column via the shared arena
    run, so there is no second pass.
    """
    published = published or {}
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report = {"timestamp": timestamp, "sample_size": sample_size, "datasets": {}}

    for dataset in datasets:
        key = dataset.lower()
        # Isolate each dataset: a missing FRAMES/FreshQA file or a run_calibration failure must
        # not discard calibration already computed for earlier sets. Record the error and go on.
        try:
            queries = load_benchmark(key, sample_size)
            logger.info(f"[benchmark-suite] {key}: {len(queries)} queries "
                        f"across {[a.name for a in adapters]}")

            cal = run_calibration(queries, adapters, reader_llm, judge_llm, grader_llm, config,
                                  search_gatherer=search_gatherer)
            rerun_accuracy = _accuracy_from_metrics(cal["metrics"])
            ledger = build_ledger(key, rerun_accuracy, published.get(key, {}), timestamp)

            report["datasets"][key] = {
                "n_queries": len(queries),
                "calibration": {
                    "agreement": cal["agreement"],
                    "n_decidable_pairs": cal["n_decidable_pairs"],
                    "n_judge_abstained": cal["n_judge_abstained"],
                    "grader": cal["grader"],
                },
                "rerun_accuracy": rerun_accuracy,
                "benchmark_rank": cal["ranking"],
                "ledger": ledger,
            }
        except Exception as e:
            logger.error(f"[benchmark-suite] {key} failed: {e}")
            report["datasets"][key] = {"error": str(e)}
    return report


def write_benchmark_report(report: dict, output_dir: str) -> str:
    """Write ``benchmark_suite.json`` (redacted) alongside the arena results. Source of truth
    for the calibration report + marketing ledger + cross-signal."""
    from arena.report import redact  # reuse the single scrub boundary; keep report.py footprint nil
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_suite.json")
    with open(path, "w") as f:
        json.dump(redact(report), f, indent=2)
    return path


def render_benchmark_summary(report: dict) -> str:
    """Compact CLI view: per-dataset calibration + marketing ledger + cross-signal."""
    W = 60
    out = ["", "═" * W, "  BENCHMARK SUITE (one policy, all providers)".ljust(W), "═" * W]
    out.append(f"  sample {report.get('sample_size')} / set · run {report.get('timestamp')}")
    for dataset, d in report.get("datasets", {}).items():
        if d.get("error"):  # isolated failure — surface it, keep rendering the rest
            out.append("")
            out.append(f"  {dataset}  ·  FAILED: {d['error']}")
            continue
        cal = d.get("calibration", {}) or {}
        agree = cal.get("agreement")
        agree_s = (f"{agree:.0%} ({'≥0.80 ok' if agree >= 0.80 else 'below 0.80 bar'})"
                   if agree is not None else "n/a (too few decidable pairs)")
        out.append("")
        out.append(f"  {dataset}  ·  {d.get('n_queries')} queries  ·  grader {cal.get('grader')}")
        out.append(f"    judge-vs-gold agreement: {agree_s}  ({cal.get('n_decidable_pairs')} decidable pairs)")
        out.append("    LEDGER  rerun = neutral re-run acc · pub = vendor-published (dated)")
        for row in d.get("ledger", []):
            rr = f"{row['rerun_rate']:.0%}" if row.get("rerun_rate") is not None else "n/a"
            pub = f"{row['published_score']:.0%}" if row.get("published_score") is not None else "—"
            if row.get("delta") is not None:
                delta = f"Δ {row['delta']:+.0%}"
                trace = f" (pub {row.get('published_as_of') or '?'}; {row.get('published_source') or 'no source'})"
            else:
                delta, trace = "Δ n/a", "  (no published number)"
            out.append(f"      {row['provider']:<16} rerun {rr:<5} pub {pub:<5} {delta}{trace}")
    cs = report.get("cross_signal")
    if cs:
        out.append("")
        out.append("  ARENA-vs-BENCHMARK CROSS-SIGNAL (your workload vs public rank)")
        out.append(f"    arena:     {' > '.join(cs.get('arena_order', [])) or 'n/a'}")
        for dataset, order in (cs.get("benchmark_order") or {}).items():
            out.append(f"    {dataset:<10} {' > '.join(order) or 'n/a'}")
    out.append("═" * W + "\n")
    return "\n".join(out)


def cross_signal(arena_ranking: Optional[List[dict]],
                 benchmark_report: dict) -> Optional[dict]:
    """Arena-vs-benchmark cross-signal (§7): the user's-workload rank next to the public rank.

    Emitted only when BOTH ran. Disagreement is the insight the blogs can't sell. Returns a
    per-provider ordering from each side (by rank), or None when the arena did not run.
    """
    if not arena_ranking:
        return None
    arena_order = [s["provider"] for s in sorted(
        (s for s in arena_ranking if s.get("rank") is not None), key=lambda s: s["rank"])]
    out = {"arena_order": arena_order, "benchmark_order": {}}
    for dataset, d in benchmark_report.get("datasets", {}).items():
        ranked = [s for s in d.get("benchmark_rank", []) if s.get("rank") is not None]
        out["benchmark_order"][dataset] = [s["provider"] for s in sorted(ranked, key=lambda s: s["rank"])]
    return out
