"""Orchestration: adapters -> reader -> order-swapped judge -> aggregate -> canonical result.

Work is **pipelined per query**: provider searches run as rate-limited per-provider streams,
and the moment a query's searches all finish it is handed to an LLM worker that runs its
reader -> grade -> judge chain. So the slow, rate-limited search phase (Exa's 2s/query, Brave's
1 req/s) overlaps with the reader/judge LLM work instead of blocking it. Every stage feeds a
health signal into ``stage_status`` so a live failure names its stage.
"""

import itertools
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock
from typing import Callable, Dict, List, Optional

from arena import reader as reader_mod
from arena.adapters.base import UnifiedResult
from arena.aggregate import aggregate
from arena.anchors import compute_anchors, machine_verify
from arena.config import ArenaConfig, Query
from arena.evidence import cap_evidence
from arena.grade import grade_answer
from arena.judge import judge_pair
from arena.cost import attach_cost, effective_weights, load_pricing
from arena.metrics import (aggregate_freshness, evidence_coverage, freshness_score,
                           latency_percentiles, parse_freshness_window_days)
from arena.scope import Scope
from arena.self_preference import self_preference_label
from arena.tokens import calculate_token_consumption

logger = logging.getLogger(__name__)


def _run_search_sync(adapter, query: str) -> UnifiedResult:
    """Run one adapter's async search in a fresh event loop (thread-pool worker)."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(adapter.search(query))
    finally:
        loop.close()


def _search_with_retry(adapter, query: str, retries: int = 3) -> UnifiedResult:
    """Search with backoff on a provider *error* (e.g. 429). Retry only when the call clearly
    errored — a base handler swallows the error and returns ``search_response=None`` with no
    results. A result with any docs, or a genuine empty response (``search_response`` is a
    dict), is NOT retried."""
    res = _run_search_sync(adapter, query)
    for attempt in range(retries):
        if res.results:
            return res  # clearly succeeded
        raw = res.raw if isinstance(res.raw, dict) else None
        errored = raw is not None and raw.get("search_response") is None
        if not errored:
            return res  # genuine empty (or unknown shape) — not an error, don't retry
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 8))
            res = _run_search_sync(adapter, query)
    return res


def _pipelined_run(adapters: List, queries: List, conc: int, process_query: Callable) -> List[dict]:
    """Overlap searches with per-query LLM work.

    Each provider runs a rate-limited search stream; the moment a query's searches all land it
    is enqueued for an LLM worker to run its reader/grade/judge chain. So the slow search phase
    overlaps with LLM work instead of blocking it. Returns the per-query local deltas."""
    nq = len(queries)
    search_results: Dict[tuple, UnifiedResult] = {}
    remaining = [len(adapters)] * nq
    lock = Lock()
    ready: Queue = Queue()
    out: List[dict] = []

    def _search_stream(a):
        interval = getattr(a, "min_interval_s", 0.0)
        for qi in range(nq):
            try:
                r = _search_with_retry(a, queries[qi].query)
            except Exception as e:  # guarantee remaining[qi] still decrements (no deadlock)
                logger.error(f"[{a.name}] search failed: {e}")
                r = UnifiedResult(raw={"error": str(e)}, empty_evidence=True)
            with lock:
                search_results[(qi, a.name)] = r
                remaining[qi] -= 1
                done = remaining[qi] == 0
            if done:
                ready.put(qi)
            if interval:
                time.sleep(interval)

    def _llm_worker():
        while True:
            qi = ready.get()
            if qi is None:
                return
            out.append(process_query(qi, {a.name: search_results.get((qi, a.name)) for a in adapters}))

    with ThreadPoolExecutor(max_workers=len(adapters) + conc) as ex:
        sfuts = [ex.submit(_search_stream, a) for a in adapters]
        lfuts = [ex.submit(_llm_worker) for _ in range(conc)]
        wait(sfuts)                       # all searches finished + enqueued
        for _ in range(conc):
            ready.put(None)               # stop the LLM workers
        wait(lfuts)
    return out


def run_arena(config: ArenaConfig, queries: List[Query], adapters: List, scope: Scope,
              reader_llm, judge_llm, token_model: str = "gpt-4.1",
              search_gatherer: Optional[Callable] = None, grader_llm=None) -> dict:
    """Run the arena and return the canonical result dict (source of truth for report.py).

    When queries carry ``expected_answer``, each provider's answer is graded against gold to
    produce a judge-free **accuracy** column alongside the arena rank (§8). ``grader_llm`` is a
    Claude fallback grader used only when ``OPENAI_API_KEY`` is absent.
    """
    run_nonce = uuid.uuid4().hex
    provider_names = [a.name for a in adapters]
    conc = max(1, config.max_concurrency)

    # Self-preference caveat inputs (§5/§6). Native-answer providers keep their own answer;
    # a Claude judge may favour a Claude-family native answer by style.
    from arena.adapters.registry import claude_family_providers
    native_providers = {a.name for a in adapters if getattr(a, "native_answer", False)}
    claude_family = set(claude_family_providers())
    judge_is_claude = (config.judge_primary or "").lower() == "claude"
    # A configured secondary judge is not actually invoked yet (config.judge_secondary is
    # reserved, not wired), so it must NOT be allowed to suppress the mitigation: we neither
    # route Claude-native pairs to it nor let its mere presence silence the label/caveat.
    has_secondary_judge = False
    # The caveat fires only for a *Claude-family* native provider — a non-Claude native
    # provider (e.g. a future Perplexity Sonar) shares the native-answer path but not the
    # self-preference risk.
    claude_native_mode = bool(native_providers & claude_family)

    # Per-provider accumulators.
    latencies: Dict[str, List] = {p: [] for p in provider_names}
    coverage_tokens: Dict[str, List[int]] = {p: [] for p in provider_names}
    cells_attempted: Dict[str, int] = {p: 0 for p in provider_names}
    cells_succeeded: Dict[str, int] = {p: 0 for p in provider_names}
    empty_evidence_count: Dict[str, int] = {p: 0 for p in provider_names}
    reader_degenerate_count: Dict[str, int] = {p: 0 for p in provider_names}
    reader_answers_made: Dict[str, int] = {p: 0 for p in provider_names}

    acc_correct: Dict[str, int] = {p: 0 for p in provider_names}
    acc_total: Dict[str, int] = {p: 0 for p in provider_names}
    # Cost units reported by adapters, and the count of cells that carried units. Both None/0
    # until a provider reports units → blank cost, §8.2. Normalized to $/query on attach.
    cost_units: Dict[str, Optional[float]] = {p: None for p in provider_names}
    cost_unit_cells: Dict[str, int] = {p: 0 for p in provider_names}

    # Per-provider per-query freshness tallies, gathered only on time-sensitive queries (§8.3).
    freshness_tallies: Dict[str, List[dict]] = {p: [] for p in provider_names}
    any_freshness_query = any(q.freshness_need for q in queries)

    mv_correct: Dict[str, int] = {p: 0 for p in provider_names}  # Tier-1 machine-verify anchors
    mv_total: Dict[str, int] = {p: 0 for p in provider_names}

    # ---- The per-query LLM chain (reader -> grade -> judge). Pure: reads only its own
    #      search results + config/llms and returns a local delta to merge (no shared state). ----
    def _process_query(qi: int, results_for_q: Dict[str, UnifiedResult]) -> dict:
        q = queries[qi]
        loc = {"qi": qi, "comparisons": [], "rationale": [], "reader_answers": {},
               "swap_total": 0, "swap_flips": 0, "judge_skipped": 0, "injection_flags": 0,
               "self_pref_flags": 0,
               "cal_agree": 0, "cal_decidable": 0, "cal_abstained": 0,
               "prov": {p: {"latency": [], "coverage": [], "cells_att": 0, "cells_succ": 0,
                            "empty": 0, "reader_made": 0, "reader_degen": 0,
                            "acc_correct": 0, "acc_total": 0,
                            "units": None, "units_cells": 0,
                            "freshness": [],
                            "mv_correct": 0, "mv_total": 0} for p in provider_names}}
        fresh_window = parse_freshness_window_days(q.freshness_need) if q.freshness_need else None
        answers, correct = {}, {}
        for name in provider_names:
            pv = loc["prov"][name]
            pv["cells_att"] += 1
            res = results_for_q.get(name)
            if res is None or res.empty_evidence or not res.results:
                pv["empty"] += 1
                continue
            pv["cells_succ"] += 1
            if res.cost_units is not None:  # sum billable units for the cost column (§8.2)
                pv["units"] = (pv["units"] or 0.0) + res.cost_units
                pv["units_cells"] += 1
            if res.latency_ms is not None:
                pv["latency"].append(res.latency_ms)
            if fresh_window is not None:  # freshness measures the returned evidence's dating
                pv["freshness"].append(freshness_score(res.results, fresh_window))
            capped = cap_evidence(res.results, config.evidence_budget_tokens, token_model)
            for d in capped:
                pv["coverage"].append(calculate_token_consumption(d.content, token_model))
            ans = reader_mod.synthesize(reader_llm, q.query, capped, run_nonce)
            pv["reader_made"] += 1
            if reader_mod.is_degenerate(ans, capped):
                pv["reader_degen"] += 1
                continue
            answers[name] = {"answer": ans, "docs": capped}
            loc["reader_answers"][name] = ans  # Tier-1 consensus (§3): raw reader answer
            if q.expected_answer is not None and str(q.expected_answer).strip() != "":
                # Tier-1 FREE anchor: a deterministic machine check needs no model. Only fall
                # back to the LLM grader when the expected answer isn't mechanically checkable.
                c = machine_verify(ans, q.expected_answer)
                if c is not None:
                    pv["mv_total"] += 1
                    pv["mv_correct"] += int(c)
                else:
                    c = grade_answer(q.query, ans, q.expected_answer, llm=grader_llm)
                if c is not None:
                    pv["acc_total"] += 1
                    pv["acc_correct"] += int(c)
                    correct[name] = bool(c)

        for x, y in itertools.combinations([p for p in provider_names if p in answers], 2):
            sp_label = self_preference_label(
                x, y, x in native_providers, y in native_providers, claude_family,
                judge_is_claude, has_secondary_judge)
            verdict = judge_pair(judge_llm, q.query, answers[x], answers[y], run_nonce,
                                 order_swap=config.order_swap, exclude_on_flip=config.exclude_on_flip,
                                 self_preference=sp_label)
            if verdict.get("skipped"):
                loc["judge_skipped"] += 1
            else:
                loc["swap_total"] += 1
                if verdict["flipped"]:
                    loc["swap_flips"] += 1
            if verdict["injection_flag"]:
                loc["injection_flags"] += 1
            if verdict.get("self_preference"):
                loc["self_pref_flags"] += 1
            outcome = verdict["outcome"]
            winner = x if outcome == "x" else y if outcome == "y" else ("tie" if outcome == "tie" else None)
            cx, cy = correct.get(x), correct.get(y)
            if cx is not None and cy is not None and cx != cy:
                if winner in (x, y):
                    loc["cal_decidable"] += 1
                    loc["cal_agree"] += int(winner == (x if cx else y))
                else:
                    loc["cal_abstained"] += 1
            loc["comparisons"].append({"a": x, "b": y, "winner": winner})
            loc["rationale"].append({
                "query": q.query, "a": x, "b": y, "winner": winner,
                "flipped": verdict["flipped"], "low_confidence": verdict["low_confidence"],
                "injection_flag": verdict["injection_flag"], "rationales": verdict["rationales"],
                "self_preference": verdict.get("self_preference"),
            })
        return loc

    # ---- Orchestration: overlap searches with the per-query LLM chain ----
    if search_gatherer:  # test path: gather sync, then process each query
        locals_out = []
        for qi, q in enumerate(queries):
            d = search_gatherer(adapters, q.query)
            locals_out.append(_process_query(qi, {a.name: d.get(a.name) for a in adapters}))
    else:
        locals_out = _pipelined_run(adapters, queries, conc, _process_query)

    # ---- Merge per-query deltas ----
    comparisons: List[dict] = []
    rationale_log: List[dict] = []
    per_query_answers: List[Dict[str, Optional[str]]] = []   # Tier-1 consensus, ordered by qi
    per_query_comparisons: List[List[dict]] = []
    swap_flips = swap_total = judge_skipped = injection_flags = self_pref_flags = 0
    cal_agree = cal_decidable = cal_abstained = 0
    for loc in sorted(locals_out, key=lambda l: l["qi"]):
        comparisons.extend(loc["comparisons"])
        rationale_log.extend(loc["rationale"])
        per_query_answers.append(loc["reader_answers"])
        per_query_comparisons.append(loc["comparisons"])
        swap_total += loc["swap_total"]; swap_flips += loc["swap_flips"]
        judge_skipped += loc["judge_skipped"]; injection_flags += loc["injection_flags"]
        self_pref_flags += loc["self_pref_flags"]
        cal_agree += loc["cal_agree"]; cal_decidable += loc["cal_decidable"]; cal_abstained += loc["cal_abstained"]
        for p in provider_names:
            pv = loc["prov"][p]
            latencies[p].extend(pv["latency"]); coverage_tokens[p].extend(pv["coverage"])
            cells_attempted[p] += pv["cells_att"]; cells_succeeded[p] += pv["cells_succ"]
            empty_evidence_count[p] += pv["empty"]; reader_answers_made[p] += pv["reader_made"]
            reader_degenerate_count[p] += pv["reader_degen"]
            acc_correct[p] += pv["acc_correct"]; acc_total[p] += pv["acc_total"]
            if pv["units"] is not None:
                cost_units[p] = (cost_units[p] or 0.0) + pv["units"]
                cost_unit_cells[p] += pv["units_cells"]
            freshness_tallies[p].extend(pv["freshness"])
            mv_correct[p] += pv["mv_correct"]; mv_total[p] += pv["mv_total"]

    agg = aggregate(comparisons, provider_names, seed=0)
    anchors = compute_anchors(per_query_answers, per_query_comparisons, provider_names,
                              min_providers=config.consensus_min_providers)
    for p in provider_names:
        anchors.auto_verify[p] = {"correct": mv_correct[p], "total": mv_total[p]}

    # ---- Metrics + stage_status ----
    per_provider = {}
    for p in provider_names:
        per_provider[p] = {
            "latency": latency_percentiles(latencies[p]),
            "coverage": evidence_coverage(coverage_tokens[p]),
            "empty_evidence_rate": _rate(empty_evidence_count[p], cells_attempted[p]),
            "reader_degenerate_rate": _rate(reader_degenerate_count[p], reader_answers_made[p]),
            "accuracy": {"correct": acc_correct[p], "total": acc_total[p],
                         "rate": _rate(acc_correct[p], acc_total[p])},
            "cells_succeeded": cells_succeeded[p],
            "cells_attempted": cells_attempted[p],
        }
        # Freshness is present only when the run had time-sensitive queries AND this provider
        # returned results on them (§8.3); absent -> its weight is dropped + renormalized (§8).
        if any_freshness_query:
            fresh = aggregate_freshness(freshness_tallies[p])
            if fresh is not None:
                per_provider[p]["freshness"] = fresh

    # Cost-per-query column (§8.2): normalize each provider's summed units to units/query, then
    # price via the dated pricing map. Providers reporting no units get a blank cost.
    pricing = load_pricing(config.pricing_path)
    units_per_query = {p: (cost_units[p] / cost_unit_cells[p]) if cost_unit_cells[p] else None
                       for p in provider_names}
    attach_cost(per_provider, pricing, units_per_query)
    # Drop the cost weight and renormalize the rest when cost is blank for the run (§8).
    weights_effective = effective_weights(config.weights, per_provider) if config.weights else {}

    swap_consistency = 1.0 - (swap_flips / swap_total) if swap_total else None
    n_ranked = sum(1 for s in agg.scores if s.status == "ranked")

    stage_status = {
        "secrets": _ok(len(scope.included) > 0, f"{len(scope.included)} providers included"),
        "adapters": _ok(any(per_provider[p]["cells_succeeded"] > 0 for p in provider_names),
                        "evidence returned by ≥1 provider"),
        "reader": _ok(any((per_provider[p]["reader_degenerate_rate"] or 0) < 1.0
                          for p in provider_names if per_provider[p]["cells_succeeded"]),
                      "reader produced usable answers"),
        "judge": _judge_status(swap_consistency, injection_flags, judge_skipped),
        "aggregate": _ok(n_ranked >= 1, f"{n_ranked} providers ranked, {agg.n_excluded} comparisons excluded"),
        "pipeline": _ok(True, "run completed"),
    }

    seen, cost = set(), 0.0
    for c in (judge_llm, reader_llm, grader_llm):
        if c is not None and id(c) not in seen and hasattr(c, "cost_usd"):
            cost += c.cost_usd()
            seen.add(id(c))

    return {
        "run_nonce": run_nonce,
        "providers": provider_names,
        "scope": scope.as_dict(),
        "ranking": [_score_dict(s) for s in agg.scores],
        "tie_groups": agg.tie_groups,
        "n_decided_comparisons": agg.n_decided,
        "n_excluded_comparisons": agg.n_excluded,
        "metrics": per_provider,
        "weights_effective": weights_effective,
        "judge": {"swap_consistency": swap_consistency, "swap_total": swap_total,
                  "swap_flips": swap_flips, "judge_skipped": judge_skipped,
                  "injection_flags": injection_flags,
                  # Self-preference caveat (§5/§6): surfaced when a Claude-family native
                  # provider is ranked under a Claude judge.
                  "native_mode": claude_native_mode,
                  "self_preference_flags": self_pref_flags,
                  "self_preference_caveat": claude_native_mode and judge_is_claude},
        "calibration": {"agreement": (cal_agree / cal_decidable) if cal_decidable else None,
                        "n_decidable": cal_decidable, "n_abstained": cal_abstained},
        "cost_usd": round(cost, 4),
        "anchors": anchors.as_dict(),
        "stage_status": stage_status,
        "degenerate_run": len(scope.included) < 3,
        "rationale_log": rationale_log,
        "n_queries": len(queries),
    }


def _rate(num: int, denom: int):
    return (num / denom) if denom else None


def _cost(client) -> float:
    return client.cost_usd() if hasattr(client, "cost_usd") else 0.0


def _ok(healthy: bool, reason: str) -> dict:
    return {"status": "green" if healthy else "red", "reason": reason}


def _judge_status(swap_consistency, injection_flags, judge_skipped=0, bar: float = 0.85) -> dict:
    if swap_consistency is None:
        reason = "no comparisons judged"
        if judge_skipped:
            reason += f" ({judge_skipped} judge calls skipped — check model id / API access)"
        return {"status": "red", "reason": reason}
    healthy = swap_consistency >= bar and judge_skipped == 0
    reason = f"swap-consistency {swap_consistency:.2f}"
    if judge_skipped:
        reason += f", {judge_skipped} skipped"
    if injection_flags:
        reason += f", {injection_flags} injection-flagged rationale(s)"
    return {"status": "green" if healthy else "red", "reason": reason}


def _score_dict(s) -> dict:
    return {"provider": s.provider, "win_rate": s.win_rate, "ci_low": s.ci_low,
            "ci_high": s.ci_high, "n_comparisons": s.n_comparisons, "status": s.status,
            "rank": s.rank, "tie_group": s.tie_group}
