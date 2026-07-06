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
from arena.aggregate import aggregate, per_category_rankings, point_winrates
from arena.anchors import compute_anchors, machine_verify
from arena.config import ArenaConfig, Query
from arena.evidence import cap_evidence
from arena.grade import grade_answer
from arena.judge import judge_pair, route_native_self_preference
from arena.cost import attach_cost, attach_cost_per_success, effective_weights, load_pricing
from arena.metrics import (aggregate_freshness, evidence_coverage, freshness_score,
                           latency_percentiles, parse_freshness_window_days)
from arena.reliability import cohens_kappa, consensus_agreements, judge_weights
from arena.scope import Scope
from arena.self_preference import self_preference_label
from arena.tokens import calculate_token_consumption
from arena.tracing import NullTracer, Tracer

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
              search_gatherer: Optional[Callable] = None, grader_llm=None,
              secondary_judge_llm=None, tracer: Optional[Tracer] = None) -> dict:
    """Run the arena and return the canonical result dict (source of truth for report.py).

    When queries carry ``expected_answer``, each provider's answer is graded against gold to
    produce a judge-free **accuracy** column alongside the arena rank (§8). ``grader_llm`` is a
    Claude fallback grader used only when ``OPENAI_API_KEY`` is absent.

    ``secondary_judge_llm`` (§5): when supplied every pair is judged by both judges; inter-judge
    agreement (Cohen's κ, §6.4) is reported and per-judge reliability weighting (§6.3) engages.
    """
    run_nonce = uuid.uuid4().hex
    provider_names = [a.name for a in adapters]
    conc = max(1, config.max_concurrency)
    tracer = tracer or NullTracer()  # off by default; NullTracer spans are inert

    # Repeats (statistical honesty): providers are non-deterministic, so ``repeats: N`` runs
    # every query N times — real re-searches, not replayed results. All comparisons feed one
    # aggregation (CIs tighten with the extra samples); the per-repeat win-rate spread is
    # reported as the noise signal. qi // base_n recovers a comparison's repeat index.
    base_n = len(queries)
    repeats = max(1, getattr(config, "repeats", 1))
    if repeats > 1:
        queries = [q for _ in range(repeats) for q in queries]

    # Self-preference caveat inputs (§5/§6). Native-answer providers keep their own answer;
    # a Claude judge may favour a Claude-family native answer by style.
    from arena.adapters.registry import claude_family_providers
    native_providers = {a.name for a in adapters if getattr(a, "native_answer", False)}
    claude_family = set(claude_family_providers())
    judge_is_claude = (config.judge_primary or "").lower() == "claude"
    # The secondary judge is now wired (M1): when one is actually supplied, Claude-native pairs
    # ROUTE to it (the routing hook below) instead of carrying the ``possible-self-preference``
    # flag. With no secondary invoked, the flag mitigation stands (§5: route XOR flag).
    has_secondary_judge = secondary_judge_llm is not None
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
    error_count: Dict[str, int] = {p: 0 for p in provider_names}
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
        trace = tracer.trace("query", input={"query": q.query})
        loc = {"qi": qi, "comparisons": [], "rationale": [], "reader_answers": {}, "judge_labels": [],
               "swap_total": 0, "swap_flips": 0, "judge_skipped": 0, "injection_flags": 0,
               "self_pref_flags": 0,
               "cal_agree": 0, "cal_decidable": 0, "cal_abstained": 0,
               "prov": {p: {"latency": [], "coverage": [], "cells_att": 0, "cells_succ": 0,
                            "empty": 0, "errors": 0, "reader_made": 0, "reader_degen": 0,
                            "acc_correct": 0, "acc_total": 0,
                            "units": None, "units_cells": 0,
                            "freshness": [],
                            "mv_correct": 0, "mv_total": 0} for p in provider_names}}
        fresh_window = parse_freshness_window_days(q.freshness_need) if q.freshness_need else None
        # Audit trace (§15, opt-in): the raw provider payload + the exact evidence the reader
        # saw + the reader's answer, per provider — so any verdict can be replayed by hand.
        # Named ``audit`` to stay clear of the Langfuse ``trace`` span above.
        audit = ({"query": q.query, "category": q.category, "repeat": qi // base_n,
                  "providers": {}} if config.save_traces else None)
        loc["audit"] = audit
        answers, correct = {}, {}
        for name in provider_names:
            pv = loc["prov"][name]
            pv["cells_att"] += 1
            res = results_for_q.get(name)
            if audit is not None:
                audit["providers"][name] = {
                    "raw": (res.raw if res is not None else None),
                    "latency_ms": (res.latency_ms if res is not None else None),
                    "n_results": (len(res.results) if res is not None and res.results else 0),
                    "evidence": None, "reader_answer": None,
                }
            trace.child(f"provider.search:{name}",
                        input={"query": q.query},
                        output={"latency_ms": getattr(res, "latency_ms", None),
                                "results": [d.__dict__ for d in (res.results if res else [])]}).end()
            if res is None or res.empty_evidence or not res.results:
                pv["empty"] += 1
                # Reliability: an *errored* call (exception, or a base handler that swallowed
                # the error into search_response=None) is not the same as a genuine empty —
                # a provider that errors is unreliable, one that finds nothing is just weak.
                raw = res.raw if (res is not None and isinstance(res.raw, dict)) else None
                if raw is not None and ("error" in raw or
                                        ("search_response" in raw and raw.get("search_response") is None)):
                    pv["errors"] += 1
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
            reader_span = trace.child("reader.synthesize",
                                      input={"prompt": reader_mod.build_reader_prompt(q.query, capped, run_nonce)})
            ans = reader_mod.synthesize(reader_llm, q.query, capped, run_nonce)
            reader_span.end(output={"answer": ans})
            pv["reader_made"] += 1
            if audit is not None:
                audit["providers"][name]["evidence"] = [
                    {"url": d.url, "title": d.title, "content": d.content} for d in capped]
                audit["providers"][name]["reader_answer"] = ans
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
            # Self-preference mitigation (§5), route XOR flag:
            #  - with a secondary judge configured, a Claude-native pair ROUTES to it;
            #  - with none, ``self_preference_label`` flags it ``possible-self-preference``.
            # In M0/M1 synthesis is forced, so no pair is native and both stay no-op guards.
            x_native, y_native = x in native_providers, y in native_providers
            sp_label = self_preference_label(
                x, y, x_native, y_native, claude_family,
                judge_is_claude, has_secondary_judge)
            # Route ONLY a Claude-family native pair — a future non-Claude native provider shares
            # the native path but carries no self-preference risk, so it must not be routed.
            route = route_native_self_preference(
                x_native and x in claude_family, y_native and y in claude_family,
                secondary_judge_llm is not None)
            judge_span = trace.child("judge.compare", input={"query": q.query, "a": x, "b": y})
            verdict = judge_pair(judge_llm, q.query, answers[x], answers[y], run_nonce,
                                 order_swap=config.order_swap, exclude_on_flip=config.exclude_on_flip,
                                 secondary_llm=secondary_judge_llm, route_to_secondary=route,
                                 self_preference=sp_label)
            # If routing was intended but the secondary judge was unavailable, judge_pair fell back
            # to primary. The pair was then neither routed nor flagged — restore the flag so the
            # route-XOR-flag invariant holds (§5): recompute the label as if no secondary existed.
            if route and verdict.get("decided_by") == "primary":
                verdict["self_preference"] = self_preference_label(
                    x, y, x_native, y_native, claude_family, judge_is_claude,
                    has_secondary_judge=False)
            judge_span.end(output={"outcome": verdict["outcome"], "flipped": verdict["flipped"],
                                   "rationales": verdict["rationales"]})
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
            # Record paired per-judge labels (only when both judges decided the same pair) so κ /
            # reliability weighting can be computed downstream. Labels map "x"/"y" -> provider.
            labels = verdict.get("judge_labels") or {}
            if secondary_judge_llm is not None and labels.get("secondary") is not None \
                    and labels.get("primary") is not None:
                loc["judge_labels"].append({
                    "a": x, "b": y,
                    "primary": _lab(labels["primary"], x, y),
                    "secondary": _lab(labels["secondary"], x, y),
                })
            cx, cy = correct.get(x), correct.get(y)
            if cx is not None and cy is not None and cx != cy:
                if winner in (x, y):
                    loc["cal_decidable"] += 1
                    loc["cal_agree"] += int(winner == (x if cx else y))
                else:
                    loc["cal_abstained"] += 1
            loc["comparisons"].append({"a": x, "b": y, "winner": winner, "category": q.category,
                                       "repeat": qi // base_n,
                                       "decided_by": verdict.get("decided_by", "primary")})
            loc["rationale"].append({
                "query": q.query, "a": x, "b": y, "winner": winner,
                "flipped": verdict["flipped"], "low_confidence": verdict["low_confidence"],
                "injection_flag": verdict["injection_flag"], "rationales": verdict["rationales"],
                "self_preference": verdict.get("self_preference"),
            })
        trace.end()
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
    traces: List[dict] = []
    per_query_answers: List[Dict[str, Optional[str]]] = []   # Tier-1 consensus, ordered by qi
    per_query_comparisons: List[List[dict]] = []
    paired_labels: List[dict] = []
    swap_flips = swap_total = judge_skipped = injection_flags = self_pref_flags = 0
    cal_agree = cal_decidable = cal_abstained = 0
    for loc in sorted(locals_out, key=lambda local: local["qi"]):
        comparisons.extend(loc["comparisons"])
        rationale_log.extend(loc["rationale"])
        if loc.get("audit") is not None:
            traces.append(loc["audit"])
        per_query_answers.append(loc["reader_answers"])
        per_query_comparisons.append(loc["comparisons"])
        paired_labels.extend(loc["judge_labels"])
        swap_total += loc["swap_total"]; swap_flips += loc["swap_flips"]
        judge_skipped += loc["judge_skipped"]; injection_flags += loc["injection_flags"]
        self_pref_flags += loc["self_pref_flags"]
        cal_agree += loc["cal_agree"]; cal_decidable += loc["cal_decidable"]; cal_abstained += loc["cal_abstained"]
        for p in provider_names:
            pv = loc["prov"][p]
            latencies[p].extend(pv["latency"]); coverage_tokens[p].extend(pv["coverage"])
            cells_attempted[p] += pv["cells_att"]; cells_succeeded[p] += pv["cells_succ"]
            empty_evidence_count[p] += pv["empty"]; error_count[p] += pv["errors"]
            reader_answers_made[p] += pv["reader_made"]
            reader_degenerate_count[p] += pv["reader_degen"]
            acc_correct[p] += pv["acc_correct"]; acc_total[p] += pv["acc_total"]
            if pv["units"] is not None:
                cost_units[p] = (cost_units[p] or 0.0) + pv["units"]
                cost_unit_cells[p] += pv["units_cells"]
            freshness_tallies[p].extend(pv["freshness"])
            mv_correct[p] += pv["mv_correct"]; mv_total[p] += pv["mv_total"]

    tracer.flush()  # no-op for NullTracer; sends buffered spans for LangfuseTracer

    # ---- Inter-judge agreement (κ, §6.4) + judge-reliability weighting (§6.3) ----
    # A secondary judge yields κ (reported regardless). Reliability weighting engages only when a
    # signal actually DISCRIMINATES between judges: with just primary+secondary, cross-agreement is
    # symmetric (it says how MUCH they agree, not WHICH is right), so ``judge_weights`` returns {}
    # and BT stays UNWEIGHTED — the spec-correct default (weighting needs 3+ judges or gold, §6.3).
    # A configured secondary that produced NO paired labels is a failure surfaced in stage_status.
    inter_judge_kappa = None
    secondary_failed = bool(secondary_judge_llm is not None and not paired_labels)
    if secondary_judge_llm is not None and paired_labels:
        prim = [pl["primary"] for pl in paired_labels]
        sec = [pl["secondary"] for pl in paired_labels]
        inter_judge_kappa = cohens_kappa(prim, sec)
        per_judge = {"primary": prim, "secondary": sec}
        weights = judge_weights(consensus_agreements(per_judge),
                                mode=config.judge_reliability_weighting)
        if weights:
            for c in comparisons:
                w = weights.get(c.get("decided_by", "primary"))
                if w is not None:
                    c["weight"] = w

    agg = aggregate(comparisons, provider_names, seed=0, method=config.aggregation_method)
    anchors = compute_anchors(per_query_answers, per_query_comparisons, provider_names,
                              min_providers=config.consensus_min_providers)
    for p in provider_names:
        anchors.auto_verify[p] = {"correct": mv_correct[p], "total": mv_total[p]}

    # Per-repeat win-rate spread: the visible noise floor for this workload. Only computed when
    # repeats ran; a spread wider than the CI gap between two providers means "don't trust the
    # order between them yet".
    repeats_block: dict = {"n": repeats}
    if repeats > 1:
        per_repeat = {r: point_winrates([c for c in comparisons if c.get("repeat") == r],
                                        provider_names)
                      for r in range(repeats)}
        repeats_block["per_repeat_win_rates"] = {str(r): wr for r, wr in per_repeat.items()}
        repeats_block["win_rate_spread"] = {
            p: (max(vals) - min(vals) if (vals := [per_repeat[r][p] for r in range(repeats)
                                                   if per_repeat[r][p] is not None]) else None)
            for p in provider_names}

    # Per-category rankings (§8 use-case segmentation): "best" is undefined without a job, so
    # when the queries file tags rows with `category`, each slice is re-ranked with the same
    # aggregation. Absent categories -> empty dict, nothing rendered.
    per_category = {
        cat: {"ranking": [_score_dict(s) for s in cat_agg.scores],
              "tie_groups": cat_agg.tie_groups,
              "n_decided_comparisons": cat_agg.n_decided,
              "n_excluded_comparisons": cat_agg.n_excluded}
        for cat, cat_agg in per_category_rankings(comparisons, provider_names, seed=0,
                                                  method=config.aggregation_method).items()
    }

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
            "reliability": {"success_rate": _rate(cells_succeeded[p], cells_attempted[p]),
                            "error_rate": _rate(error_count[p], cells_attempted[p]),
                            "errors": error_count[p]},
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
    attach_cost_per_success(per_provider)  # $/correct-answer where anchors exist (§8)
    # Drop the cost weight and renormalize the rest when cost is blank for the run (§8).
    weights_effective = effective_weights(config.weights, per_provider) if config.weights else {}

    swap_consistency = 1.0 - (swap_flips / swap_total) if swap_total else None
    n_ranked = sum(1 for s in agg.scores if s.status == "ranked")

    weighted = any("weight" in c for c in comparisons)
    stage_status = {
        "secrets": _ok(len(scope.included) > 0, f"{len(scope.included)} providers included"),
        "adapters": _ok(any(per_provider[p]["cells_succeeded"] > 0 for p in provider_names),
                        "evidence returned by ≥1 provider"),
        "reader": _ok(any((per_provider[p]["reader_degenerate_rate"] or 0) < 1.0
                          for p in provider_names if per_provider[p]["cells_succeeded"]),
                      "reader produced usable answers"),
        "judge": _judge_status(swap_consistency, injection_flags, judge_skipped,
                               kappa=inter_judge_kappa, secondary_failed=secondary_failed),
        "aggregate": _ok(n_ranked >= 1, f"{n_ranked} providers ranked ({agg.method}"
                         + (", reliability-weighted" if weighted else "")
                         + f"), {agg.n_excluded} comparisons excluded"),
        "pipeline": _ok(True, "run completed"),
    }

    seen, cost = set(), 0.0
    for c in (judge_llm, reader_llm, grader_llm, secondary_judge_llm):
        if c is not None and id(c) not in seen and hasattr(c, "cost_usd"):
            cost += c.cost_usd()
            seen.add(id(c))

    return {
        "run_nonce": run_nonce,
        "providers": provider_names,
        "scope": scope.as_dict(),
        "ranking": [_score_dict(s) for s in agg.scores],
        "tie_groups": agg.tie_groups,
        "per_category": per_category,
        "repeats": repeats_block,
        "n_decided_comparisons": agg.n_decided,
        "n_excluded_comparisons": agg.n_excluded,
        "metrics": per_provider,
        "weights_effective": weights_effective,
        "aggregation_method": agg.method,
        "reliability_weighted": weighted,
        "judge": {"swap_consistency": swap_consistency, "swap_total": swap_total,
                  "swap_flips": swap_flips, "judge_skipped": judge_skipped,
                  "injection_flags": injection_flags, "inter_judge_kappa": inter_judge_kappa,
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
        "traces": traces if config.save_traces else None,
        "n_queries": base_n,
    }


def _lab(outcome, x, y):
    """Map a judge outcome ("x"/"y"/"tie") to a provider-name label for κ / weighting."""
    return x if outcome == "x" else y if outcome == "y" else "tie"


def _rate(num: int, denom: int):
    return (num / denom) if denom else None


def _cost(client) -> float:
    return client.cost_usd() if hasattr(client, "cost_usd") else 0.0


def _ok(healthy: bool, reason: str) -> dict:
    return {"status": "green" if healthy else "red", "reason": reason}


def _judge_status(swap_consistency, injection_flags, judge_skipped=0, bar: float = 0.85,
                  kappa=None, kappa_bar: float = 0.6, secondary_failed: bool = False) -> dict:
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
    if kappa is not None:  # §6.4: inter-judge agreement, reported regardless of value
        reason += f", inter-judge κ {kappa:.2f}"
        healthy = healthy and kappa >= kappa_bar
    if secondary_failed:  # configured secondary judge produced no paired verdicts -> broken path
        reason += ", secondary judge produced no verdicts (check secondary model id / API access)"
        healthy = False
    return {"status": "green" if healthy else "red", "reason": reason}


def _score_dict(s) -> dict:
    return {"provider": s.provider, "win_rate": s.win_rate, "ci_low": s.ci_low,
            "ci_high": s.ci_high, "n_comparisons": s.n_comparisons, "status": s.status,
            "rank": s.rank, "tie_group": s.tie_group}
