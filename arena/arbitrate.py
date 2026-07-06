"""Tier-2 human adjudication (M4, §3): arbitrate ONLY the pivotal ties.

Surfaces the comparisons where providers are statistically tied AND the judge's verdict was
excluded (order-swap flip) or low-confidence — the disagreements that could actually flip the
ranking. Never a full gold set: the queue is capped (§16 effort budget ~15–60 min) and says
what it left out. Human verdicts are appended as high-weight comparisons and the ranking is
re-aggregated, printed before/after.

Requires a run made with ``--save-traces`` (the blinded answers shown to the human come from
the audit traces; results.json alone doesn't carry them).

    python -m arena.arbitrate results/arena/<timestamp>/
"""

import argparse
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITEMS = 30     # §16: arbitrating 10–30 ties is the whole human budget
DEFAULT_HUMAN_WEIGHT = 5.0  # one human verdict outweighs one judge comparison 5:1


# ---- pure selection / re-aggregation (Tier-A testable, no IO) ----

def comparisons_from_rationale(rationale_log: List[dict]) -> List[dict]:
    """Rebuild the aggregation input from the persisted per-decision log.

    ``winner`` is None for excluded (swap-flip/skip) verdicts — aggregate() already ignores
    those, so the rebuilt list reproduces the original ranking."""
    return [{"a": r["a"], "b": r["b"], "winner": r["winner"]} for r in rationale_log]


def select_pivotal(doc: dict, max_items: int = DEFAULT_MAX_ITEMS) -> Tuple[List[dict], dict]:
    """Pick the comparisons a human should arbitrate: statistically-tied provider pairs ×
    (excluded or low-confidence) verdicts, closest win-rates first.

    Returns ``(items, meta)`` where meta reports the tied pairs considered and how many
    candidates were left out by the cap (no silent truncation)."""
    ranked = [s for s in doc.get("ranking", []) if s.get("status") == "ranked"]
    wr = {s["provider"]: s["win_rate"] for s in ranked}

    tied_pairs: List[Tuple[str, str]] = []
    for group in doc.get("tie_groups", []):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                tied_pairs.append((group[i], group[j]))
    tied_pairs.sort(key=lambda ab: abs((wr.get(ab[0]) or 0) - (wr.get(ab[1]) or 0)))

    candidates: List[dict] = []
    for a, b in tied_pairs:
        for r in doc.get("rationale_log", []):
            if {r["a"], r["b"]} == {a, b} and (r["winner"] is None or r.get("low_confidence")):
                candidates.append(r)

    items = candidates[:max_items]
    meta = {"tied_pairs": tied_pairs, "n_candidates": len(candidates),
            "n_selected": len(items), "n_left_out": len(candidates) - len(items)}
    return items, meta


def blind_order(query: str, a: str, b: str) -> Tuple[str, str]:
    """Deterministic per-item blinding: which provider is shown as 'Answer 1'.

    Seeded by content hash (not Python's randomized ``hash``) so a re-run shows the same
    order — reproducible, but uncorrelated with provider identity across items."""
    digest = hashlib.sha256(f"{query}\x00{a}\x00{b}".encode()).digest()
    return (a, b) if digest[0] % 2 == 0 else (b, a)


def reapply(base_comparisons: List[dict], adjudications: List[dict], providers: List[str],
            weight: float = DEFAULT_HUMAN_WEIGHT, method: str = "bradley_terry"):
    """Re-aggregate with human verdicts appended as high-weight comparisons."""
    from arena.aggregate import aggregate
    comps = list(base_comparisons)
    for adj in adjudications:
        if adj.get("winner") is None:  # human skipped -> contributes nothing
            continue
        comps.append({"a": adj["a"], "b": adj["b"], "winner": adj["winner"], "weight": weight})
    return aggregate(comps, providers, seed=0, method=method)


def render_rankings(before, after) -> str:
    """Compact before/after table (provider, rank, win-rate scale)."""
    W = 56
    b_by = {s.provider: s for s in before.scores}
    out = ["", "═" * W, "  ARBITRATION — ranking before → after", "═" * W]
    for s in after.scores:
        prev = b_by.get(s.provider)
        b_rank = prev.rank if prev and prev.rank else "—"
        a_rank = s.rank if s.rank else "—"
        moved = "  ←" if b_rank != a_rank else ""
        wr = f"{s.win_rate:.2f}" if s.win_rate is not None else "  — "
        out.append(f"  {s.provider:<18} #{b_rank} → #{a_rank}   {wr}{moved}")
    out.append("═" * W + "\n")
    return "\n".join(out)


# ---- IO / interactive loop ----

def load_run(run_dir: str) -> Tuple[dict, Dict[str, Dict[str, Optional[str]]]]:
    """Load results.json + the per-query reader answers from the audit traces."""
    with open(os.path.join(run_dir, "results.json")) as f:
        doc = json.load(f)
    trace_dir = os.path.join(run_dir, "traces")
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(
            f"{trace_dir} not found — arbitration shows the blinded answers from the audit "
            "traces. Re-run the arena with --save-traces first.")
    answers: Dict[str, Dict[str, Optional[str]]] = {}
    for name in sorted(os.listdir(trace_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(trace_dir, name)) as f:
            tr = json.load(f)
        answers[tr["query"]] = {p: (e or {}).get("reader_answer")
                                for p, e in (tr.get("providers") or {}).items()}
    return doc, answers


def arbitrate_interactively(items: List[dict], answers: Dict[str, Dict[str, Optional[str]]],
                            ask=input, echo=print, on_verdict=None) -> List[dict]:
    """The prompt loop. ``ask``/``echo`` injectable for tests. Verdicts: 1 / 2 / t(ie) /
    s(kip) / q(uit). ``on_verdict`` fires per recorded verdict so the session can persist
    incrementally — a §16 session is up to an hour of human effort; a Ctrl-C must not
    discard it."""
    adjudications = []
    for i, item in enumerate(items, 1):
        q, a, b = item["query"], item["a"], item["b"]
        first, second = blind_order(q, a, b)
        ans_map = answers.get(q, {})
        echo(f"\n[{i}/{len(items)}] QUERY: {q}\n")
        echo(f"--- Answer 1 ---\n{ans_map.get(first) or '(no answer recorded)'}\n")
        echo(f"--- Answer 2 ---\n{ans_map.get(second) or '(no answer recorded)'}\n")
        while True:
            v = ask("Which answer is better-supported? [1/2/t=tie/s=skip/q=quit] ").strip().lower()
            if v in ("1", "2", "t", "s", "q"):
                break
            echo("  enter 1, 2, t, s or q")
        if v == "q":
            break
        winner = {"1": first, "2": second, "t": "tie", "s": None}[v]
        adj = {"query": q, "a": a, "b": b, "winner": winner, "shown_first": first}
        adjudications.append(adj)
        if on_verdict:
            on_verdict(adj)
    return adjudications


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Arbitrate the pivotal ties of an arena run (M4)")
    parser.add_argument("run_dir", help="Run directory containing results.json + traces/")
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_ITEMS,
                        help="Cap on items to arbitrate (§16 effort budget)")
    parser.add_argument("--weight", type=float, default=DEFAULT_HUMAN_WEIGHT,
                        help="Weight of one human verdict vs one judge comparison")
    args = parser.parse_args()

    doc, answers = load_run(args.run_dir)
    items, meta = select_pivotal(doc, max_items=args.max)
    if not items:
        print("No pivotal ties to arbitrate — the ranking has no statistically-tied pair "
              "with excluded/low-confidence verdicts. Nothing for a human to add.")
        return 0
    print(f"{meta['n_selected']} pivotal comparison(s) selected "
          f"(from {meta['n_candidates']} candidates across "
          f"{len(meta['tied_pairs'])} tied pair(s)"
          + (f"; {meta['n_left_out']} left out by --max {args.max}" if meta["n_left_out"] else "")
          + ")")

    # Verdicts append to disk AS they are recorded — an interrupted session keeps its progress.
    adj_path = os.path.join(args.run_dir, "adjudications.jsonl")
    with open(adj_path, "a") as f:
        def _persist(adj):
            f.write(json.dumps(adj) + "\n")
            f.flush()
        adjudications = arbitrate_interactively(items, answers, on_verdict=_persist)
    if not adjudications:
        print("No verdicts recorded.")
        return 0
    print(f"Wrote {len(adjudications)} verdict(s) to {adj_path}")

    providers = [s["provider"] for s in doc["ranking"]]
    base = comparisons_from_rationale(doc["rationale_log"])
    method = doc.get("aggregation_method", "bradley_terry")
    from arena.aggregate import aggregate
    before = aggregate(base, providers, seed=0, method=method)
    after = reapply(base, adjudications, providers, weight=args.weight, method=method)
    print(render_rankings(before, after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
