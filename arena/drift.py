"""Run-over-run drift: diff two canonical results.json documents.

Providers ship changes weekly, so a ranking is a dated snapshot, not a fact — this makes the
drift *visible* (freshness / continuous-eval requirement). Pure functions over the canonical
artifact; no network, no model calls. A rank move is only called out as significant when the
win-rate CIs of the two runs don't overlap — otherwise it's within this workload's noise.
"""

from typing import List, Optional

COMPARABILITY_FIELDS = ("query_set_hash", "model_id")


def _by_provider(doc: dict) -> dict:
    return {s["provider"]: s for s in doc.get("ranking", [])}


def _ci_shifted(before: dict, after: dict) -> Optional[bool]:
    """True when the two runs' win-rate CIs don't overlap (a shift beyond noise); None if
    either side is unranked/missing a CI."""
    b_lo, b_hi = before.get("ci_low"), before.get("ci_high")
    a_lo, a_hi = after.get("ci_low"), after.get("ci_high")
    if None in (b_lo, b_hi, a_lo, a_hi):
        return None
    return b_hi < a_lo or a_hi < b_lo


def diff_runs(before: dict, after: dict) -> dict:
    """Diff two results.json documents into a drift report (pure; deterministic)."""
    b_scores, a_scores = _by_provider(before), _by_provider(after)
    all_providers = sorted(set(b_scores) | set(a_scores))

    providers = {}
    for p in sorted(set(b_scores) & set(a_scores)):
        b, a = b_scores[p], a_scores[p]
        rank_delta = (b["rank"] - a["rank"]) if (b.get("rank") and a.get("rank")) else None
        wr_delta = (round(a["win_rate"] - b["win_rate"], 4)
                    if (b.get("win_rate") is not None and a.get("win_rate") is not None) else None)
        providers[p] = {
            "rank_before": b.get("rank"), "rank_after": a.get("rank"),
            "rank_delta": rank_delta,          # positive = moved up
            "win_rate_before": b.get("win_rate"), "win_rate_after": a.get("win_rate"),
            "win_rate_delta": wr_delta,
            "status_before": b.get("status"), "status_after": a.get("status"),
            "shifted_beyond_ci": _ci_shifted(b, a),
        }

    comparable = {f: (before.get(f) == after.get(f)) for f in COMPARABILITY_FIELDS}
    return {
        "before": {"timestamp": before.get("timestamp"), "model_id": before.get("model_id"),
                   "query_set_hash": before.get("query_set_hash")},
        "after": {"timestamp": after.get("timestamp"), "model_id": after.get("model_id"),
                  "query_set_hash": after.get("query_set_hash")},
        "comparable": comparable,
        "providers": providers,
        "added": sorted(set(a_scores) - set(b_scores)),
        "removed": sorted(set(b_scores) - set(a_scores)),
        "rank_changes": [p for p in providers if providers[p]["rank_delta"] not in (None, 0)],
        "n_providers": len(all_providers),
    }


def render_drift(diff: dict) -> str:
    """Changelog-style text for the terminal (or a CHANGELOG entry)."""
    W = 64
    b, a = diff["before"], diff["after"]
    out = ["", "═" * W, "  SEARCH ARENA DRIFT", "═" * W,
           f"  {b['timestamp']}  →  {a['timestamp']}"]
    if not diff["comparable"]["query_set_hash"]:
        out.append("  ⚠  different query sets — deltas are NOT apples-to-apples")
    if not diff["comparable"]["model_id"]:
        out.append(f"  ⚠  different judge models ({b['model_id']} → {a['model_id']}) — "
                   "deltas confound provider drift with judge drift")

    out.append("")
    for p, d in sorted(diff["providers"].items(),
                       key=lambda kv: abs(kv[1]["win_rate_delta"] or 0), reverse=True):
        if d["rank_before"] is None or d["rank_after"] is None:
            out.append(f"  {p:<18} {d['status_before']} → {d['status_after']}")
            continue
        arrow = ("↑" if d["rank_delta"] > 0 else "↓" if d["rank_delta"] < 0 else "=")
        move = f"#{d['rank_before']} → #{d['rank_after']} ({arrow}{abs(d['rank_delta']) or ''})"
        wr = (f"win-rate {d['win_rate_before']:.2f} → {d['win_rate_after']:.2f} "
              f"({d['win_rate_delta']:+.2f})") if d["win_rate_delta"] is not None else ""
        flag = "  ⚠ beyond CI overlap" if d["shifted_beyond_ci"] else ""
        out.append(f"  {p:<18} {move:<16} {wr}{flag}")

    if diff["added"] or diff["removed"]:
        out.append("")
        if diff["added"]:
            out.append(f"  added:   {', '.join(diff['added'])}")
        if diff["removed"]:
            out.append(f"  removed: {', '.join(diff['removed'])}")
    if not diff["rank_changes"]:
        out.append("")
        out.append("  no rank changes")
    out.append("═" * W + "\n")
    return "\n".join(out)
