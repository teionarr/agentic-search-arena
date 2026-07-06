"""Reporting: one canonical results.json (source of truth) + rendered table / CSV / CLI.

Security: a single ``redact()`` boundary scrubs any resolved secret value and drops
key-bearing fields from the persisted ``raw`` payloads; CSV cells are neutralized against
formula injection.
"""

import csv
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

RESULTS_SCHEMA_VERSION = "1.0"  # frozen canonical schema; table/CSV/stage_status render from it
_KEY_FIELDS = {"api_key", "authorization", "x-api-key", "x-subscription-token", "token"}


def _secret_values() -> List[str]:
    """Current resolved secret values (env), for scrubbing. Never persisted themselves."""
    vals = []
    for k, v in os.environ.items():
        if v and (k.endswith("_API_KEY") or k in ("OPENAI_API_KEY",)):
            vals.append(v)
    return vals


def redact(obj: Any, secret_values: List[str] = None) -> Any:
    """Recursively scrub secret values and drop key-bearing fields. Applied before persist."""
    secret_values = secret_values if secret_values is not None else _secret_values()

    def scrub_str(s: str) -> str:
        for sv in secret_values:
            if sv and sv in s:
                s = s.replace(sv, "«redacted»")
        return s

    if isinstance(obj, str):
        return scrub_str(obj)
    if isinstance(obj, dict):
        return {k: ("«redacted»" if k.lower() in _KEY_FIELDS else redact(v, secret_values))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, secret_values) for v in obj]
    return obj


def query_set_hash(queries: List[str]) -> str:
    h = hashlib.sha256()
    for q in queries:
        h.update(q.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _csv_safe(value: Any) -> str:
    """Neutralize CSV formula injection: prefix cells starting with = + - @ (or tab/CR)."""
    s = "" if value is None else str(value)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def build_document(result: dict, queries: List[str], config_snapshot: dict,
                   model_id: str) -> dict:
    """Assemble the canonical results.json document from the pipeline result."""
    doc = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "query_set_hash": query_set_hash(queries),
        "n_queries": result.get("n_queries"),
        "model_id": model_id,
        "config": config_snapshot,
        "scope": result["scope"],
        "degenerate_run": result["degenerate_run"],
        "calibration": result.get("calibration"),
        "ranking": result["ranking"],
        "tie_groups": result["tie_groups"],
        "metrics": result["metrics"],
        "judge": result["judge"],
        "stage_status": result["stage_status"],
        "n_decided_comparisons": result["n_decided_comparisons"],
        "n_excluded_comparisons": result["n_excluded_comparisons"],
        "cost_usd": result.get("cost_usd"),
        "rationale_log": result["rationale_log"],
    }
    return redact(doc)


def write_results(doc: dict, output_dir: str) -> Dict[str, str]:
    """Write results.json (source of truth) + ranking.csv (rendered from it)."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)

    csv_path = os.path.join(output_dir, "ranking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "provider", "win_rate", "ci_low", "ci_high", "n_comparisons",
                    "tie_group", "status", "avg_tokens_per_result", "latency_p50_ms",
                    "accuracy_rate", "accuracy_correct", "accuracy_total",
                    "freshness_score", "freshness_coverage", "freshness_low_confidence"])
        for s in doc["ranking"]:
            m = doc["metrics"].get(s["provider"], {})
            acc = m.get("accuracy", {}) or {}
            fr = m.get("freshness", {}) or {}
            w.writerow([_csv_safe(x) for x in [
                s["rank"], s["provider"], s["win_rate"], s["ci_low"], s["ci_high"],
                s["n_comparisons"], s["tie_group"], s["status"],
                m.get("coverage", {}).get("avg_tokens_per_result"),
                m.get("latency", {}).get("p50"),
                acc.get("rate"), acc.get("correct"), acc.get("total"),
                fr.get("score"), fr.get("coverage"),
                fr.get("low_confidence") if fr else None,
            ]])

    return {"json": json_path, "csv": csv_path}


def _winrate_bar(win_rate, width: int = 22) -> str:
    """An ASCII win-rate bar (0.0–1.0) with a marker at the 0.50 'even' line."""
    if win_rate is None:
        return " " * width
    n = max(0, min(width, round(win_rate * width)))
    chars = ["█"] * n + ["░"] * (width - n)
    mid = min(width - 1, round(0.5 * width))
    chars[mid] = "┃" if chars[mid] == "█" else "│"  # 0.50 marker, visible on filled or empty
    return "".join(chars)


def render_cli_summary(doc: dict) -> str:
    """A terminal dashboard: ranking bars + scope + per-stage health. All CLI, no artifacts."""
    W = 56
    ranked = [s for s in doc["ranking"] if s["status"] == "ranked"]
    group_sizes: Dict[int, int] = {}
    for s in ranked:
        group_sizes[s["tie_group"]] = group_sizes.get(s["tie_group"], 0) + 1

    out = ["", "═" * W, "  SEARCH ARENA".ljust(W), "═" * W]
    if doc.get("degenerate_run"):
        out.append("  ⚠  degenerate run (<3 providers): ranking not statistically meaningful")

    cost = doc.get("cost_usd")
    cost_s = f" · cost ${cost:.2f}" if cost is not None else ""
    sc = (doc.get("judge") or {}).get("swap_consistency")
    n_dec = doc.get("n_decided_comparisons", 0)
    n_tot = n_dec + doc.get("n_excluded_comparisons", 0)
    out.append(f"  {doc['n_queries']} queries · judge {doc['model_id']}{cost_s}")
    out.append(f"  {n_dec}/{n_tot} comparisons used" +
               (f" · judge reliability {sc:.2f}" if sc is not None else ""))
    cal = doc.get("calibration") or {}
    if cal.get("agreement") is not None:
        bar = "≥0.80 ok" if cal["agreement"] >= 0.80 else "below 0.80 bar"
        out.append(f"  judge-vs-gold agreement {cal['agreement']:.0%} "
                   f"({cal['n_decidable']} decidable pairs · {bar})")

    has_acc = any((doc["metrics"].get(s["provider"], {}).get("accuracy", {}) or {}).get("rate") is not None
                  for s in doc["ranking"])
    has_fresh = any((doc["metrics"].get(s["provider"], {}).get("freshness", {}) or {}).get("score") is not None
                    for s in doc["ranking"])
    acc_hdr = " · acc = judge-free accuracy vs gold" if has_acc else ""
    fresh_hdr = " · fresh = dated-in-window share (cov = date coverage; ! = low-confidence)" if has_fresh else ""
    out.append("")
    out.append("  RANKING   bar = win-rate · │ = 0.50 even line · [ ] = 95% CI · cov = avg tok/result" + acc_hdr + fresh_hdr)
    out.append("  " + "─" * (W - 2))
    for s in doc["ranking"]:
        acc = (doc["metrics"].get(s["provider"], {}).get("accuracy", {}) or {})
        acc_s = f"  acc {acc['rate']:.0%} ({acc['correct']}/{acc['total']})" if acc.get("rate") is not None else ""
        fr = (doc["metrics"].get(s["provider"], {}).get("freshness", {}) or {})
        fresh_s = (f"  fresh {fr['score']:.0%} (cov {fr['coverage']:.0%}{' !' if fr.get('low_confidence') else ''})"
                   if fr.get("score") is not None else "")
        if s["status"] == "unranked":
            out.append(f"   --  {s['provider']:<18}  unranked — insufficient valid comparisons{acc_s}{fresh_s}")
            continue
        bar = _winrate_bar(s["win_rate"])
        ci = f"[{s['ci_low']:.2f}–{s['ci_high']:.2f}]"
        cov = (doc["metrics"].get(s["provider"], {}).get("coverage", {}) or {}).get("avg_tokens_per_result")
        cov_s = f"cov {cov:.0f}" if cov is not None else "cov n/a"
        if group_sizes.get(s["tie_group"], 0) == 1:
            tag = "  ← clear leader"
        elif group_sizes.get(s["tie_group"], 0) > 1:
            tag = "  · tied"
        else:
            tag = ""
        out.append(f"  #{s['rank']} {s['provider']:<18} {bar} {s['win_rate']:.2f} {ci:<13} {cov_s}{acc_s}{fresh_s}{tag}")

    out.append("")
    out.append("  SCOPE")
    for prov, info in doc["scope"].items():
        mark = "✓" if info["status"] == "included" else "·"
        detail = f"  ({info['detail']})" if info["detail"] else ""
        out.append(f"   {mark} {prov:<18} {info['status']}{detail}")

    out.append("")
    out.append("  STAGE STATUS")
    for stage, st in doc["stage_status"].items():
        mark = "✅" if st["status"] == "green" else "❌"
        out.append(f"   {mark} {stage:<11} {st['reason']}")
    out.append("═" * W + "\n")
    return "\n".join(out)
