"""Tier-1 free anchors (§3 Tier 1, §8) — deterministic, no AI, no network.

Two mechanisms, both pure functions over the reader answers already produced upstream:

1. **Consensus silver labels.** Where ``>= min_providers`` INDEPENDENT providers' reader-answers
   converge on the same NORMALIZED answer for a query, that normalized answer is adopted as a
   provisional silver label. We cross-check the arena on this easy portion: report consensus
   COVERAGE (share of queries that reached consensus) and arena-vs-consensus AGREEMENT (does the
   arena winner of a pair agree with the consensus label?).

2. **Machine-verifiable checks.** Where a query carries an ``expected_answer`` that is
   mechanically checkable (a number, a date, or a name/string) — or the query asks "does the
   evidence contain string X" — the answer is auto-verified DETERMINISTICALLY with no model.
   The result feeds the existing accuracy-anchor path (``arena.metrics`` accuracy column) as a
   FREE anchor, so accuracy is available without ``OPENAI_API_KEY``.

Never fabricated: consensus forms only at genuine >= N convergence; machine-verify returns
``None`` (blank, unanchored) whenever the expected answer is not mechanically checkable.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from arena.metrics import accuracy_dict

logger = logging.getLogger(__name__)

DEFAULT_MIN_PROVIDERS = 3  # §3 Tier 1 default: >= N independent providers must converge

# Month names -> zero-padded number, for date canonicalization.
_MONTHS = {
    "january": "01", "jan": "01", "february": "02", "feb": "02", "march": "03", "mar": "03",
    "april": "04", "apr": "04", "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
    "august": "08", "aug": "08", "september": "09", "sep": "09", "sept": "09", "october": "10",
    "oct": "10", "november": "11", "nov": "11", "december": "12", "dec": "12",
}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# "January 5, 2020" / "Jan 5 2020" / "5 January 2020"
_MONTH_DAY_YEAR = re.compile(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?,?\s+(\d{4})\b")
# 12/31/2020 or 31/12/2020 -> we keep the written order as (a, b, year); only used when
# unambiguous (a > 12 => a is the day). Ambiguous numeric dates are left as-is.
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _canon_dates(text: str) -> str:
    """Rewrite recognizable dates to a canonical ``YYYY-MM-DD`` token so equivalent spellings
    (``Jan 5 2020`` == ``2020-01-05``) normalize to the same string. Unrecognized dates are left
    untouched."""

    def iso(y: str, m: str, d: str) -> str:
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    def sub_mdy(match: "re.Match") -> str:
        mon = _MONTHS.get(match.group(1).lower())
        return iso(match.group(3), mon, match.group(2)) if mon else match.group(0)

    def sub_dmy(match: "re.Match") -> str:
        mon = _MONTHS.get(match.group(2).lower())
        return iso(match.group(3), mon, match.group(1)) if mon else match.group(0)

    def sub_iso(match: "re.Match") -> str:
        return iso(match.group(1), match.group(2), match.group(3))

    def sub_numeric(match: "re.Match") -> str:
        a, b, y = match.group(1), match.group(2), match.group(3)
        # Only canonicalize when order is unambiguous (a value > 12 fixes day vs month).
        if int(a) > 12 >= int(b):
            return iso(y, b, a)   # DD/MM/YYYY
        if int(b) > 12 >= int(a):
            return iso(y, a, b)   # MM/DD/YYYY
        return match.group(0)     # ambiguous -> leave as-is

    text = _MONTH_DAY_YEAR.sub(sub_mdy, text)
    text = _DAY_MONTH_YEAR.sub(sub_dmy, text)
    text = _NUMERIC_DATE.sub(sub_numeric, text)
    text = _ISO_DATE.sub(sub_iso, text)
    return text


def _canon_numbers(text: str) -> str:
    """Strip thousands separators so ``1,000`` == ``1000`` and drop trailing ``.0``."""

    def sub(match: "re.Match") -> str:
        raw = match.group(0).replace(",", "")
        try:
            f = float(raw)
        except ValueError:
            return match.group(0)
        return str(int(f)) if f == int(f) else str(f)

    return _NUMBER.sub(sub, text)


def normalize_answer(answer: Optional[str]) -> str:
    """Canonicalize an answer for equality comparison: lowercase, trim, strip punctuation, and
    canonicalize numbers/dates. Returns ``""`` for a missing answer."""
    if not answer:
        return ""
    s = answer.strip().lower()
    s = _canon_dates(s)          # before punctuation strip (dates use '-' and '/')
    s = _canon_numbers(s)
    s = _PUNCT.sub(" ", s)       # drop remaining punctuation
    s = _WS.sub(" ", s).strip()  # collapse whitespace
    return s


def consensus_label(answers_by_provider: Dict[str, Optional[str]],
                    min_providers: int = DEFAULT_MIN_PROVIDERS) -> Optional[str]:
    """The normalized answer on which ``>= min_providers`` INDEPENDENT providers converge, else
    ``None``. Each provider votes at most once (its own reader answer); empties don't vote."""
    counts: Dict[str, int] = {}
    for ans in answers_by_provider.values():
        norm = normalize_answer(ans)
        if norm:
            counts[norm] = counts.get(norm, 0) + 1
    if not counts:
        return None
    max_count = max(counts.values())
    if max_count < min_providers:
        return None
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) > 1:
        return None  # ambiguous split: multiple answers tie at the top -> no genuine convergence
    return winners[0]


# ---- Machine-verifiable checks (no model) ----

def verify_string_contains(answer: Optional[str], needle: str) -> bool:
    """Deterministic 'does the answer contain string X' (normalized, case/punct-insensitive)."""
    return normalize_answer(needle) in normalize_answer(answer) if needle else False


def _extract_numbers(text: str) -> List[str]:
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(text or "")]


def verify_number(answer: Optional[str], expected: str) -> Optional[bool]:
    """True/False if ``expected`` is a number and it appears in ``answer``; ``None`` if
    ``expected`` is not a number (not mechanically checkable this way)."""
    exp = _extract_numbers(expected)
    if len(exp) != 1:
        return None
    try:
        target = float(exp[0])
    except ValueError:
        return None
    for tok in _extract_numbers(answer or ""):
        try:
            if float(tok) == target:
                return True
        except ValueError:
            continue
    return False


def verify_date(answer: Optional[str], expected: str) -> Optional[bool]:
    """True/False if ``expected`` canonicalizes to an ISO date and that date appears in
    ``answer``; ``None`` if ``expected`` is not a recognizable date."""
    exp_iso = _ISO_DATE.search(_canon_dates(expected.strip().lower()))
    if not exp_iso:
        return None
    target = exp_iso.group(0)
    return target in _canon_dates((answer or "").lower())


def machine_verify(answer: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Deterministically verify ``answer`` against ``expected`` with NO model.

    Tries, in order: date match, exact-number match, then normalized string-contains (covers
    names/strings). Returns ``True``/``False`` when a mechanical check applies, else ``None``
    (unanchored — never fabricate). String-contains always applies as the fallback, so a
    non-empty ``expected`` is always checkable; ``None`` is returned only for empty inputs."""
    if not answer or not expected:
        return None
    d = verify_date(answer, expected)
    if d is not None:
        return d
    n = verify_number(answer, expected)
    if n is not None:
        return n
    return verify_string_contains(answer, expected)


# ---- Aggregate anchors over a run ----

@dataclass
class QueryConsensus:
    """Consensus outcome for one query."""

    qi: int
    label: Optional[str]                       # normalized silver label, or None (no consensus)
    n_providers: int                           # how many providers produced a usable answer
    arena_agrees: Optional[bool] = None        # did the arena winner agree with consensus?


@dataclass
class Anchors:
    """Run-level Tier-1 anchor summary (surfaced in the canonical result)."""

    min_providers: int
    per_query: List[QueryConsensus] = field(default_factory=list)
    # auto-verify (machine-checkable) accuracy per provider: {provider: {correct, total}}
    auto_verify: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        n_consensus = sum(1 for c in self.per_query if c.label is not None)
        n_total = len(self.per_query)
        checked = [c for c in self.per_query if c.arena_agrees is not None]
        agree = sum(1 for c in checked if c.arena_agrees)
        return {
            "min_providers": self.min_providers,
            "consensus_coverage": (n_consensus / n_total) if n_total else None,
            "n_consensus_queries": n_consensus,
            "n_queries": n_total,
            "arena_vs_consensus_agreement": (agree / len(checked)) if checked else None,
            "n_arena_checked": len(checked),
            "auto_verify": {p: accuracy_dict(v["correct"], v["total"])
                            for p, v in self.auto_verify.items()},
        }


def compute_anchors(per_query_answers: List[Dict[str, Optional[str]]],
                    per_query_comparisons: List[List[dict]],
                    provider_names: List[str],
                    min_providers: int = DEFAULT_MIN_PROVIDERS) -> Anchors:
    """Assemble the run-level anchors from per-query reader answers + pairwise comparisons.

    ``per_query_answers[qi]`` maps provider -> that provider's reader answer for query ``qi``.
    ``per_query_comparisons[qi]`` is the list of ``{a, b, winner}`` verdicts for query ``qi``.
    Both are indexed by the same query order. Pure: no I/O, deterministic."""
    anchors = Anchors(min_providers=min_providers,
                      auto_verify={p: {"correct": 0, "total": 0} for p in provider_names})
    for qi, answers in enumerate(per_query_answers):
        label = consensus_label(answers, min_providers)
        qc = QueryConsensus(qi=qi, label=label,
                            n_providers=sum(1 for a in answers.values() if normalize_answer(a)))
        if label is not None:
            comps = per_query_comparisons[qi] if qi < len(per_query_comparisons) else []
            qc.arena_agrees = _arena_agrees_with_consensus(comps, answers, label)
        anchors.per_query.append(qc)
    return anchors


def _arena_agrees_with_consensus(comparisons: List[dict], answers: Dict[str, Optional[str]],
                                 label: str) -> Optional[bool]:
    """For pairs where exactly one side matches the consensus label, does the arena winner pick
    the matching side? Returns the majority verdict, or ``None`` if no such decidable pair."""
    matches = {p for p, a in answers.items() if normalize_answer(a) == label}
    agree = decided = 0
    for c in comparisons:
        a, b, w = c.get("a"), c.get("b"), c.get("winner")
        am, bm = a in matches, b in matches
        if am == bm:                       # both or neither match consensus -> not decidable
            continue
        if w not in (a, b):                # judge tied/abstained -> not decidable
            continue
        decided += 1
        agree += int((w == a and am) or (w == b and bm))
    if decided == 0:
        return None
    return agree >= (decided - agree)      # majority agree (ties -> agree)
