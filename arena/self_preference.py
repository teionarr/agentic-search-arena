"""Self-preference caveat for the native-answer path (§5/§6).

The judge is always blinded + order-swapped, which neutralizes self-preference in the
primary (reader-synthesized) path — the provider's identity and style are invisible there.
It survives only in the **native-answer path**, where a Claude-family provider returns its own
answer that a Claude judge might favor by style.

Mitigation policy (§5): in native-answer mode, a pair involving a Claude-family provider is
routed to a configured neutral secondary judge; if none is configured, the pair is flagged
``possible-self-preference`` instead. Primary-path pairs are never flagged.

This module is pure (no provider identity beyond the family list from the registry) so the
labeling rule is testable in isolation.
"""

from typing import Optional, Set

POSSIBLE_SELF_PREFERENCE = "possible-self-preference"


def self_preference_label(
    a: str,
    b: str,
    a_native: bool,
    b_native: bool,
    claude_family: Set[str],
    judge_is_claude: bool,
    has_secondary_judge: bool,
) -> Optional[str]:
    """Return ``"possible-self-preference"`` for a pair that must be flagged, else ``None``.

    Flagged iff: Claude is the primary judge AND no secondary judge is configured AND at least
    one side is a Claude-family provider answering via the native-answer path (so its own,
    style-bearing answer — not a reader synthesis — reaches the judge).
    """
    if not judge_is_claude or has_secondary_judge:
        return None
    claude_native = (a in claude_family and a_native) or (b in claude_family and b_native)
    return POSSIBLE_SELF_PREFERENCE if claude_native else None
