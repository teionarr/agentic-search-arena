"""Scope report — what ran and what didn't, and why.

Provider inclusion is decided by explicit key-presence BEFORE any handler is instantiated
(never by relying on ``ProviderHandler.__init__`` raising — that is non-uniform across
handlers). Every provider ends up in exactly one bucket.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from arena import secrets
from arena.adapters.registry import REGISTRY

logger = logging.getLogger(__name__)

INCLUDED = "included"
NO_KEY = "excluded — no API key"
USER_CHOICE = "excluded — user choice"
RUNTIME_ERROR = "excluded — runtime error"


@dataclass
class ScopeEntry:
    provider: str
    status: str
    detail: str = ""


@dataclass
class Scope:
    entries: List[ScopeEntry] = field(default_factory=list)

    @property
    def included(self) -> List[str]:
        return [e.provider for e in self.entries if e.status == INCLUDED]

    def mark_runtime_error(self, provider: str, message: str) -> None:
        for e in self.entries:
            if e.provider == provider and e.status == INCLUDED:
                e.status = RUNTIME_ERROR
                e.detail = message
                return

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {e.provider: {"status": e.status, "detail": e.detail} for e in self.entries}


def resolve_scope(config_providers: Dict[str, dict]) -> Scope:
    """Decide inclusion for every registered provider.

    ``config_providers``: optional overrides ``{name: {enabled, config}}``. Absent name =>
    default enabled if key present.
    """
    scope = Scope()
    for name, spec in REGISTRY.items():
        override = config_providers.get(name, {}) if config_providers else {}
        if override.get("enabled") is False:
            scope.entries.append(ScopeEntry(name, USER_CHOICE))
            continue
        missing = [k for k in spec.required_env_keys if not secrets.has(k)]
        if missing:
            scope.entries.append(ScopeEntry(name, NO_KEY, f"missing {', '.join(missing)}"))
            continue
        scope.entries.append(ScopeEntry(name, INCLUDED))
    return scope
