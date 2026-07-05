"""Secret resolution — .env-first, Doppler-optional, through one interface.

The reused base handlers call ``os.getenv("X_API_KEY")`` directly in ``__init__``. So a
resolved secret is only visible to them if it lives in ``os.environ``. This module therefore
loads secrets (from ``.env`` by default, or Doppler if configured) and **exports them into
``os.environ``**, so the rest of the code — handlers included — never has to know the source.

Security invariant: secret *values* are never logged, never placed on the CLI, never written
to any artifact. Callers ask about *presence* (``has(name)``), not values.
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_doppler() -> Dict[str, str]:
    """Return secrets from the Doppler CLI if it is installed and configured, else {}.

    Best-effort: any failure (not installed, not logged in, no project) silently yields {}
    so ``.env`` remains the working default.
    """
    if not shutil.which("doppler"):
        return {}
    try:
        out = subprocess.run(
            ["doppler", "secrets", "download", "--no-file", "--format", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return {}
        data = json.loads(out.stdout)
        return {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}
    except Exception as e:
        logger.info(f"Doppler not used ({e.__class__.__name__}); falling back to .env")
        return {}


def ensure_ca_bundle() -> None:
    """Make this process's default TLS verification use the certifi CA bundle.

    The python.org macOS framework builds ship without system CA certs, so aiohttp's default
    TLS context fails for every provider host (``unable to get local issuer certificate``).
    Setting ``SSL_CERT_FILE`` alone doesn't fix aiohttp's default context, so we also wrap
    ``ssl.create_default_context`` to load certifi's bundle — process-wide, scoped to this run,
    without touching the reused handlers' TLS setup (what macOS 'Install Certificates' does,
    but without modifying the Python install). No-op if certifi is unavailable.
    """
    try:
        import ssl
        import certifi
    except Exception:
        return

    bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)

    if getattr(ssl, "_arena_certifi_patched", False):
        return
    _orig = ssl.create_default_context

    def _with_certifi(*args, **kwargs):
        if not kwargs.get("cafile") and not kwargs.get("capath") and not kwargs.get("cadata"):
            kwargs["cafile"] = bundle
        return _orig(*args, **kwargs)

    ssl.create_default_context = _with_certifi
    ssl._arena_certifi_patched = True

    # aiohttp caches its default verified context at IMPORT time (connector._SSL_CONTEXT_VERIFIED),
    # before this patch runs — so replace it directly with a certifi-backed context. This is the
    # context the reused handlers' default HTTPS calls actually use.
    try:
        import aiohttp.connector as _c
        _c._SSL_CONTEXT_VERIFIED = _orig(cafile=bundle)
    except Exception:
        pass


def load_secrets(dotenv_path: Optional[str] = None) -> None:
    """Resolve secrets and export them into ``os.environ``.

    Order: ``.env`` first (via python-dotenv), then Doppler overlays on top if available.
    Also ensures a usable TLS CA bundle. Idempotent. Never prints values.
    """
    ensure_ca_bundle()
    load_dotenv(dotenv_path)
    for k, v in _load_doppler().items():
        os.environ[k] = v


def get_secret(name: str) -> Optional[str]:
    """Return a secret value from the environment (or None). For internal use only."""
    return os.environ.get(name)


def has(name: str) -> bool:
    """Whether a secret is present and non-empty — the only thing scope logic should ask."""
    return bool(os.environ.get(name))
