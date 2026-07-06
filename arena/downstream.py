"""Tier-3 downstream success (§3): measure end-task success in the USER'S own loop.

The contract is deliberately dumb (§12 spirit — no SDK, no callback registry): the config
names one command; the arena runs it N times per provider with the provider name substituted
in (``{provider}`` placeholder, also exported as ``ARENA_PROVIDER``); exit code 0 counts as
success. Whatever the user's agent/eval loop does inside is theirs — no search-level gold
needed, which is the whole point of Tier 3.

    downstream:
      command: "python my_agent_eval.py --provider {provider}"
      runs: 5
      timeout_s: 300
"""

import logging
import os
import shlex
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RUNS = 5
DEFAULT_TIMEOUT_S = 300


def run_downstream(command: str, providers: List[str], runs: int = DEFAULT_RUNS,
                   timeout_s: int = DEFAULT_TIMEOUT_S, runner=subprocess.run) -> Dict[str, dict]:
    """Run the user's command ``runs`` times per provider; exit 0 = success.

    A timeout or spawn failure counts as a failure (an agent loop that hangs on a provider is
    a real downstream signal, not missing data). ``runner`` is injectable for tests."""
    outcomes: Dict[str, dict] = {}
    for provider in providers:
        argv = [a.replace("{provider}", provider) for a in shlex.split(command)]
        env = {**os.environ, "ARENA_PROVIDER": provider}
        successes = 0
        for i in range(runs):
            try:
                proc = runner(argv, env=env, timeout=timeout_s,
                              capture_output=True)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                logger.warning(f"[{provider}] downstream run {i + 1}/{runs} timed out "
                               f"({timeout_s}s) — counted as failure")
                ok = False
            except Exception as e:
                logger.warning(f"[{provider}] downstream run {i + 1}/{runs} failed to start: "
                               f"{e.__class__.__name__}: {e} — counted as failure")
                ok = False
            successes += int(ok)
        outcomes[provider] = {"success_rate": successes / runs if runs else None,
                              "successes": successes, "n": runs}
        logger.info(f"[{provider}] downstream: {successes}/{runs} succeeded")
    return outcomes


def attach_downstream(metrics: Dict[str, dict], outcomes: Dict[str, dict]) -> Dict[str, dict]:
    """Add a ``downstream`` block to each provider's metrics in-place. Additive; providers
    without an outcome (not run) get nothing — absent, not zero (§8)."""
    for provider, m in metrics.items():
        if provider in outcomes:
            m["downstream"] = outcomes[provider]
    return metrics
