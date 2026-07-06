"""Diff two arena runs — the drift report (freshness / continuous-eval).

Providers ship changes weekly; a ranking is a dated snapshot. Re-run the arena on a schedule
and diff the canonical results.json files to see what actually moved:

    python compare_runs.py results/arena_old/results.json results/arena_new/results.json
    python compare_runs.py old.json new.json --json drift.json   # also save the diff

A rank move is flagged only when the win-rate CIs of the two runs don't overlap — anything
inside the CIs is this workload's noise, not provider drift.
"""

import argparse
import json
import sys

from arena.drift import diff_runs, render_drift


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two arena results.json files (drift report)")
    parser.add_argument("before", help="Earlier run's results.json")
    parser.add_argument("after", help="Later run's results.json")
    parser.add_argument("--json", default=None, help="Also write the structured diff to this path")
    args = parser.parse_args()

    # CLI boundary: a missing or unparseable results file is a user error — print the
    # message, not a traceback.
    try:
        with open(args.before) as f:
            before = json.load(f)
        with open(args.after) as f:
            after = json.load(f)
    except FileNotFoundError as e:
        print(f"Results file not found: {e.filename}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Not a valid results.json ({e})", file=sys.stderr)
        return 1

    diff = diff_runs(before, after)
    print(render_drift(diff))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(diff, f, indent=2)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
