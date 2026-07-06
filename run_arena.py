"""Arena mode entrypoint — a reference-free ranking of search-API providers on your queries.

Zero-config: point it at a queries file; keys come from .env (or Doppler). It runs Arena
across every in-scope provider whose key is present and writes a ranked, confidence-scored
result. This is a separate entrypoint and never touches run_evaluation.py.

    python run_arena.py --queries queries.csv
"""

import argparse
import json
import logging
import os
import sys

from arena import secrets
from arena.adapters.registry import REGISTRY
from arena.config import load_config, load_queries, resolve_config_path
from arena.llm import DEFAULT_MODEL, LLMClient, build_llm_client
from arena.paths import copy_config_to_results, get_output_dir
from arena.pipeline import run_arena
from arena.report import build_document, render_cli_summary, write_results
from arena.scope import resolve_scope
from arena.tracing import build_tracer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# The committed example run (docs/example-run/) — resolved relative to this file so --demo
# works from any cwd.
DEMO_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "docs", "example-run", "results.json")


def run_demo() -> int:
    """Render the committed example run through the real CLI renderer — no keys, no network."""
    with open(DEMO_RESULTS_PATH) as f:
        doc = json.load(f)
    print("\nDEMO — a real committed run: 20 queries × 2 repeats × 6 providers, "
          "judge claude-sonnet-4-6, total cost $15.47. "
          "Your run will look like this on YOUR queries.")
    print(render_cli_summary(doc))
    print("Next: copy .env.example to .env, add ANTHROPIC_API_KEY plus any provider keys, then\n"
          "run  python run_arena.py --queries your_queries.csv  — see the README quickstart\n"
          "for a cheap (~$1-2) first run.")
    return 0


FIRST_RUN_EPILOG = (
    "first run?\n"
    "  1. python run_arena.py --demo — a real committed result, zero keys, zero cost.\n"
    "  2. Cheap first run (~$1-2): 10 queries + --reader-model claude-haiku-4-5-20251001.\n"
    "  3. Full runs cost roughly $8-15 for ~20 queries × 6 providers × 2 repeats\n"
    "     (the committed example run in docs/example-run/ cost a measured $15.47)."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the search-arena reference-free ranking",
                                     epilog=FIRST_RUN_EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=None,
                        help="Queries file (CSV or JSONL); required column 'query'. "
                             "Required unless --demo is set.")
    parser.add_argument("--demo", action="store_true",
                        help="Render the committed example run (docs/example-run/) through the "
                             "real CLI renderer and exit — zero keys, zero network.")
    parser.add_argument("--config", default=None, help="Optional arena config (YAML/JSON)")
    parser.add_argument("--output_dir", default="results", help="Base output directory")
    parser.add_argument("--model", default=None, help=f"Judge model (default {DEFAULT_MODEL})")
    parser.add_argument("--reader-model", default=None,
                        help="Reader/grader model (default = judge model). Use a cheaper model "
                             "here to cut cost — the reader/grader are less quality-sensitive.")
    parser.add_argument("--save-traces", action="store_true", default=None,
                        help="Persist per-query audit traces (redacted raw provider payloads, "
                             "the exact evidence the reader saw, and reader answers) to "
                             "<output_dir>/traces/ so any verdict can be replayed by hand.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Run each query N times per provider (default 1, or config "
                             "'repeats'). Providers are non-deterministic — repeats turn a "
                             "single-shot snapshot into a variance-aware ranking; the "
                             "per-repeat win-rate spread is reported.")
    parser.add_argument("--benchmark-suite", action="store_true", default=None,
                        help="Also re-run public sets (SimpleQA/FRAMES/FreshQA) under one policy: "
                             "calibration report + marketing-claims ledger (§7). Defaults to a "
                             "sample per set; opt into full runs via config.")
    args = parser.parse_args()

    if args.demo:  # before load_secrets(): the demo needs no keys, no .env, no network
        return run_demo()
    if not args.queries:
        parser.error("--queries is required (or use --demo to see an example run)")

    secrets.load_secrets()

    # The judge/reader/grader run on the Anthropic API — this key is required for EVERY run,
    # regardless of which provider keys are present. Fail here, before any provider work.
    if not secrets.has("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY is not set — the judge and reader run on the Anthropic "
                     "API, so this key is required for every arena run (provider keys alone are "
                     "not enough). Add ANTHROPIC_API_KEY to your .env (see .env.example) and retry.")
        return 1

    config_path = resolve_config_path(args.config)  # honors configs/arena.yaml by default
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        # CLI boundary: bad config is a user error — print the message, not a traceback.
        logger.error(str(e))
        return 1
    config.output_dir = args.output_dir
    if args.repeats is not None:
        if args.repeats < 1:
            logger.error("--repeats must be >= 1")
            return 1
        config.repeats = args.repeats
    if args.save_traces:
        config.save_traces = True
    try:
        queries = load_queries(args.queries)
    except (FileNotFoundError, ValueError) as e:
        # CLI boundary: a missing/malformed queries file is a user error — no traceback.
        logger.error(str(e))
        return 1

    scope = resolve_scope(config.providers)
    included = scope.included
    if not included:
        logger.error("No providers are enabled with a key present. Add provider keys to your "
                     ".env (e.g. TAVILY_API_KEY) and retry. (ANTHROPIC_API_KEY is additionally "
                     "required for every run — it powers the judge and reader.)")
        for prov, info in scope.as_dict().items():
            logger.error(f"  {prov}: {info['status']} {info['detail']}")
        return 1

    # Build adapters for included providers (registry default_config merged with overrides).
    adapters = []
    for name in included:
        spec = REGISTRY[name]
        override = (config.providers.get(name, {}) or {}).get("config", {})
        merged = {**spec.default_config, **(override or {})}
        try:
            adapters.append(spec.build(name, merged))
        except Exception as e:
            scope.mark_runtime_error(name, str(e))
            logger.error(f"[{name}] failed to initialize: {e}")

    if not [a for a in adapters]:
        logger.error("All included providers failed to initialize; see scope report above.")
        return 1

    model_id = args.model or DEFAULT_MODEL
    reader_model = args.reader_model or config.reader_model or model_id
    judge_llm = LLMClient(model=model_id)
    reader_llm = LLMClient(model=reader_model)
    grader_llm = LLMClient(model=reader_model)  # accuracy anchor (Claude fallback if no OPENAI_API_KEY)
    # Optional neutral second judge (§5): ensemble + self-preference. Cross-family via a model-id
    # prefix — judge.secondary: "openai:<model>" builds an OpenAI-family judge (the genuinely
    # non-Claude judge §5 asks for); a bare id (or "claude:<model>") stays on the Anthropic client.
    secondary_judge_llm = build_llm_client(config.judge_secondary) if config.judge_secondary else None

    tracer = build_tracer(config.langfuse_enabled)  # NullTracer unless enabled + Langfuse keys present

    logger.info(f"Running arena over: {', '.join(a.name for a in adapters)} "
                f"({len(queries)} queries, judge={model_id}"
                + (f", secondary={config.judge_secondary}" if config.judge_secondary else "") + ")")
    result = run_arena(config, queries, adapters, scope, reader_llm, judge_llm,
                       grader_llm=grader_llm, secondary_judge_llm=secondary_judge_llm, tracer=tracer)

    # Tier-3 downstream success (§3): run the user's own end-task loop per provider and add
    # the judge-free success-rate column. Off unless downstream.command is configured.
    if config.downstream_command:
        from arena.downstream import attach_downstream, run_downstream
        # The command string itself is never logged — quick eval scripts often embed tokens.
        logger.info(f"Running downstream loop ({config.downstream_runs}× per provider, "
                    f"timeout {config.downstream_timeout_s}s)")
        outcomes = run_downstream(config.downstream_command, [a.name for a in adapters],
                                  runs=config.downstream_runs,
                                  timeout_s=config.downstream_timeout_s)
        attach_downstream(result["metrics"], outcomes)

    out_dir = get_output_dir(config.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if config_path:  # snapshot whichever config was actually used (incl. auto-detected arena.yaml)
        copy_config_to_results(config_path, out_dir)

    doc = build_document(result, [q.query for q in queries],
                         config_snapshot={"model_id": model_id,
                                          "reader_model": reader_model,
                                          "judge_secondary": config.judge_secondary,
                                          "aggregation_method": config.aggregation_method,
                                          "judge_reliability_weighting": config.judge_reliability_weighting,
                                          "evidence_budget_tokens": config.evidence_budget_tokens,
                                          "order_swap": config.order_swap,
                                          "exclude_on_flip": config.exclude_on_flip},
                         model_id=model_id)
    paths = write_results(doc, out_dir)
    if config.save_traces and result.get("traces"):
        from arena.report import write_traces
        trace_paths = write_traces(result["traces"], out_dir)
        logger.info(f"Wrote {len(trace_paths)} audit trace(s) to {out_dir}/traces/")
    print(render_cli_summary(doc))
    logger.info(f"Wrote {paths['json']} and {paths['csv']}")

    # Benchmark-suite mode (M2, §7): re-run public sets under the same policy, then emit the
    # calibration report, marketing-claims ledger, and arena-vs-benchmark cross-signal. The
    # --benchmark-suite flag overrides the config toggle; either turns it on.
    if args.benchmark_suite or config.benchmark_suite:
        from arena.benchmark import (cross_signal, load_published_claims,
                                     render_benchmark_summary, run_benchmark_suite,
                                     write_benchmark_report)
        published = load_published_claims(config.published_claims_path)
        bench = run_benchmark_suite(config.benchmark_datasets, config.benchmark_sample_size,
                                    adapters, reader_llm, judge_llm, grader_llm, config,
                                    published=published)
        bench["cross_signal"] = cross_signal(doc["ranking"], bench)
        bench_path = write_benchmark_report(bench, out_dir)
        print(render_benchmark_summary(bench))
        logger.info(f"Wrote {bench_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
