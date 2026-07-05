"""Arena mode entrypoint — a reference-free ranking of search-API providers on your queries.

Zero-config: point it at a queries file; keys come from .env (or Doppler). It runs Arena
across every in-scope provider whose key is present and writes a ranked, confidence-scored
result. This is a separate entrypoint and never touches run_evaluation.py.

    python run_arena.py --queries queries.csv
"""

import argparse
import logging
import os
import sys

from arena import secrets
from arena.adapters.registry import REGISTRY
from arena.config import load_config, load_queries, resolve_config_path
from arena.llm import DEFAULT_MODEL, LLMClient
from arena.paths import copy_config_to_results, get_output_dir
from arena.pipeline import run_arena
from arena.report import build_document, render_cli_summary, write_results
from arena.scope import resolve_scope

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the search-arena reference-free ranking")
    parser.add_argument("--queries", required=True, help="Queries file (CSV or JSONL); required column 'query'")
    parser.add_argument("--config", default=None, help="Optional arena config (YAML/JSON)")
    parser.add_argument("--output_dir", default="results", help="Base output directory")
    parser.add_argument("--model", default=None, help=f"Judge model (default {DEFAULT_MODEL})")
    parser.add_argument("--reader-model", default=None,
                        help="Reader/grader model (default = judge model). Use a cheaper model "
                             "here to cut cost — the reader/grader are less quality-sensitive.")
    args = parser.parse_args()

    secrets.load_secrets()

    config = load_config(resolve_config_path(args.config))  # honors configs/arena.yaml by default
    config.output_dir = args.output_dir
    queries = load_queries(args.queries)

    scope = resolve_scope(config.providers)
    included = scope.included
    if not included:
        logger.error("No providers are enabled with a key present. Add provider keys to your "
                     ".env (e.g. TAVILY_API_KEY) and retry.")
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

    logger.info(f"Running arena over: {', '.join(a.name for a in adapters)} "
                f"({len(queries)} queries, judge={model_id})")
    result = run_arena(config, queries, adapters, scope, reader_llm, judge_llm, grader_llm=grader_llm)

    out_dir = get_output_dir(config.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.config:
        copy_config_to_results(args.config, out_dir)

    doc = build_document(result, [q.query for q in queries],
                         config_snapshot={"model_id": model_id,
                                          "evidence_budget_tokens": config.evidence_budget_tokens,
                                          "order_swap": config.order_swap,
                                          "exclude_on_flip": config.exclude_on_flip},
                         model_id=model_id)
    paths = write_results(doc, out_dir)
    print(render_cli_summary(doc))
    logger.info(f"Wrote {paths['json']} and {paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
