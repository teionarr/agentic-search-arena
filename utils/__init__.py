# Exports are resolved lazily (PEP 562) so that importing a light submodule such as
# utils.token_utils does not pull in the heavy legacy stack (langchain_openai via
# post_processor, pandas/quotientai via utils) under the slim arena install
# (requirements-arena.txt). `from utils import PostProcessor, ...` still works.
_UTILS_EXPORTS = {
    "save_summary",
    "load_csv_data",
    "prepare_examples",
    "get_output_dir",
    "save_result",
    "EvaluationType",
    "get_quotient_ai_client",
    "load_document_relevance_eval_data",
    "copy_config_to_results",
}

__all__ = ["PostProcessor", *sorted(_UTILS_EXPORTS)]


def __getattr__(name):
    if name == "PostProcessor":
        from .post_processor import PostProcessor
        return PostProcessor
    if name in _UTILS_EXPORTS:
        from . import utils as _utils
        return getattr(_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
