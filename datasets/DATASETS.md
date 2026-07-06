# Vendored benchmark datasets

Data files consumed by the benchmark-suite loaders in `arena/benchmark.py`
(`DATASET_PATHS`). Each entry records source, license, retrieval date, row count, and
any transformation applied, so a re-run is auditable and the files can be refreshed
from source at any time.

## SimpleQA — `simple_qa_test_set.csv`

- **Source:** OpenAI simple-evals — <https://github.com/openai/simple-evals>
  (data file: <https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv>)
- **License:** MIT (per the simple-evals repository)
- **Vendored by:** the base repository (predates this file); row count verified 2026-07-06
- **Rows:** 4,326
- **Format:** CSV — `metadata,problem,answer`
- **Transformation:** none (verbatim upstream file)

## FRAMES — `frames_test_set.csv`

- **Source:** Google FRAMES benchmark —
  <https://huggingface.co/datasets/google/frames-benchmark> (file `test.tsv`, `test` split)
- **License:** Apache-2.0 (verified on the Hugging Face dataset card metadata,
  `license: apache-2.0`, on 2026-07-06) — redistribution permitted with attribution
- **Retrieved:** 2026-07-06
- **Rows:** 824
- **Format:** CSV — `id,Prompt,Answer,wikipedia_link_1..wikipedia_link_11+,reasoning_types,wiki_links`
- **Transformation:** converted upstream TSV to CSV; the unnamed index column was renamed
  to `id`. All columns and rows preserved otherwise. The loader (`load_frames`) reads only
  `Prompt`/`Answer`.
- **Citation:** Krishna et al., *Fact, Fetch, and Reason: A Unified Evaluation of
  Retrieval-Augmented Generation* (2024), <https://arxiv.org/abs/2409.12941>

## FreshQA — `freshqa_test_set.csv`

- **Source:** FreshLLMs / FreshQA — <https://github.com/freshllms/freshqa>, dataset
  edition **April 21, 2026** (the latest weekly sheet linked from the README at retrieval
  time: <https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM>)
- **License:** Apache-2.0 (verified in the repository `LICENSE` file on 2026-07-06) —
  redistribution permitted with attribution
- **Retrieved:** 2026-07-06
- **Rows:** 500 (the `TEST` split; the 100 `DEV` rows are not vendored)
- **Format:** CSV — `id,question,answer,alt_answers,fact_type,false_premise,num_hops,effective_year,source`
- **Transformation:** exported the Google sheet as CSV; dropped the 2-line preamble;
  kept `split == TEST` rows only; `answer_0` became `answer`, `answer_1..answer_9` were
  joined into `alt_answers` (separated by `" | "`); dropped `split`, `next_review`, `note`.
  The loader (`load_freshqa`) reads `question`, `answer`, and `fact_type` (freshness tag).
- **Freshness caveat:** FreshQA answers drift by design — upstream updates weekly. This
  snapshot is dated; refresh from the latest sheet in the upstream README when answer
  currency matters (fast-changing rows especially).
- **Citation:** Vu et al., *FreshLLMs: Refreshing Large Language Models with Search
  Engine Augmentation* (2023), <https://arxiv.org/abs/2310.03214>
