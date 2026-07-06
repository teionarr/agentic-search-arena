"""Shipped marketing-claims ledger file (§7): configs/published_claims.yaml is real data.

All Tier-A: deterministic, NO AI, NO network — parses the checked-in file through
``load_published_claims`` and validates every entry is complete and sane.
"""

import datetime
import os

from arena.benchmark import DATASET_PATHS, load_published_claims

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAIMS_PATH = os.path.join(REPO_ROOT, "configs", "published_claims.yaml")


def test_shipped_claims_file_exists_and_parses():
    assert os.path.isfile(CLAIMS_PATH)
    claims = load_published_claims(CLAIMS_PATH)
    assert claims, "shipped published_claims.yaml must contain at least one claim"


def test_shipped_claims_datasets_are_known():
    claims = load_published_claims(CLAIMS_PATH)
    for dataset in claims:
        assert dataset in DATASET_PATHS, f"unknown dataset key: {dataset}"


def test_every_claim_has_score_as_of_and_source():
    claims = load_published_claims(CLAIMS_PATH)
    for dataset, provs in claims.items():
        assert provs, f"{dataset}: empty provider map"
        for prov, entry in provs.items():
            where = f"{dataset}/{prov}"
            # score: published accuracy in [0, 1] per the schema (example + loader docstring).
            score = entry["score"]
            assert isinstance(score, (int, float)) and not isinstance(score, bool), where
            assert 0.0 <= score <= 1.0, f"{where}: score {score} outside [0, 1]"
            # as_of: ISO date the claim was published.
            as_of = entry["as_of"]
            assert isinstance(as_of, str) and as_of, where
            datetime.date.fromisoformat(as_of)  # raises if not a valid ISO date
            # source: URL citation for the claim.
            source = entry["source"]
            assert isinstance(source, str) and source.startswith("https://"), where
