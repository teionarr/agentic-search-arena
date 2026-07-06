"""Calibration-vs-gold: the pure agreement logic, the grader fallback, and an e2e check."""

from arena.adapters.base import EvidenceDoc
from arena.calibrate import run_calibration
from arena.config import ArenaConfig, Query
from arena.grade import _GradeResult, grade_answer, pair_agreement
from arena.judge import PairwiseVerdict
from _fakes import FakeAdapter, FakeLLM, sync_gather


# ---- pair_agreement (pure) ----

def test_pair_agreement_decidable_correct():
    # x is correct, judge picked x -> agree
    assert pair_agreement("x", "y", {"x": True, "y": False}, "x") is True


def test_pair_agreement_decidable_wrong():
    assert pair_agreement("x", "y", {"x": True, "y": False}, "y") is False


def test_pair_agreement_not_decidable_same_correctness():
    assert pair_agreement("x", "y", {"x": True, "y": True}, "x") is None


def test_pair_agreement_judge_abstained():
    assert pair_agreement("x", "y", {"x": True, "y": False}, "tie") is None
    assert pair_agreement("x", "y", {"x": True, "y": False}, None) is None


# ---- grader fallback ----

def test_grade_answer_uses_claude_fallback_without_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = FakeLLM(structured_fn=lambda s, u, sch: _GradeResult(correct=True))
    assert grade_answer("q", "some answer", "gold", llm=llm) is True


def test_grade_answer_none_when_no_grader(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert grade_answer("q", "answer", "gold", llm=None) is None
    assert grade_answer("q", "", "gold", llm=FakeLLM()) is None  # empty answer


# ---- end-to-end agreement ----

def _reader_fn(system, user):
    # Reader "synthesizes" the fact present in the evidence block.
    return "The capital is Paris." if "Paris" in user else "The capital is Lyon."


def _judge_fn(system, user, schema):
    # Judge prefers whichever Answer-A/B block cites Paris (the correct fact).
    a = user.find("### Answer A"); b = user.find("### Answer B")
    a_has = "Paris" in user[a:b]
    return PairwiseVerdict(winner="A" if a_has else "B", rationale="more supported")


def _grader_fn(system, user, schema):
    # Grade only the predicted answer (the gold "Paris" is also in the prompt, so scope it).
    pred = user.split("Predicted answer:")[-1]
    return _GradeResult(correct="Paris" in pred)


def test_calibration_agreement_perfect(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # force Claude grader path
    good = FakeAdapter("tavily", [EvidenceDoc(url="a", title="t", content="Paris is the capital of France.")])
    bad = FakeAdapter("brave", [EvidenceDoc(url="b", title="t", content="Lyon is a city in France.")])
    cfg = ArenaConfig(evidence_budget_tokens=600)
    r = run_calibration(
        [Query(query="capital of France?", expected_answer="Paris")],
        [good, bad],
        reader_llm=FakeLLM(complete_fn=_reader_fn),
        judge_llm=FakeLLM(structured_fn=_judge_fn),
        grader_llm=FakeLLM(structured_fn=_grader_fn),
        config=cfg, search_gatherer=sync_gather,
    )
    # One decidable pair (Paris-answer correct, Lyon-answer wrong); judge picked the correct one.
    assert r["n_decidable_pairs"] == 1
    assert r["agreement"] == 1.0
    assert r["grader"] == "claude"


def test_calibration_agreement_wrong_judge(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    good = FakeAdapter("tavily", [EvidenceDoc(url="a", title="t", content="Paris is the capital of France.")])
    bad = FakeAdapter("brave", [EvidenceDoc(url="b", title="t", content="Lyon is a city in France.")])
    # Judge always picks A regardless of content; provider order -> A is tavily (correct) here,
    # so force the WRONG pick by always choosing B.
    r = run_calibration(
        [Query(query="capital of France?", expected_answer="Paris")],
        [good, bad],
        reader_llm=FakeLLM(complete_fn=_reader_fn),
        judge_llm=FakeLLM(structured_fn=lambda s, u, sch: PairwiseVerdict(winner="B", rationale="x")),
        grader_llm=FakeLLM(structured_fn=_grader_fn),
        config=ArenaConfig(evidence_budget_tokens=600), search_gatherer=sync_gather,
    )
    # order_swap on: pass1 B=brave(wrong) wins, pass2 B=tavily(correct) wins -> flip -> excluded
    # -> judge abstained, not decidable. Agreement undefined.
    assert r["n_decidable_pairs"] == 0 and r["n_judge_abstained"] == 1
    assert r["agreement"] is None