"""Unit tests (no models or internet needed).  Run:  python -m pytest -q"""
from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from harag.config import load_settings
from harag.confidence import ConfidenceScorer
from harag.documents import chunk_document
from harag.evaluation.calibration import fit_calibration
from harag.evaluation.metrics import compare, expected_calibration_error, summarize
from harag.llm import OpenAICompatibleLLM, ResponseCache, parse_json
from harag.pipeline import RAGSystem, ingest
from harag.textutils import ABSTAIN_TOKEN, is_abstention, split_sentences, strip_citations
from harag.verifier import LexicalVerifier

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ text + documents

def test_split_sentences_keeps_citations_and_decimals():
    s = split_sentences("The rate is 0.30 EUR per km. [1] In London it is 220 EUR. [1, 2]")
    assert s == ["The rate is 0.30 EUR per km. [1]", "In London it is 220 EUR. [1, 2]"]
    assert strip_citations(s[1]) == "In London it is 220 EUR."


def test_abstention_detection():
    assert is_abstention(ABSTAIN_TOKEN)
    assert is_abstention("I don't know based on the context.")
    assert not is_abstention("Employees get 25 days of leave. [1]")


def test_chunking_tracks_sections():
    text = "# Policy\n\n## Leave\n\n" + "Word " * 50 + "\n\n## Sick\n\nCall before 10:00."
    chunks = chunk_document("p.md", text, chunk_size=30, overlap=5)
    assert {c.section for c in chunks} == {"Leave", "Sick"}
    assert all(c.text.startswith("[Policy") for c in chunks)


# ------------------------------------------------------------------ verification + confidence

def test_lexical_verifier_catches_wrong_number():
    ctx = ["Employees may carry over a maximum of 5 unused leave days. Bereavement leave is 3 days."]
    v = LexicalVerifier().verify(["Employees may carry over a maximum of 8 unused leave days.",
                                  "Employees may carry over a maximum of 5 unused leave days."], ctx)
    assert [c.label for c in v.claims] == ["unsupported", "supported"]
    assert v.faithfulness == 0.5


def test_confidence_weighted_and_logistic(tmp_path):
    sc = ConfidenceScorer({"method": "weighted", "weights": {"retrieval": 1, "similarity": 1, "faithfulness": 2},
                           "threshold": 0.6, "flag_band": 0.1})
    assert abs(sc.combine(0.4, 0.4, 1.0) - 0.7) < 1e-9
    assert sc.decide(0.65) == "answer" and sc.decide(0.55) == "flag" and sc.decide(0.3) == "abstain"
    cal = {"weights": {"retrieval": 0.2, "similarity": 0.2, "faithfulness": 0.6},
           "logistic": {"intercept": -2.0, "coef": {"retrieval": 1.0, "similarity": 1.0, "faithfulness": 3.0}},
           "threshold_weighted": 0.5, "threshold_logistic": 0.7}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    lg = ConfidenceScorer({"method": "logistic"}, p)
    assert lg.threshold == 0.7
    assert abs(lg.combine(1, 1, 1) - 1 / (1 + np.exp(-3.0))) < 1e-9


# ------------------------------------------------------------------ LLM client against a mock server

class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        if "response_format" in body and body.get("model") == "no-json-mode":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format not supported"}')
            return
        reply = {"choices": [{"message": {"content": '```json\n{"claims": ["a", "b"]}\n```'}}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_compatible_client_caching_and_json_fallback(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        cache = ResponseCache(tmp_path / "c.sqlite")
        llm = OpenAICompatibleLLM(base, "m1", api_key="k", cache=cache)
        msgs = [{"role": "user", "content": "hi"}]
        assert llm.chat_json(msgs) == {"claims": ["a", "b"]}
        assert _Handler.requests[-1]["response_format"] == {"type": "json_object"}
        assert llm.stats.calls == 1 and llm.stats.prompt_tokens == 11
        n = len(_Handler.requests)
        llm.chat_json(msgs)  # second identical call -> cache
        assert len(_Handler.requests) == n and llm.stats.cached_calls == 1
        # endpoint without JSON mode: client retries without response_format
        llm2 = OpenAICompatibleLLM(base, "no-json-mode", cache=ResponseCache(None))
        assert llm2.chat_json(msgs) == {"claims": ["a", "b"]}
        assert "response_format" not in _Handler.requests[-1]
    finally:
        server.shutdown()


def test_parse_json_tolerates_prose():
    assert parse_json('Sure! {"label": "correct"} hope this helps') == {"label": "correct"}
    assert parse_json("not json") == {}


# ------------------------------------------------------------------ pipeline (offline profile)

def _offline_settings(tmp_path):
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copy(ROOT / "config.offline.yaml", tmp_path / "config.offline.yaml")
    shutil.copytree(ROOT / "data" / "corpus", tmp_path / "data" / "corpus")
    shutil.copytree(ROOT / "data" / "eval", tmp_path / "data" / "eval")
    return load_settings(tmp_path / "config.offline.yaml")


def test_pipeline_correction_removes_injected_hallucination(tmp_path):
    settings = _offline_settings(tmp_path)
    info = ingest(settings)
    assert info["documents"] == 10
    rag = RAGSystem(settings)
    q = "What is the hotel limit per night in London?"
    a = rag.answer(q, "A")
    b = rag.answer(q, "B")
    assert "155" in a.final_answer  # demo LLM injected a wrong number (150 -> 155)
    assert b.corrected and "155" not in b.final_answer and "220" in b.final_answer
    assert b.verification["faithfulness"] == 1.0
    assert a.llm_calls == 1 and b.llm_calls == 2  # analysis-only work is not counted for A


# ------------------------------------------------------------------ metrics + calibration

def _row(qid, cfg, answered, halluc, answerable=True, conf=0.5, faith=1.0):
    return {"question_id": qid, "config": cfg, "answered": answered, "answerable": answerable,
            "model_abstained": not answered, "decision": "answer" if answered else "abstain", "corrected": False,
            "judge": {"hallucinated": halluc if answered else None, "faithfulness": faith if answered else None,
                      "relevancy": 0.8 if answered else None, "correctness_score": 1.0 if answered and not halluc else 0.0},
            "signals": {"confidence": conf, "retrieval": conf, "similarity": conf, "faithfulness": faith},
            "latency_s": 1.0, "llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}


def test_summarize_and_compare():
    a = [_row("q1", "A", True, True, conf=0.2, faith=0.5), _row("q2", "A", True, False, conf=0.9),
         _row("q3", "A", True, True, answerable=False, conf=0.3, faith=0.0)]
    d = [_row("q1", "D", False, None), _row("q2", "D", True, False, conf=0.9), _row("q3", "D", False, None, answerable=False)]
    sa, sd = summarize(a), summarize(d)
    assert abs(sa["hallucination_rate"] - 2 / 3) < 1e-9 and sd["hallucination_rate"] == 0
    assert sd["unanswerable_abstention"] == 1.0 and sa["auroc_confidence"] == 1.0
    c = compare(a, d, n_boot=200)
    assert c["fixed"] == 2 and c["introduced"] == 0 and c["hallucination_reduction"] == 1.0


def test_calibration_learns_faithfulness_weight():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        halluc = bool(i % 3 == 0)
        f = float(rng.uniform(0, 0.5) if halluc else rng.uniform(0.6, 1.0))
        r = _row(f"q{i}", "A", True, halluc, faith=f, conf=0.5)
        r["signals"]["retrieval"] = float(rng.uniform())  # pure noise
        rows.append(r)
    cal = fit_calibration(rows, "coverage", 0.8)
    assert cal["weights"]["faithfulness"] > cal["weights"]["retrieval"]
    assert cal["auroc_fit"]["faithfulness"] > 0.95
    assert 0 < cal["threshold_weighted"] < 1


def test_ece_perfect_calibration_is_zero():
    assert expected_calibration_error([1.0, 1.0, 0.0], [1, 1, 0]) == 0.0
