"""Command-line interface.

    python -m harag ingest                       build the vector index from data/corpus
    python -m harag check                        test the connection to the LLMs and models
    python -m harag ask "question" [-c D] [--compare]
    python -m harag calibrate                    fit confidence weights + threshold on the calib split
    python -m harag evaluate                     run configs A-D on the test split and write the report
    python -m harag annotate-export              sample answers for blind human labelling
    python -m harag kappa SAMPLE KEY             agreement between human labels and the judge
    python -m harag ragas -c A                   optional cross-check with the RAGAS library

Use --config config.offline.yaml for the offline smoke-test profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path

from .config import load_settings
from .pipeline import CONFIGS, RAGSystem, ingest


def _p(*args) -> None:
    print(*args, flush=True)


def _fingerprint(settings, rag: RAGSystem, judge_name: str) -> dict:
    keys = ("llm", "judge", "embeddings", "reranker", "retrieval", "verification", "chunking")
    fp = {k: settings.get(k) for k in keys}
    fp["confidence"] = rag.scorer.describe()
    fp["judge_impl"] = judge_name
    fp["questions_sha"] = hashlib.sha1(settings.path("questions_file").read_bytes()).hexdigest()[:12]
    return json.loads(json.dumps(fp, default=str))


def _prepare_run_dir(run_dir: Path, fp: dict, fresh: bool) -> None:
    meta = run_dir / "run_meta.json"
    if run_dir.exists() and (fresh or not meta.exists() or json.loads(meta.read_text()) != fp):
        if not fresh and meta.exists():
            _p(f"  settings changed since the last run -> recomputing {run_dir} (LLM responses come from cache)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(fp, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ commands

def cmd_ingest(settings, args) -> None:
    info = ingest(settings)
    _p(f"Indexed {info['documents']} documents into {info['chunks']} chunks -> {info['index_dir']}")


def cmd_check(settings, args) -> None:
    from .llm import build_llm

    ok = True
    for name in ("llm", "judge"):
        cfg = settings[name]
        try:
            llm = build_llm(cfg, None)
            reply = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=5,
                             task="check")
            _p(f"[ok]   {name}: {cfg.get('model')} -> {reply.strip()[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            _p(f"[fail] {name}: {cfg.get('model')} -> {exc}")
    try:
        rag = RAGSystem(settings, load_index=False)
        rag.embedder.embed_query("test") if settings["embeddings"]["provider"] != "tfidf" else None
        _p(f"[ok]   embeddings: {rag.embedder.describe()}  reranker: {rag.reranker.name}  "
           f"verifier: {rag.verifier.name}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        _p(f"[fail] models: {exc}")
    idx = settings.path("index_dir") / "vectors.npy"
    _p(f"[{'ok' if idx.exists() else 'todo'}] index: {idx.parent}" + ("" if idx.exists() else "  (run ingest)"))
    sys.exit(0 if ok else 1)


def _print_result(r) -> None:
    wrap = lambda t: textwrap.fill(t, 100, initial_indent="  ", subsequent_indent="  ")  # noqa: E731
    _p(f"\n=== {CONFIGS[r.config]['label']} ===")
    badge = {"answer": "ANSWER", "flag": "ANSWER (low confidence)", "abstain": "ABSTAINED"}[r.decision]
    _p(f"{badge}   C={r.signals['confidence']:.2f}  [R={r.signals['retrieval']:.2f} "
       f"S={r.signals['similarity']:.2f} F={r.signals['faithfulness']:.2f}]   "
       f"{r.latency_s:.2f}s, {r.llm_calls} LLM calls")
    _p(wrap(r.final_answer))
    if r.corrected:
        _p("  (draft was corrected; original draft:)")
        _p(wrap(r.draft_answer))
    for c in r.verification.get("claims", []):
        mark = {"supported": "✓", "unsupported": "?", "contradicted": "✗"}.get(c["label"], "?")
        _p(f"    {mark} {c['claim']}  ({c['label']}, {c['score']:.2f})")
    _p("  sources: " + ", ".join(f"[{i}] {h['doc_id']} › {h['section']} ({h['rerank_score']:.2f})"
                                  for i, h in enumerate(r.retrieved, start=1)))


def cmd_ask(settings, args) -> None:
    rag = RAGSystem(settings)
    configs = ["A", "D"] if args.compare else [args.c]
    results = [rag.answer(args.question, cfg, analysis=True) for cfg in configs]
    if args.json:
        _p(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        return
    for r in results:
        _print_result(r)


def _judged_runs(settings, rag, judge, split, configs, limit, fresh) -> dict:
    from .evaluation.runner import load_questions, run_config

    questions = load_questions(settings.path("questions_file"), split, limit)
    if not questions:
        raise SystemExit(f"No questions in split '{split}'")
    run_dir = settings.path("results_dir") / split / "runs"
    _prepare_run_dir(run_dir, _fingerprint(settings, rag, type(judge).__name__), fresh)
    _p(f"Running {len(questions)} questions ({split} split) x configs {', '.join(configs)}")
    return {cfg: run_config(rag, judge, questions, cfg, run_dir / f"{cfg}.jsonl") for cfg in configs}


def cmd_calibrate(settings, args) -> None:
    from .evaluation.calibration import fit_calibration, save_calibration
    from .evaluation.judge import build_judge

    rag = RAGSystem(settings)
    judge = build_judge(settings, rag.embedder)
    runs = _judged_runs(settings, rag, judge, args.split, ["A", "B"], args.limit, args.fresh)
    rows = [r for rs in runs.values() for r in rs]
    cc = settings.get("calibration", {})
    cal = fit_calibration(rows, cc.get("strategy", "coverage"), float(cc.get("min_coverage", 0.8)),
                          settings.get("seed", 42))
    path = rag._calibration_path()
    save_calibration(cal, path)
    _p(f"\nCalibration written to {path}")
    _p(f"  fitted on {cal['n_fit']} answers, {cal['grounded_share'] * 100:.0f}% grounded")
    _p(f"  weights: {cal['weights']}")
    _p(f"  threshold: weighted {cal['threshold_weighted']}, logistic {cal['threshold_logistic']}")
    _p(f"  AUROC on calibration data: {json.dumps(cal['auroc_fit'])}")
    if cal.get("note"):
        _p(f"  WARNING: {cal['note']}")
    _p("Next: python -m harag evaluate")


def cmd_evaluate(settings, args) -> None:
    from .evaluation.judge import build_judge
    from .evaluation.report import write_report

    rag = RAGSystem(settings)
    judge = build_judge(settings, rag.embedder)
    configs = args.configs or settings.get("evaluation", {}).get("configs", list(CONFIGS))
    if rag.scorer.source == "config":
        _p("Note: no calibration file found; using weights/threshold from the config. Run `calibrate` first "
           "for fitted values.")
    runs = _judged_runs(settings, rag, judge, args.split, configs, args.limit, args.fresh)
    cal_path = rag._calibration_path()
    calibration = json.loads(cal_path.read_text()) if cal_path and cal_path.exists() else None
    out_dir = settings.path("results_dir") / args.split
    res = write_report(runs, out_dir, settings, calibration, rag.scorer.describe(), type(judge).__name__, args.split)
    s = res["summary"]
    _p("")
    _p(f"{'config':<34}{'halluc':>8}{'faithf':>8}{'relev':>8}{'acc':>8}{'cover':>8}{'unans↑':>8}{'AUROC':>8}{'lat s':>8}")
    for cfg, m in s.items():
        f = lambda v, pct=True: "   –" if v is None else (f"{v * 100:7.1f}%" if pct else f"{v:8.3f}")  # noqa: E731
        _p(f"{CONFIGS[cfg]['label']:<34}{f(m['hallucination_rate'])}{f(m['faithfulness'], False)}"
           f"{f(m['answer_relevancy'], False)}{f(m['accuracy'])}{f(m['coverage'])}"
           f"{f(m['unanswerable_abstention'])}{f(m['auroc_confidence'], False)}{f(m['mean_latency_s'], False)}")
    _p(f"\nReport: {out_dir / 'report.md'}\nFigures: {out_dir / 'figures'}")


def cmd_annotate_export(settings, args) -> None:
    from .evaluation.annotation import export_sample

    run_dir = settings.path("results_dir") / args.split / "runs"
    files = [run_dir / f"{c}.jsonl" for c in args.configs if (run_dir / f"{c}.jsonl").exists()]
    if not files:
        raise SystemExit(f"No runs in {run_dir}; run evaluate first.")
    blind, key = export_sample(files, settings.path("results_dir") / "annotation", args.n, settings.get("seed", 42))
    _p(f"Blind sample for annotators: {blind}\nJudge labels (do not show annotators): {key}")
    _p("Fill human_hallucinated with 1/0, then: python -m harag kappa <sample.csv> <key.csv>")


def cmd_kappa(settings, args) -> None:
    from .evaluation.annotation import agreement

    _p(json.dumps(agreement(Path(args.sample), Path(args.key), Path(args.second) if args.second else None), indent=2))


def cmd_ragas(settings, args) -> None:
    from .evaluation.ragas_eval import run_ragas

    run_file = settings.path("results_dir") / args.split / "runs" / f"{args.c}.jsonl"
    out = settings.path("results_dir") / args.split / f"ragas_{args.c}.csv"
    df = run_ragas(run_file, settings, out)
    _p(df.drop(columns=[c for c in df.columns if df[c].dtype == object], errors="ignore").mean(numeric_only=True))
    _p(f"Saved {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="harag", description="Hallucination-aware RAG prototype")
    ap.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="build the vector index")
    sub.add_parser("check", help="test LLM/model connectivity")

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("-c", default="D", choices=list(CONFIGS), help="configuration (default D)")
    a.add_argument("--compare", action="store_true", help="show baseline A next to D")
    a.add_argument("--json", action="store_true")

    for name, default_split in (("calibrate", "calib"), ("evaluate", "test")):
        p = sub.add_parser(name)
        p.add_argument("--split", default=default_split, help=f"question split (default {default_split}; 'all' = every question)")
        p.add_argument("--limit", type=int, default=None, help="only the first N questions (for quick trials)")
        p.add_argument("--fresh", action="store_true", help="discard previous runs (LLM cache is kept)")
        if name == "evaluate":
            p.add_argument("--configs", nargs="+", choices=list(CONFIGS))

    e = sub.add_parser("annotate-export", help="export a blind sample for human labelling")
    e.add_argument("--split", default="test")
    e.add_argument("--configs", nargs="+", default=["A", "B"])
    e.add_argument("--n", type=int, default=100)

    k = sub.add_parser("kappa", help="judge vs human agreement")
    k.add_argument("sample")
    k.add_argument("key")
    k.add_argument("--second", help="second annotator's file (inter-annotator kappa)")

    r = sub.add_parser("ragas", help="optional RAGAS cross-check of saved runs")
    r.add_argument("-c", default="A", choices=list(CONFIGS))
    r.add_argument("--split", default="test")

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles cannot print some symbols
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    settings = load_settings(args.config)
    {
        "ingest": cmd_ingest, "check": cmd_check, "ask": cmd_ask, "calibrate": cmd_calibrate,
        "evaluate": cmd_evaluate, "annotate-export": cmd_annotate_export, "kappa": cmd_kappa, "ragas": cmd_ragas,
    }[args.cmd](settings, args)


if __name__ == "__main__":
    main()
