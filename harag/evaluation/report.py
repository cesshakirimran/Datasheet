"""Writes result tables (CSV/JSON), figures (PNG) and a Markdown report."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..pipeline import CONFIGS  # noqa: E402
from . import metrics as M  # noqa: E402
from .runner import retrieval_hit_rate  # noqa: E402

COLORS = {"A": "#7a7f87", "B": "#e8a33d", "C": "#4c8fd6", "D": "#2a9d6f"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _fmt(v, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def per_question_table(rows_by_config: dict[str, list[dict]]) -> pd.DataFrame:
    recs = []
    for cfg, rows in rows_by_config.items():
        for r in rows:
            j = r["judge"]
            recs.append({
                "config": cfg, "question_id": r["question_id"], "type": r["type"], "answerable": r["answerable"],
                "question": r["question"], "decision": r["decision"], "answered": r["answered"],
                "model_abstained": r["model_abstained"], "corrected": r["corrected"],
                "final_answer": r["final_answer"], "pre_gate_answer": r["pre_gate_answer"],
                "reference": r.get("reference"),
                "R_retrieval": r["signals"]["retrieval"], "S_similarity": r["signals"]["similarity"],
                "F_faithfulness_runtime": r["signals"]["faithfulness"], "C_confidence": r["signals"]["confidence"],
                "judge_hallucinated": j.get("hallucinated"), "judge_faithfulness": j.get("faithfulness"),
                "judge_relevancy": j.get("relevancy"), "judge_correctness": j.get("correctness"),
                "hallucinated_final": M.hallucinated_final(r), "correct_final": M.correct_final(r),
                "latency_s": r["latency_s"], "llm_calls": r["llm_calls"],
                "tokens": r["prompt_tokens"] + r["completion_tokens"],
                "retrieved_docs": ";".join(r.get("retrieved_docs", [])),
            })
    return pd.DataFrame(recs)


# ------------------------------------------------------------------ figures

def plot_metrics(summary: dict[str, dict], out: Path) -> None:
    cfgs = list(summary)
    panels = [
        ("Lower is better", [("hallucination_rate", "Hallucination rate"),
                             ("selective_hallucination_rate", "Halluc. rate (answered)")]),
        ("Higher is better", [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer relevancy"),
                              ("accuracy", "Accuracy"), ("coverage", "Coverage"),
                              ("unanswerable_abstention", "Abstains on\nunanswerable")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 5]})
    for ax, (title, items) in zip(axes, panels):
        width = 0.8 / len(cfgs)
        for i, cfg in enumerate(cfgs):
            vals = [summary[cfg].get(k) or 0 for k, _ in items]
            xs = [x + (i - (len(cfgs) - 1) / 2) * width for x in range(len(items))]
            bars = ax.bar(xs, vals, width, color=COLORS.get(cfg, "#999"), label=CONFIGS[cfg]["label"])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([lbl for _, lbl in items], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10, loc="left")
    axes[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(rows_by_config: dict[str, list[dict]], summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cfg in ("A", "B"):
        if cfg in rows_by_config:
            rc = M.risk_coverage(rows_by_config[cfg])
            if rc["coverage"]:
                ax.plot(rc["coverage"], rc["risk"], color=COLORS[cfg], lw=2,
                        label=f"{CONFIGS[cfg]['label']} answers ranked by C (AURC {rc['aurc']:.3f})")
    for cfg in ("C", "D"):
        if cfg in summary and summary[cfg]["n_scored"]:
            rows = rows_by_config[cfg]
            scored = [r for r in rows if not r["model_abstained"] and r["judge"].get("hallucinated") is not None]
            shown = [r for r in scored if r["answered"]]
            if scored and shown:
                cov = len(shown) / len(scored)
                risk = sum(bool(r["judge"]["hallucinated"]) for r in shown) / len(shown)
                ax.scatter([cov], [risk], s=70, color=COLORS[cfg], zorder=5, edgecolor="white",
                           label=f"{CONFIGS[cfg]['label']} operating point")
    ax.set_xlabel("Coverage (share of non-abstained answers kept)")
    ax.set_ylabel("Hallucination rate among kept answers")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, None)
    ax.set_title("Risk–coverage curve (answers kept in order of confidence C)", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(rows: list[dict], cfg: str, out: Path) -> None:
    bins = M.reliability_bins(rows)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1, label="Perfect calibration")
    if bins:
        ax.plot([b["mean_conf"] for b in bins], [b["grounded_rate"] for b in bins], marker="o",
                color=COLORS.get(cfg, "#333"), lw=2, label=f"Config {cfg}")
        for b in bins:
            ax.annotate(f"n={b['n']}", (b["mean_conf"], b["grounded_rate"]), textcoords="offset points",
                        xytext=(4, -10), fontsize=6.5, color="#555")
    ax.set_xlabel("Confidence score C")
    ax.set_ylabel("Share of answers judged grounded")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Reliability diagram", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_signal_auroc(summary: dict, cfg: str, out: Path) -> None:
    keys = [("retrieval", "R retrieval"), ("similarity", "S similarity"),
            ("faithfulness", "F faithfulness"), ("confidence", "C combined")]
    vals = [summary[cfg].get(f"auroc_{k}") or 0 for k, _ in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.barh([l for _, l in keys], vals, color=["#9bb7d4", "#9bb7d4", "#9bb7d4", COLORS.get(cfg, "#333")])
    ax.axvline(0.5, ls="--", color="#bbb", lw=1)
    ax.text(0.505, -0.6, "chance", fontsize=7, color="#888")
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC for separating grounded from hallucinated answers")
    ax.set_title(f"Which signal predicts hallucination? (config {cfg})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cost(summary: dict, out: Path) -> None:
    cfgs = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, label in ((axes[0], "mean_latency_s", "Mean latency per question (s)"),
                           (axes[1], "mean_llm_calls", "Mean LLM calls per question")):
        vals = [summary[c].get(key) or 0 for c in cfgs]
        bars = ax.bar(cfgs, vals, color=[COLORS.get(c, "#999") for c in cfgs])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ report

def write_report(rows_by_config: dict[str, list[dict]], out_dir: Path, settings, calibration: dict | None,
                 scorer_desc: dict, judge_name: str, split: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    n_boot = int(settings.get("evaluation", {}).get("bootstrap_samples", 2000))

    summary = {cfg: M.summarize(rows) for cfg, rows in rows_by_config.items()}
    for cfg, rows in rows_by_config.items():
        summary[cfg]["retrieval_hit_rate"] = retrieval_hit_rate(rows)
    comparisons = {}
    if "A" in rows_by_config:
        for cfg in rows_by_config:
            if cfg != "A":
                comparisons[f"A_vs_{cfg}"] = M.compare(rows_by_config["A"], rows_by_config[cfg], n_boot,
                                                       settings.get("seed", 42))
        for cfg in summary:
            base = summary["A"]["hallucination_rate"]
            other = summary[cfg]["hallucination_rate"]
            summary[cfg]["hallucination_reduction_vs_A"] = ((base - other) / base) if base else None

    per_q = per_question_table(rows_by_config)
    per_q.to_csv(out_dir / "per_question.csv", index=False)
    pd.DataFrame(summary).T.rename_axis("config").to_csv(out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps(
        {"split": split, "summary": summary, "comparisons": comparisons, "confidence": scorer_desc,
         "calibration": calibration, "judge": judge_name}, indent=2, default=str), encoding="utf-8")

    plot_metrics(summary, fig_dir / "metrics_by_config.png")
    plot_risk_coverage(rows_by_config, summary, fig_dir / "risk_coverage.png")
    gate_cfg = "D" if "D" in rows_by_config else next(iter(rows_by_config))
    plot_reliability(rows_by_config[gate_cfg], gate_cfg, fig_dir / "reliability.png")
    sig_cfg = "A" if "A" in summary else gate_cfg
    plot_signal_auroc(summary, sig_cfg, fig_dir / "signal_auroc.png")
    plot_cost(summary, fig_dir / "cost_latency.png")

    md = _markdown(summary, comparisons, settings, calibration, scorer_desc, judge_name, split)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    return {"summary": summary, "comparisons": comparisons}


def _model_label(cfg: dict) -> str:
    provider = cfg.get("provider", "")
    return provider if provider in ("tfidf", "none") else f"{provider}:{cfg.get('model', '')}"


def _markdown(summary, comparisons, settings, calibration, scorer_desc, judge_name, split) -> str:
    lines = [f"# Evaluation report ({split} split)", ""]
    if settings["llm"].get("provider") == "demo":
        lines += ["> **Offline smoke-test run** (demo LLM, TF-IDF, lexical judge). These numbers only show that the",
                  "> pipeline works. Do not report them; rerun with `config.yaml` and real models.", ""]
    lines += [
        f"- Generator: `{settings['llm'].get('model')}` · Judge: `{settings['judge'].get('model')}` ({judge_name})",
        f"- Embeddings: `{_model_label(settings['embeddings'])}` · "
        f"Reranker: `{_model_label(settings['reranker'])}` · "
        f"Verifier: `{settings['verification'].get('method')}`",
        f"- Confidence: method `{scorer_desc['method']}`, weights {scorer_desc['weights']}, "
        f"threshold τ = {scorer_desc['threshold']}, flag band {scorer_desc['flag_band']} (source: {scorer_desc['source']})",
        "",
        "## Results by configuration",
        "",
        "| Metric | " + " | ".join(CONFIGS[c]["label"] for c in summary) + " |",
        "| --- | " + " | ".join("---" for _ in summary) + " |",
    ]
    rows = [
        ("Questions", "n_questions", False, 0),
        ("Hallucination rate (all questions) ↓", "hallucination_rate", True, 3),
        ("Hallucination rate (answered) ↓", "selective_hallucination_rate", True, 3),
        ("Hallucination reduction vs A ↑", "hallucination_reduction_vs_A", True, 3),
        ("Faithfulness (judge) ↑", "faithfulness", False, 3),
        ("Answer relevancy ↑", "answer_relevancy", False, 3),
        ("Accuracy (answerable) ↑", "accuracy", True, 3),
        ("Coverage (answerable) ↑", "coverage", True, 3),
        ("Abstains on unanswerable ↑", "unanswerable_abstention", True, 3),
        ("Answers flagged low-confidence", "flagged_share", True, 3),
        ("Answers corrected", "corrected_share", True, 3),
        ("AUROC of C ↑", "auroc_confidence", False, 3),
        ("ECE of C ↓", "ece_confidence", False, 3),
        ("Retrieval hit rate", "retrieval_hit_rate", True, 3),
        ("Mean latency (s)", "mean_latency_s", False, 2),
        ("p95 latency (s)", "p95_latency_s", False, 2),
        ("Mean LLM calls", "mean_llm_calls", False, 2),
        ("Mean tokens", "mean_tokens", False, 0),
    ]
    for label, key, pct, digits in rows:
        lines.append(f"| {label} | " + " | ".join(_fmt(summary[c].get(key), pct, digits) for c in summary) + " |")
    if comparisons:
        lines += ["", "## Paired comparison with the baseline (A)", "",
                  "| Comparison | Halluc. rate A → X | Difference [95% CI] | Fixed / introduced | McNemar p | "
                  "Δ faithfulness | Wilcoxon p |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, c in comparisons.items():
            ci = c["halluc_rate_diff_ci95"]
            lines.append(
                f"| {name.replace('_', ' ')} | {_fmt(c['halluc_rate_base'], True)} → {_fmt(c['halluc_rate_other'], True)} | "
                f"{_fmt(c['halluc_rate_diff'], True)} [{_fmt(ci[0], True)}, {_fmt(ci[1], True)}] | "
                f"{c['fixed']} / {c['introduced']} | {_fmt(c['mcnemar_p'])} | {_fmt(c['faithfulness_diff'])} | "
                f"{_fmt(c['wilcoxon_p_faithfulness'])} |")
    sig = "A" if "A" in summary else next(iter(summary))
    lines += ["", f"## Which signal predicts hallucination (AUROC, config {sig})", "",
              "| Signal | AUROC |", "| --- | --- |"]
    for k in ("retrieval", "similarity", "faithfulness", "confidence"):
        lines.append(f"| {k} | {_fmt(summary[sig].get('auroc_' + k))} |")
    if calibration:
        lines += ["", "## Calibration (fitted on the calibration split)", "",
                  f"- Fitted on {calibration['n_fit']} judged answers ({_fmt(calibration['grounded_share'], True)} grounded)",
                  f"- Weights: {calibration['weights']}",
                  f"- Thresholds: weighted τ = {calibration['threshold_weighted']}, "
                  f"logistic τ = {calibration['threshold_logistic']} (strategy: {calibration['strategy']}, "
                  f"min coverage {calibration['min_coverage']})"]
    lines += ["", "## Figures", "",
              "![Metrics](figures/metrics_by_config.png)", "",
              "![Risk-coverage](figures/risk_coverage.png)", "",
              "![Signal AUROC](figures/signal_auroc.png)", "",
              "![Reliability](figures/reliability.png)", "",
              "![Cost](figures/cost_latency.png)", "",
              "Per-question details: `per_question.csv`. Raw runs: `runs/<config>.jsonl`."]
    return "\n".join(lines) + "\n"
