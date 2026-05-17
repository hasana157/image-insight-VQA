"""
evaluation.py
Member 02 - Core Ownership
Runs full evaluation on predictions.csv and generates:
  - results/metrics_summary.csv
  - results/question_type_accuracy.png
  - results/inference_time_chart.png
  - results/failure_cases.csv
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for Colab and servers)
import matplotlib.pyplot as plt


# ── Paths ────────────────────────────────────────────────────────────────────
PREDICTIONS_CSV  = "data/predictions.csv"
RESULTS_DIR      = "results"
METRICS_CSV      = os.path.join(RESULTS_DIR, "metrics_summary.csv")
QTYPE_CHART      = os.path.join(RESULTS_DIR, "question_type_accuracy.png")
INFTIME_CHART    = os.path.join(RESULTS_DIR, "inference_time_chart.png")
FAILURE_CSV      = os.path.join(RESULTS_DIR, "failure_cases.csv")


def evaluate_predictions(predictions_csv: str = PREDICTIONS_CSV) -> dict:
    """
    Master evaluation function. Reads predictions CSV, computes all metrics,
    saves charts and CSVs to results/, and returns a metrics summary dict.

    Args:
        predictions_csv: Path to predictions CSV with columns:
            id, image_path, question, question_type,
            ground_truth_answer, predicted_answer, is_correct, inference_time_sec

    Returns:
        dict with keys: overall_accuracy, per_type_accuracy,
                        avg_inference_time, failure_rate, total_rows
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = _load_and_validate(predictions_csv)

    overall_acc   = _overall_accuracy(df)
    per_type_acc  = _per_type_accuracy(df)
    avg_inf_time  = _average_inference_time(df)
    failure_rate  = round(1.0 - overall_acc, 4)

    _save_metrics_csv(overall_acc, per_type_acc, avg_inf_time, failure_rate)
    _save_failure_cases(df)
    _plot_question_type_accuracy(per_type_acc)
    _plot_inference_time(df)

    summary = {
        "overall_accuracy":  overall_acc,
        "per_type_accuracy": per_type_acc,
        "avg_inference_time": avg_inf_time,
        "failure_rate":       failure_rate,
        "total_rows":         len(df),
    }

    _print_report(summary)
    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_and_validate(path: str) -> pd.DataFrame:
    required = {"id", "question_type", "ground_truth_answer",
                "predicted_answer", "is_correct", "inference_time_sec"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {missing}")
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
    df["inference_time_sec"] = pd.to_numeric(df["inference_time_sec"], errors="coerce").fillna(0.0)
    return df


def _overall_accuracy(df: pd.DataFrame) -> float:
    return round(df["is_correct"].mean(), 4)


def _per_type_accuracy(df: pd.DataFrame) -> dict:
    result = {}
    for qtype, group in df.groupby("question_type"):
        total   = len(group)
        correct = group["is_correct"].sum()
        result[qtype] = {
            "total":    total,
            "correct":  int(correct),
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        }
    return result


def _average_inference_time(df: pd.DataFrame) -> float:
    return round(df["inference_time_sec"].mean(), 4)


def _save_metrics_csv(overall_acc, per_type_acc, avg_inf_time, failure_rate):
    rows = []
    for qtype, stats in per_type_acc.items():
        rows.append({
            "question_type":    qtype,
            "total":            stats["total"],
            "correct":          stats["correct"],
            "accuracy":         stats["accuracy"],
            "overall_accuracy": overall_acc,
            "avg_inference_time_sec": avg_inf_time,
            "failure_rate":     failure_rate,
        })
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"[evaluation] Metrics saved to {METRICS_CSV}")


def _save_failure_cases(df: pd.DataFrame):
    failures = df[df["is_correct"] == 0].copy()
    failures["failure_category"] = failures.apply(_classify_failure, axis=1)
    cols = ["id", "image_path", "question", "question_type",
            "ground_truth_answer", "predicted_answer",
            "inference_time_sec", "failure_category"]
    # Only keep columns that exist
    cols = [c for c in cols if c in failures.columns]
    failures[cols].to_csv(FAILURE_CSV, index=False)
    print(f"[evaluation] {len(failures)} failure cases saved to {FAILURE_CSV}")


def _classify_failure(row) -> str:
    """
    Rule-based failure categorization based on question type.
    Maps to the 5 failure types defined in the project plan.
    """
    qtype = str(row.get("question_type", "")).lower()
    mapping = {
        "counting":      "counting_error",
        "yes_no":        "ambiguous_object_presence",
        "color":         "color_confusion",
        "object":        "object_recognition_error",
        "action":        "action_scene_misclassification",
        "spatial_scene": "spatial_reasoning_error",
    }
    return mapping.get(qtype, "other")


def _plot_question_type_accuracy(per_type_acc: dict):
    qtypes     = list(per_type_acc.keys())
    accuracies = [per_type_acc[q]["accuracy"] * 100 for q in qtypes]

    colors = ["#4CAF50" if a >= 70 else "#FF9800" if a >= 50 else "#F44336"
              for a in accuracies]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(qtypes, accuracies, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Question Type", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Question-Type Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=70, color="gray", linestyle="--", linewidth=1, label="70% target")
    ax.legend(fontsize=10)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(QTYPE_CHART, dpi=150)
    plt.close()
    print(f"[evaluation] Chart saved to {QTYPE_CHART}")


def _plot_inference_time(df: pd.DataFrame):
    avg_by_type = df.groupby("question_type")["inference_time_sec"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(avg_by_type.index, avg_by_type.values,
           color="#2196F3", edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Question Type", fontsize=12)
    ax.set_ylabel("Avg Inference Time (sec)", fontsize=12)
    ax.set_title("Average Inference Time per Question Type", fontsize=14, fontweight="bold")

    for i, (qtype, val) in enumerate(avg_by_type.items()):
        ax.text(i, val + 0.005, f"{val:.3f}s", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(INFTIME_CHART, dpi=150)
    plt.close()
    print(f"[evaluation] Chart saved to {INFTIME_CHART}")


def _print_report(summary: dict):
    print("\n" + "="*50)
    print("  EVALUATION SUMMARY")
    print("="*50)
    print(f"  Total rows:          {summary['total_rows']}")
    print(f"  Overall accuracy:    {summary['overall_accuracy']*100:.1f}%")
    print(f"  Failure rate:        {summary['failure_rate']*100:.1f}%")
    print(f"  Avg inference time:  {summary['avg_inference_time']}s")
    print("\n  Per-type accuracy:")
    for qtype, stats in summary["per_type_accuracy"].items():
        print(f"    {qtype:<16} {stats['correct']}/{stats['total']}  "
              f"({stats['accuracy']*100:.1f}%)")
    print("="*50 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_predictions()