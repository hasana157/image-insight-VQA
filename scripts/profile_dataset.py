"""Create a compact data quality profile for the VQA evaluation set."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, TEST_SET_CSV


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def extract_note_value(note: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^;]+)", note)
    return match.group(1).strip() if match else "unknown"


def image_stats(rows: list[dict[str, str]]) -> tuple[dict[str, tuple[int, int, int]], list[str]]:
    stats: dict[str, tuple[int, int, int]] = {}
    missing: list[str] = []

    for image_path in sorted({row["image_path"] for row in rows}):
        path = PROJECT_ROOT / image_path
        if not path.exists():
            missing.append(image_path)
            continue

        with Image.open(path) as image:
            width, height = image.size
        stats[image_path] = (width, height, path.stat().st_size)

    return stats, missing


def write_profile_csv(summary_rows: list[tuple[str, str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "notes"])
        writer.writerows(summary_rows)


def write_profile_md(
    rows: list[dict[str, str]],
    summary_rows: list[tuple[str, str, str]],
    question_counts: Counter[str],
    answer_type_counts: Counter[str],
    output_md: Path,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Dataset Profile",
        "",
        "## Summary",
        "",
        "| Metric | Value | Notes |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {metric} | {value} | {notes} |" for metric, value, notes in summary_rows)

    lines.extend(["", "## Question Type Distribution", "", "| Type | Rows |", "| --- | ---: |"])
    lines.extend(f"| `{question_type}` | {count} |" for question_type, count in sorted(question_counts.items()))

    lines.extend(["", "## VQA Answer Type Distribution", "", "| Answer Type | Rows |", "| --- | ---: |"])
    lines.extend(f"| `{answer_type}` | {count} |" for answer_type, count in sorted(answer_type_counts.items()))

    lines.extend(["", "## Sample Rows", "", "| ID | Type | Question | Answer |", "| --- | --- | --- | --- |"])
    for row in rows[:8]:
        question = row["question"].replace("|", "\\|")
        answer = row["ground_truth_answer"].replace("|", "\\|")
        lines.append(f"| {row['id']} | `{row['question_type']}` | {question} | {answer} |")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_charts(question_counts: Counter[str], answer_type_counts: Counter[str], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    for filename, title, counts in (
        ("dataset_question_type_distribution.png", "Question Type Distribution", question_counts),
        ("dataset_answer_type_distribution.png", "VQA Answer Type Distribution", answer_type_counts),
    ):
        labels = list(counts.keys())
        values = [counts[label] for label in labels]
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values, color="#3274a1")
        plt.title(title)
        plt.ylabel("Rows")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        chart_paths.append(path)

    return chart_paths


def build_profile(rows: list[dict[str, str]]) -> tuple[list[tuple[str, str, str]], Counter[str], Counter[str]]:
    question_counts = Counter(row["question_type"] for row in rows)
    image_counts = Counter(row["image_path"] for row in rows)
    answer_counts = Counter(row["ground_truth_answer"] for row in rows)
    answer_type_counts = Counter(extract_note_value(row["notes"], "answer_type") for row in rows)
    consensus_values = [
        int(match.group(1))
        for row in rows
        for match in [re.search(r"consensus=(\d+)/10", row["notes"])]
        if match
    ]
    hard_count = sum(1 for row in rows if "hard:" in row["notes"].lower())
    duplicate_pairs = len(rows) - len({(row["image_path"], row["question"]) for row in rows})
    stats, missing_images = image_stats(rows)
    total_image_bytes = sum(size for _, _, size in stats.values())

    summary_rows = [
        ("total_rows", str(len(rows)), "VQA image-question-answer pairs"),
        ("unique_images", str(len(stats)), "Selected COCO val2014 images"),
        ("hard_examples", str(hard_count), "Rows marked for reasoning/failure analysis"),
        ("question_types", str(len(question_counts)), "Balanced taxonomy categories"),
        ("min_rows_per_type", str(min(question_counts.values())), "Smallest category count"),
        ("max_rows_per_image", str(max(image_counts.values())), "Limits image repetition"),
        ("duplicate_image_question_pairs", str(duplicate_pairs), "Should be 0"),
        ("missing_images", str(len(missing_images)), "Should be 0"),
        ("unique_answers", str(len(answer_counts)), "Answer vocabulary size"),
        ("avg_consensus", f"{sum(consensus_values) / len(consensus_values):.2f}/10", "VQA annotator agreement"),
        ("image_storage_mb", f"{total_image_bytes / (1024 * 1024):.2f}", "Committed selected images only"),
    ]

    return summary_rows, question_counts, answer_type_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the curated VQA dataset.")
    parser.add_argument("--csv", type=Path, default=TEST_SET_CSV)
    parser.add_argument("--output-csv", type=Path, default=RESULTS_DIR / "dataset_profile.csv")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "report" / "dataset_profile.md")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    summary_rows, question_counts, answer_type_counts = build_profile(rows)
    write_profile_csv(summary_rows, args.output_csv)
    write_profile_md(rows, summary_rows, question_counts, answer_type_counts, args.output_md)

    chart_paths = maybe_write_charts(question_counts, answer_type_counts, RESULTS_DIR) if args.charts else []

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    for path in chart_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
