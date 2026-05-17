"""Validate the curated VQA test dataset for Member 01."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_image
from src.config import PROJECT_ROOT, SAMPLE_IMAGES_DIR
from src.question_types import QUESTION_TYPES, classify_question_type

REQUIRED_COLUMNS = (
    "id",
    "image_path",
    "question",
    "question_type",
    "ground_truth_answer",
    "source",
    "predicted_answer",
    "is_correct",
    "inference_time_sec",
    "notes",
)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or missing a header row.")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

        return list(reader)


def validate_rows(rows: list[dict[str, str]], min_rows: int, min_per_type: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    referenced_images: set[Path] = set()
    type_counts: Counter[str] = Counter()
    hard_count = 0

    if len(rows) < min_rows:
        errors.append(f"Dataset has {len(rows)} rows but needs at least {min_rows}.")

    for index, row in enumerate(rows, start=2):
        row_id = row["id"].strip()
        image_path = row["image_path"].strip()
        question = row["question"].strip()
        question_type = row["question_type"].strip()
        answer = row["ground_truth_answer"].strip()
        source = row["source"].strip()
        notes = row["notes"].strip().lower()

        if not row_id:
            errors.append(f"Line {index}: id is empty.")
        elif row_id in ids:
            errors.append(f"Line {index}: duplicate id '{row_id}'.")
        ids.add(row_id)

        if not image_path:
            errors.append(f"Line {index}: image_path is empty.")
        else:
            try:
                preprocess_image(image_path)
                referenced_images.add((PROJECT_ROOT / image_path).resolve())
            except Exception as exc:
                errors.append(f"Line {index}: image_path '{image_path}' failed validation: {exc}")

        if not question:
            errors.append(f"Line {index}: question is empty.")

        if question_type not in QUESTION_TYPES:
            errors.append(f"Line {index}: invalid question_type '{question_type}'.")
        else:
            type_counts[question_type] += 1
            detected_type = classify_question_type(question)
            if detected_type != question_type:
                warnings.append(
                    f"Line {index}: question_type is '{question_type}' but detector returns '{detected_type}'."
                )

        if not answer:
            errors.append(f"Line {index}: ground_truth_answer is empty.")

        if not source:
            warnings.append(f"Line {index}: source is empty.")
        elif not source.startswith("VQA v2"):
            warnings.append(f"Line {index}: source '{source}' is not an official VQA v2 source label.")

        if "hard:" in notes:
            hard_count += 1

    for question_type in QUESTION_TYPES:
        count = type_counts[question_type]
        if count < min_per_type:
            errors.append(f"Question type '{question_type}' has {count} rows but needs at least {min_per_type}.")

    if hard_count < 5:
        warnings.append(f"Dataset marks {hard_count} hard examples; target is at least 5.")

    sample_images = {
        path.resolve()
        for path in SAMPLE_IMAGES_DIR.glob("COCO_val2014_*.jpg")
        if path.is_file()
    }
    unused_images = sorted(sample_images - referenced_images)
    if unused_images:
        warnings.append(f"{len(unused_images)} COCO sample image files are not referenced by the CSV.")

    return errors, warnings


def print_summary(rows: list[dict[str, str]]) -> None:
    type_counts = Counter(row["question_type"].strip() for row in rows)
    hard_count = sum(1 for row in rows if "hard:" in row["notes"].lower())
    unique_images = {row["image_path"].strip() for row in rows}

    print(f"Rows: {len(rows)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Hard examples: {hard_count}")
    print("Question type counts:")
    for question_type in QUESTION_TYPES:
        print(f"  {question_type}: {type_counts[question_type]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the VQA test set CSV and image files.")
    parser.add_argument("--csv", default="data/vqa_test_set.csv", help="Path to the VQA test CSV.")
    parser.add_argument("--min-rows", type=int, default=30, help="Minimum required dataset rows.")
    parser.add_argument("--min-per-type", type=int, default=5, help="Minimum rows required per question type.")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    rows = read_rows(csv_path)
    errors, warnings = validate_rows(rows, args.min_rows, args.min_per_type)

    print_summary(rows)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
