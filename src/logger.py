"""Prediction logging helpers.

Member 03 — logs every demo/UI prediction to data/predictions.csv.
"""

import csv
import os
from datetime import datetime


def log_demo_result(
    image_path: str,
    question: str,
    answer: str,
    question_type: str,
    inference_time_sec: float,
    predictions_csv: str = "data/predictions.csv",
) -> None:
    """Append a demo prediction row to the predictions CSV.

    If the CSV does not exist, it creates one with a header row.
    """
    file_exists = os.path.isfile(predictions_csv)

    fieldnames = [
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
    ]

    # Determine next id
    next_id = 1
    if file_exists:
        try:
            with open(predictions_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ids = []
                for row in reader:
                    try:
                        ids.append(int(row.get("id", 0)))
                    except (ValueError, TypeError):
                        pass
                if ids:
                    next_id = max(ids) + 1
        except Exception:
            pass

    row = {
        "id": next_id,
        "image_path": image_path,
        "question": question,
        "question_type": question_type,
        "ground_truth_answer": "",
        "source": "demo_ui",
        "predicted_answer": answer,
        "is_correct": "",
        "inference_time_sec": round(inference_time_sec, 4),
        "notes": f"demo run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    }

    with open(predictions_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
