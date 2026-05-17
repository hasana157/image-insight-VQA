"""Download only the VQA v2 subset needed for this project.

This avoids the huge COCO 2014 image ZIP files. It downloads the VQA v2
validation question/annotation ZIPs, selects a balanced project subset, then
downloads only the selected COCO val2014 image files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_IMAGES_DIR, TEST_SET_CSV, VQA_RAW_DIR
from src.question_types import QUESTION_TYPES, classify_question_type
from src.utils import normalize_answer

QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
ANNOTATIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
QUESTIONS_ZIP = VQA_RAW_DIR / "v2_Questions_Val_mscoco.zip"
ANNOTATIONS_ZIP = VQA_RAW_DIR / "v2_Annotations_Val_mscoco.zip"
QUESTIONS_JSON = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_JSON = "v2_mscoco_val2014_annotations.json"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2014/{filename}"

CSV_COLUMNS = (
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

COLOR_ANSWERS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
ACTION_HINTS = {
    "sitting",
    "standing",
    "walking",
    "running",
    "playing",
    "riding",
    "holding",
    "eating",
    "sleeping",
    "surfing",
    "skiing",
    "skateboarding",
    "snowboarding",
    "flying",
    "swimming",
    "jumping",
}
QUESTION_EXCLUSION_TERMS = {
    "brand",
    "copyright",
    "letter",
    "license plate",
    "logo",
    "number is on",
    "number on",
    "read",
    "sentence",
    "sign",
    "slogan",
    "text",
    "website",
    "who ",
    "will ",
    "word",
    "written",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return

    temp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "image-insight-VQA/1.0"})
    with urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temp_path.replace(destination)


def load_json_from_zip(zip_path: Path, member_name: str) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as file:
            return json.load(file)


def image_filename(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def answer_consensus(annotation: dict) -> tuple[str, int]:
    answers = [normalize_answer(answer["answer"]) for answer in annotation.get("answers", [])]
    counts = Counter(answers)
    multiple_choice_answer = normalize_answer(annotation["multiple_choice_answer"])
    return multiple_choice_answer, counts[multiple_choice_answer]


def is_good_candidate(question_type: str, answer: str, consensus: int, min_consensus: int) -> bool:
    if consensus < min_consensus:
        return False

    if len(answer) > 24 or not answer:
        return False

    if question_type == "counting":
        return answer.isdigit()

    if question_type == "yes_no":
        return answer in {"yes", "no"}

    if question_type == "color":
        return answer in COLOR_ANSWERS

    if question_type == "action":
        return any(hint in answer for hint in ACTION_HINTS)

    return True


def is_model_friendly_question(question: str) -> bool:
    normalized = question.lower()
    return not any(term in normalized for term in QUESTION_EXCLUSION_TERMS)


def note_for(question_type: str, annotation: dict, consensus: int) -> str:
    answer_type = annotation.get("answer_type", "unknown")
    note = f"official VQA v2 val2014; answer_type={answer_type}; consensus={consensus}/10"
    if question_type in {"counting", "action", "spatial_scene"}:
        return f"hard: reasoning-sensitive example; {note}"
    return note


def select_rows(rows_per_type: int, min_consensus: int, max_rows_per_image: int) -> list[dict[str, str]]:
    questions = load_json_from_zip(QUESTIONS_ZIP, QUESTIONS_JSON)["questions"]
    annotation_items = load_json_from_zip(ANNOTATIONS_ZIP, ANNOTATIONS_JSON)["annotations"]
    annotations = {annotation["question_id"]: annotation for annotation in annotation_items}

    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    image_counts: Counter[int] = Counter()
    seen_images: set[int] = set()

    for question in questions:
        question_id = question["question_id"]
        annotation = annotations.get(question_id)
        if annotation is None:
            continue

        question_text = question["question"].strip()
        if not is_model_friendly_question(question_text):
            continue

        question_type = classify_question_type(question_text)
        if counts[question_type] >= rows_per_type:
            continue

        image_id = question["image_id"]
        if image_counts[image_id] >= max_rows_per_image:
            continue

        answer, consensus = answer_consensus(annotation)
        if not is_good_candidate(question_type, answer, consensus, min_consensus):
            continue

        filename = image_filename(image_id)

        selected.append(
            {
                "id": str(len(selected) + 1),
                "image_path": str(Path("data") / "sample_images" / filename).replace("\\", "/"),
                "question": question_text,
                "question_type": question_type,
                "ground_truth_answer": answer,
                "source": "VQA v2 val2014",
                "predicted_answer": "",
                "is_correct": "",
                "inference_time_sec": "",
                "notes": note_for(question_type, annotation, consensus),
                "_image_id": str(image_id),
                "_filename": filename,
            }
        )
        counts[question_type] += 1
        image_counts[image_id] += 1
        seen_images.add(image_id)

        if all(counts[question_type] >= rows_per_type for question_type in QUESTION_TYPES):
            break

    missing = [
        f"{question_type}={counts[question_type]}/{rows_per_type}"
        for question_type in QUESTION_TYPES
        if counts[question_type] < rows_per_type
    ]
    if missing:
        raise RuntimeError(f"Could not find enough balanced VQA rows: {', '.join(missing)}")

    print(f"Selected {len(selected)} rows across {len(seen_images)} unique images.")
    for question_type in QUESTION_TYPES:
        print(f"{question_type}: {counts[question_type]}")
    return selected


def download_selected_images(rows: list[dict[str, str]]) -> None:
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = row["_filename"]
        destination = SAMPLE_IMAGES_DIR / filename
        download_file(COCO_IMAGE_URL.format(filename=filename), destination)


def clean_unused_images(rows: list[dict[str, str]]) -> None:
    used_filenames = {row["_filename"] for row in rows}
    for path in SAMPLE_IMAGES_DIR.glob("COCO_val2014_*.jpg"):
        if path.name not in used_filenames:
            path.unlink()
            print(f"Removed unused image: {path}")


def write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a minimal official VQA v2 subset.")
    parser.add_argument("--rows-per-type", type=int, default=8)
    parser.add_argument("--min-consensus", type=int, default=7)
    parser.add_argument("--max-rows-per-image", type=int, default=3)
    parser.add_argument("--output-csv", type=Path, default=TEST_SET_CSV)
    parser.add_argument("--clean-unused-images", action="store_true")
    args = parser.parse_args()

    download_file(QUESTIONS_URL, QUESTIONS_ZIP)
    download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP)

    rows = select_rows(args.rows_per_type, args.min_consensus, args.max_rows_per_image)
    download_selected_images(rows)
    if args.clean_unused_images:
        clean_unused_images(rows)
    write_csv(rows, args.output_csv)

    print(f"Wrote CSV: {args.output_csv}")
    print("Run: python scripts/validate_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
