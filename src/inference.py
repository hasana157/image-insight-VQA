"""
inference.py
Member 02 - Core Ownership
Implements answer_question() — the main inference function.
Handles answer normalization, timing, and error cases.
"""

import time
import torch
from src.config import DEVICE, MAX_ANSWER_LENGTH
from src.model_loader import load_model
from src.preprocessing import preprocess_image
from src.question_types import classify_question_type


def answer_question(image_path: str, question: str) -> tuple[str, float]:
    """
    Main inference function. Accepts an image path and natural-language question,
    returns the predicted answer and inference time.

    Args:
        image_path: Path to the image file (str).
        question:   Natural-language question string.

    Returns:
        (predicted_answer: str, inference_time_sec: float)

    Raises:
        ValueError: If image_path or question is empty/invalid.
        FileNotFoundError: If the image file does not exist.
    """
    # --- Input validation ---
    if not image_path or not str(image_path).strip():
        raise ValueError("image_path must not be empty.")
    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    # --- Load image via Member 1's preprocessor ---
    image = preprocess_image(image_path)  # returns PIL.Image in RGB

    # --- Load (or retrieve cached) model ---
    processor, model = load_model()
    active_model = _get_model_type()

    # --- Run inference with timer ---
    start = time.time()

    if active_model == "blip":
        answer = _run_blip_inference(processor, model, image, question)
    elif active_model == "vilt":
        answer = _run_vilt_inference(processor, model, image, question)
    else:
        raise RuntimeError("No supported model is loaded.")

    inference_time = round(time.time() - start, 4)

    # --- Normalize answer ---
    answer = normalize_answer(answer, question)

    return answer, inference_time


def _run_blip_inference(processor, model, image, question: str) -> str:
    """Run inference using BLIP-VQA model."""
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_ANSWER_LENGTH)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer


def _run_vilt_inference(processor, model, image, question: str) -> str:
    """Run inference using ViLT model."""
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_idx = logits.argmax(-1).item()
    answer = model.config.id2label[predicted_idx]
    return answer


def _get_model_type() -> str:
    """Return 'blip' or 'vilt' based on currently loaded model name."""
    from src.model_loader import get_active_model_name
    name = get_active_model_name() or ""
    if "blip" in name.lower():
        return "blip"
    elif "vilt" in name.lower():
        return "vilt"
    return "unknown"


def normalize_answer(answer: str, question: str = "") -> str:
    from src.config import ANSWER_SYNONYMS
    if not answer:
        return "unknown"
    answer = answer.strip().lower().rstrip(".,!?;:")
    yes_variants = {"yes", "yeah", "yep", "yup", "correct", "true", "right"}
    no_variants  = {"no", "nope", "nah", "false", "wrong", "not"}
    if answer in yes_variants:
        return "yes"
    if answer in no_variants:
        return "no"
    for article in ("a ", "an ", "the "):
        if answer.startswith(article):
            answer = answer[len(article):]
            break
    # Apply synonym normalization
    answer = ANSWER_SYNONYMS.get(answer, answer)
    return answer


def run_batch_inference(test_csv_path: str) -> list[dict]:
    """
    Run answer_question() over every row in the test CSV.
    Saves predicted_answer, is_correct, and inference_time_sec back to each row.

    Args:
        test_csv_path: Path to vqa_test_set.csv

    Returns:
        List of result dicts (one per row).
    """
    import pandas as pd
    import os

    df = pd.read_csv(test_csv_path)
    results = []

    print(f"[inference] Running batch inference on {len(df)} rows...")

    for i, row in df.iterrows():
        image_path = row["image_path"]
        question   = row["question"]
        ground_truth = str(row["ground_truth_answer"]).strip().lower()

        try:
            predicted, inf_time = answer_question(image_path, question)
            is_correct = int(predicted.strip().lower() == ground_truth)
            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:02d}] {status} Q: {question[:50]} | Pred: {predicted} | GT: {ground_truth} | {inf_time}s")
        except Exception as e:
            predicted = "error"
            inf_time  = 0.0
            is_correct = 0
            print(f"  [{i+1:02d}] ERROR on row {i}: {e}")

        df.at[i, "predicted_answer"]  = predicted
        df.at[i, "is_correct"]        = is_correct
        df.at[i, "inference_time_sec"] = inf_time

        results.append({
            "id":                  row["id"],
            "image_path":          image_path,
            "question":            question,
            "question_type":       row["question_type"],
            "ground_truth_answer": ground_truth,
            "predicted_answer":    predicted,
            "is_correct":          is_correct,
            "inference_time_sec":  inf_time,
        })

    # Save predictions CSV
    out_path = os.path.join(os.path.dirname(test_csv_path), "predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[inference] Predictions saved to {out_path}")

    return results