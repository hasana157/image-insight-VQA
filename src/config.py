"""
config.py
Central configuration for the VQA project.
Edit model names or paths here — nowhere else.
"""

import torch

# ── Model selection ───────────────────────────────────────────────────────────
PRIMARY_MODEL  = "Salesforce/blip-vqa-base"   # ~900MB, needs GPU ideally
FALLBACK_MODEL = "dandelin/vilt-b32-finetuned-vqa"  # ~450MB, CPU-friendly

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Inference ─────────────────────────────────────────────────────────────────
MAX_ANSWER_LENGTH = 10   # Max tokens for BLIP generation

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_CSV_PATH    = "data/vqa_test_set.csv"
PREDICTIONS_PATH = "data/predictions.csv"
RESULTS_DIR      = "results"

# --- Member 01 preprocessing requirements ---
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_IMAGE_SIZE = (1024, 1024)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}