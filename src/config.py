"""Project configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
TEST_SET_CSV = DATA_DIR / "vqa_test_set.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL_NAME = "Salesforce/blip-vqa-base"
FALLBACK_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MAX_IMAGE_SIZE = (1024, 1024)
