"""Central configuration for the VQA project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VQA_RAW_DIR = RAW_DATA_DIR / "vqa"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
TEST_SET_CSV = DATA_DIR / "vqa_test_set.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# String path aliases used by Member 02 scripts and notebooks.
TEST_CSV_PATH = "data/vqa_test_set.csv"
PREDICTIONS_PATH = "data/predictions.csv"

# Model selection.
PRIMARY_MODEL = "Salesforce/blip-vqa-base"
FALLBACK_MODEL = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_MODEL_NAME = PRIMARY_MODEL
FALLBACK_MODEL_NAME = FALLBACK_MODEL

# Runtime. Use VQA_DEVICE=cpu or VQA_DEVICE=cuda to force a device.
DEVICE = "auto"
MAX_ANSWER_LENGTH = 10


def get_device() -> str:
    """Resolve the runtime device only when model code needs Torch."""
    requested_device = DEVICE.strip().lower()
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except Exception:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"

# Member 01 image preprocessing settings.
DEFAULT_MAX_IMAGE_SIZE = (1024, 1024)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Synonym map for answer normalization.
ANSWER_SYNONYMS = {
    "bicycles": "bikes",
    "bicycle": "bikes",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycles",
    "sofa": "couch",
    "settee": "couch",
    "automobile": "car",
    "auto": "car",
    "cab": "taxi",
    "plane": "airplane",
    "aeroplane": "airplane",
    "tv": "television",
    "telly": "television",
    "fridge": "refrigerator",
    "specs": "glasses",
    "spectacles": "glasses",
    "jumping": "riding",
    "leaping": "jumping",
}
