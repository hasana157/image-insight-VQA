"""Image validation and preprocessing utilities."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import DEFAULT_MAX_IMAGE_SIZE, PROJECT_ROOT, SUPPORTED_IMAGE_EXTENSIONS


def validate_image_path(image_path: str | Path) -> Path:
    """Return a resolved image path after checking existence and extension."""
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image format '{path.suffix}'. Allowed: {allowed}")

    return path


def preprocess_image(
    image_path: str | Path,
    max_size: tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE,
) -> Image.Image:
    """Load an image, convert it to RGB, and resize only if it is too large."""
    path = validate_image_path(image_path)

    try:
        image = Image.open(path)
        image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image file: {path}") from exc

    image.thumbnail(max_size)
    return image
