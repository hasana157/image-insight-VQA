"""
model_loader.py
Member 02 - Core Ownership
Loads and caches the VQA model and processor.
Supports BLIP-VQA (primary) and ViLT (fallback).
"""

import time
import torch
from src.config import PRIMARY_MODEL, FALLBACK_MODEL, DEVICE

# --- Module-level cache so model loads only once ---
_cache = {
    "model_name": None,
    "processor": None,
    "model": None,
}


def load_model(model_name: str = None):
    """
    Load and cache the VQA processor and model.
    Returns (processor, model) tuple.
    Uses module-level cache so the model is only loaded once per session.

    Args:
        model_name: Hugging Face model ID. Defaults to PRIMARY_MODEL from config.

    Returns:
        (processor, model) ready for inference.
    """
    if model_name is None:
        model_name = PRIMARY_MODEL

    # Return from cache if already loaded
    if _cache["model_name"] == model_name and _cache["model"] is not None:
        print(f"[model_loader] Using cached model: {model_name}")
        return _cache["processor"], _cache["model"]

    print(f"[model_loader] Loading model: {model_name} on {DEVICE} ...")
    start = time.time()

    try:
        processor, model = _load_single_model(model_name)
    except Exception as e:
        print(f"[model_loader] Primary model failed: {e}")
        print(f"[model_loader] Trying fallback model: {FALLBACK_MODEL}")
        try:
            processor, model = _load_single_model(FALLBACK_MODEL)
            model_name = FALLBACK_MODEL
        except Exception as e2:
            raise RuntimeError(
                f"Both primary and fallback model loading failed.\n"
                f"Primary error: {e}\nFallback error: {e2}"
            )

    elapsed = time.time() - start
    print(f"[model_loader] Model ready in {elapsed:.1f}s")

    # Store in cache
    _cache["model_name"] = model_name
    _cache["processor"] = processor
    _cache["model"] = model

    return processor, model


def _load_single_model(model_name: str):
    """
    Internal helper: load one specific model by name.
    Detects BLIP vs ViLT and uses the correct Hugging Face class.
    """
    if "blip" in model_name.lower():
        from transformers import BlipProcessor, BlipForQuestionAnswering
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
    elif "vilt" in model_name.lower():
        from transformers import ViltProcessor, ViltForQuestionAnswering
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Add its loader class here.")

    model.to(DEVICE)
    model.eval()
    return processor, model


def get_active_model_name() -> str:
    """Return the currently loaded model name, or None if not loaded."""
    return _cache["model_name"]


def clear_cache():
    """Clear the model cache to free GPU/CPU memory."""
    _cache["model_name"] = None
    _cache["processor"] = None
    _cache["model"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[model_loader] Cache cleared.")