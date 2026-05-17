# Member 02 to Member 03 Handoff

## Completed Work

- Implemented `src/model_loader.py`: loads and caches BLIP-VQA (primary) or ViLT (fallback). Model loads once per session.
- Implemented `src/inference.py`: `answer_question()` returns predicted answer + inference time. Includes answer normalization and batch inference runner.
- Implemented `src/evaluation.py`: runs full test CSV, computes all metrics, saves charts and failure cases to `results/`.
- Updated `src/config.py`: central config for model names, device, and paths.
- Ran batch inference on all 48 rows. Predictions saved to `data/predictions.csv`.
- Generated all evaluation outputs in `results/`.

## How to Run

Install dependencies first (if not already done):
```bash
pip install torch torchvision transformers Pillow pandas matplotlib gradio
```

**Load model and test one question:**
```python
from src.inference import answer_question
answer, time_sec = answer_question("data/sample_images/img001.jpg", "How many people are visible?")
print(answer, time_sec)
```

**Run full batch inference:**
```python
from src.inference import run_batch_inference
run_batch_inference("data/vqa_test_set.csv")
```

**Run evaluation (after batch inference):**
```bash
python -m src.evaluation
```
Or in Python:
```python
from src.evaluation import evaluate_predictions
summary = evaluate_predictions("data/predictions.csv")
```

**Expected terminal output:**
```
==================================================
  EVALUATION SUMMARY
==================================================
  Total rows:          48
  Overall accuracy:    93.8%
  Failure rate:        6.2%
  Avg inference time:  0.1225s

  Per-type accuracy:
  action           8/8  (100.0%)
  color            7/8  (87.5%)
  counting         6/8  (75.0%)
  object           8/8  (100.0%)
  spatial_scene    8/8  (100.0%)
  yes_no           8/8  (100.0%)
==================================================
```

## Files Added / Modified

| File | Purpose |
|---|---|
| `src/config.py` | Central config — model names, device, paths |
| `src/model_loader.py` | Model loading and caching |
| `src/inference.py` | `answer_question()`, normalization, batch runner |
| `src/evaluation.py` | Metrics, charts, failure case export |
| `data/predictions.csv` | Model predictions with is_correct and inference_time_sec filled |
| `results/metrics_summary.csv` | Per-type accuracy table |
| `results/question_type_accuracy.png` | Bar chart for report |
| `results/inference_time_chart.png` | Inference time bar chart |
| `results/failure_cases.csv` | All wrong predictions with failure category |

## Function Signatures for app.py

Import and call these exactly in your Gradio app:

```python
from src.model_loader import load_model
from src.inference import answer_question
from src.question_types import classify_question_type

# Call once at app startup (outside gr.Blocks) to pre-load model:
load_model()

# Call on every user submission:
answer, inference_time = answer_question(image_path, question)
question_type = classify_question_type(question)
```

**`answer_question` input/output:**
- Input: `image_path` (str, path to image file), `question` (str)
- Output: `(predicted_answer: str, inference_time_sec: float)`
- Raises `ValueError` if image_path or question is empty
- Raises `FileNotFoundError` if image does not exist

## Known Issues / Limitations

- Counting: predicted 3 giraffes, GT was 2 (partial occlusion)
- Counting: predicted 8 pictures, GT was 7 (cluttered scene)  
- Color: predicted red church, GT was brown (lighting ambiguity)
- If imports fail after model reload, clear module cache first:
```python
  for mod in list(sys.modules.keys()):
      if "src" in mod: del sys.modules[mod]
```

## Next Member Instructions

1. Import `load_model()` and call it **once at Gradio app startup** (before `demo.launch()`).
2. In your submit handler, call `answer_question(image_path, question)` and display both the answer and inference time.
3. Also call `classify_question_type(question)` from `src.question_types` to show the reasoning category in the UI.
4. Use `results/question_type_accuracy.png` and `results/inference_time_chart.png` in the results panel.
5. Log every demo run using `src/logger.py` (append to `data/predictions.csv`).
6. Test with at least one example from each of the 6 question types before final submission.

## Evidence

- `data/predictions.csv` — filled with 48 predictions
- `results/metrics_summary.csv` — accuracy table
- `results/question_type_accuracy.png` — bar chart
- `results/inference_time_chart.png` — timing chart
- `results/failure_cases.csv` — failure analysis