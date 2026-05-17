# image-insight-VQA

A multimodal Visual Question Answering system that answers natural-language questions about images using BLIP and Hugging Face Transformers.

## Project Goal

Build a clean VQA pipeline where a user uploads an image, asks a natural-language question, and receives:

- predicted answer
- detected question type
- inference time
- optional logging for evaluation

## Planned Stack

- Python
- PyTorch
- Hugging Face Transformers
- BLIP / BLIP-VQA
- PIL and OpenCV
- Gradio
- Pandas
- Matplotlib

## Folder Structure

```text
image-insight-VQA/
|-- app.py
|-- requirements.txt
|-- README.md
|-- TASK_CHECKLIST.md
|-- .gitignore
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- preprocessing.py
|   |-- model_loader.py
|   |-- inference.py
|   |-- question_types.py
|   |-- evaluation.py
|   |-- logger.py
|   |-- utils.py
|
|-- data/
|   |-- sample_images/
|   |-- vqa_test_set.csv
|   |-- predictions.csv
|
|-- results/
|   |-- metrics_summary.csv
|
|-- notebooks/
|   |-- exploration_and_model_testing.ipynb
|
|-- report/
|   |-- final_report_outline.md
|   |-- screenshots/
|
|-- demo/
|   |-- demo_script.txt
|
|-- handoff/
|   |-- member01_to_member02.md
|   |-- member02_to_member03.md
|   |-- member03_to_team.md
|   |-- final_submission_checklist.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Member 01 Focus

Member 01 owns the dataset, question taxonomy, image preprocessing, ground-truth quality, and the first handoff. Start with:

1. Add curated images to `data/sample_images/`.
2. Fill `data/vqa_test_set.csv` with 30 to 50 image-question-answer rows.
3. Keep at least 5 examples for each question type.
4. Use `src/preprocessing.py` to validate image paths and image formats.
5. Use `src/question_types.py` to keep question categories consistent.
6. Update `handoff/member01_to_member02.md` before passing work to Member 02.
