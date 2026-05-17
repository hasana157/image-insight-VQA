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
|   |-- IMAGE_SOURCES.md
|   |-- QUESTION_TAXONOMY.md
|
|-- scripts/
|   |-- download_vqa_minimal.py
|   |-- profile_dataset.py
|   |-- validate_dataset.py
|
|-- results/
|   |-- metrics_summary.csv
|   |-- dataset_profile.csv
|   |-- dataset_question_type_distribution.png
|   |-- dataset_answer_type_distribution.png
|
|-- notebooks/
|   |-- exploration_and_model_testing.ipynb
|
|-- report/
|   |-- final_report_outline.md
|   |-- dataset_profile.md
|   |-- member01_dataset_preprocessing.md
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

Member 01 owns the dataset, question taxonomy, image preprocessing, ground-truth quality, and the first handoff.

For this project, we use a pre-trained VQA model. Member 01 does not train the model. Member 01 prepares the official evaluation/demo subset that Member 02 will run through the pre-trained model.

1. Download the minimal official VQA v2 validation metadata and selected COCO val2014 images.
2. Build `data/vqa_test_set.csv` with 48 image-question-answer rows.
3. Keep 8 examples for each question type.
4. Use `src/preprocessing.py` to validate image paths and image formats.
5. Use `src/question_types.py` to keep question categories consistent.
6. Update `handoff/member01_to_member02.md` before passing work to Member 02.

## Official Dataset Setup

Raw dataset files are ignored by Git under `data/raw/`.

Recommended minimal download:

```bash
python scripts/download_vqa_minimal.py --rows-per-type 8 --min-consensus 7 --max-rows-per-image 3 --clean-unused-images
python scripts/validate_dataset.py
python scripts/profile_dataset.py --charts
```

This downloads only:

- VQA v2 validation questions ZIP
- VQA v2 validation annotations ZIP
- selected COCO val2014 image files used by `data/vqa_test_set.csv`

## Dataset Validation

Member 01 can check the dataset with:

```bash
python scripts/validate_dataset.py
```

The validator checks row count, image paths, question-type balance, ground-truth answers, sources, and hard-example notes.
