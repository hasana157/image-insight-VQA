# VQA Project Task Checklist

## Member 01 Checklist

### Phase 0: Setup and Alignment

- [ ] Confirm project title and scope with team.
- [ ] Work from branch `phase-1-dataset-preprocessing`.
- [ ] Read the full project plan and understand the pipeline.
- [ ] Help verify that one sample VQA model can run on one image.
- [ ] Keep README setup instructions accurate.

### Phase 1: Dataset, Taxonomy, and Preprocessing

- [x] Select official dataset source: VQA v2 + MS COCO 2014 train/validation images.
- [x] Add minimal downloader for VQA v2 validation metadata and selected COCO images.
- [x] Download VQA v2 validation questions and annotations into `data/raw/vqa/`.
- [x] Download only selected COCO val2014 images into `data/sample_images/`.
- [x] Build project subset with `python scripts/download_vqa_minimal.py`.
- [x] Create at least 30 image-question-answer rows in `data/vqa_test_set.csv`.
- [x] Keep the official subset small enough for a reliable demo and evaluation run.
- [x] Include at least 5 examples per question type.
- [x] Cover these question types: `counting`, `yes_no`, `color`, `object`, `action`, `spatial_scene`.
- [x] Add at least 5 hard examples: clutter, occlusion, ambiguous color, multiple objects, background confusion.
- [x] Verify every image path in the CSV exists.
- [x] Review ground-truth answers for short, consistent wording.
- [x] Implement or refine image validation in `src/preprocessing.py`.
- [x] Implement or refine question taxonomy rules in `src/question_types.py`.
- [x] Add notes for any weak or ambiguous examples.
- [x] Document image/data sources in `data/IMAGE_SOURCES.md`.
- [x] Document taxonomy and answer rules in `data/QUESTION_TAXONOMY.md`.
- [x] Complete Member 01 to Member 02 handoff.

### Member 01 Support in Later Phases

- [ ] Test at least 5 image-question pairs after Member 02 connects the model.
- [ ] Review predictions for dataset quality and answer normalization issues.
- [ ] Help classify at least 5 failure cases.
- [ ] Write report sections for dataset selection, question types, and preprocessing.
- [ ] Run the final app locally before submission.
- [ ] Be ready to explain dataset balance, preprocessing, and one failure case.

## Team Checklist

### Code

- [ ] `app.py` launches the Gradio app without errors.
- [ ] `requirements.txt` includes all required libraries.
- [ ] Model loads once and is reused.
- [ ] `answer_question()` works for a custom image and question.
- [ ] Invalid inputs are handled gracefully.
- [ ] README has install and run commands.

### Dataset

- [x] At least 30 image-question-answer pairs are present.
- [x] At least 5 question types are represented.
- [x] Every image path is valid.
- [x] Ground-truth answers are reviewed for short consistent wording.
- [x] Test set contains easy and challenging examples.

### Evaluation

- [ ] Overall accuracy is calculated.
- [ ] Per-question-type accuracy is calculated.
- [ ] Average inference time is calculated.
- [ ] Failure cases are saved and explained.
- [ ] Charts are exported for the report.

### Report and Demo

- [ ] 3 to 5 page report is completed.
- [ ] Architecture diagram is included.
- [ ] Screenshots are included.
- [ ] 2-minute demo video is recorded.
- [ ] Live demo is tested from a fresh clone.
- [ ] Backup screenshots are ready in case model download or internet fails.
