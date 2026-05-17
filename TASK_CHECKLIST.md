# VQA Project Task Checklist

## Member 01 Checklist

### Phase 0: Setup and Alignment

- [ ] Confirm project title and scope with team.
- [ ] Work from branch `phase-1-dataset-preprocessing`.
- [ ] Read the full project plan and understand the pipeline.
- [ ] Help verify that one sample VQA model can run on one image.
- [ ] Keep README setup instructions accurate.

### Phase 1: Dataset, Taxonomy, and Preprocessing

- [ ] Collect approved real-world images for `data/sample_images/`.
- [ ] Create at least 30 image-question-answer rows in `data/vqa_test_set.csv`.
- [ ] Target 30 to 50 total rows for a stronger submission.
- [ ] Include at least 5 examples per question type.
- [ ] Cover these question types: `counting`, `yes_no`, `color`, `object`, `action`, `spatial_scene`.
- [ ] Add at least 5 hard examples: clutter, occlusion, ambiguous color, multiple objects, background confusion.
- [ ] Verify every image path in the CSV exists.
- [ ] Review ground-truth answers for short, consistent wording.
- [ ] Implement or refine image validation in `src/preprocessing.py`.
- [ ] Implement or refine question taxonomy rules in `src/question_types.py`.
- [ ] Add notes for any weak or ambiguous examples.

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

- [ ] At least 30 image-question-answer pairs are present.
- [ ] At least 5 question types are represented.
- [ ] Every image path is valid.
- [ ] Ground-truth answers are reviewed by the team.
- [ ] Test set contains easy and challenging examples.

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
