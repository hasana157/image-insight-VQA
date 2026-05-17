# VQA Project Task Checklist

## Member 01 Checklist

### Phase 0: Setup and Alignment
- [ ] Confirm project title and scope with team.
- [ ] Work from branch `phase-1-dataset-preprocessing`.
- [ ] Read the full project plan and understand the pipeline.
- [ ] Help verify that one sample VQA model can run on one image.
- [ ] Keep README setup instructions accurate.

### Phase 1: Dataset, Taxonomy, and Preprocessing
- [x] Select official dataset source: VQA v2 validation + MS COCO 2014 validation images.
- [x] Add minimal downloader for VQA v2 validation metadata and selected COCO images.
- [x] Download VQA v2 validation questions and annotations into `data/raw/vqa/`.
- [x] Download only selected COCO val2014 images into `data/sample_images/`.
- [x] Build project subset with `python scripts/download_vqa_minimal.py`.
- [x] Create 48 image-question-answer rows in `data/vqa_test_set.csv`.
- [x] Keep the official subset small enough for a reliable demo and evaluation run.
- [x] Include 8 examples per question type.
- [x] Cover these question types: `counting`, `yes_no`, `color`, `object`, `action`, `spatial_scene`.
- [x] Add at least 5 hard examples: clutter, occlusion, ambiguous color, multiple objects, background confusion.
- [x] Verify every image path in the CSV exists.
- [x] Review ground-truth answers for short, consistent wording.
- [x] Implement or refine image validation in `src/preprocessing.py`.
- [x] Implement or refine question taxonomy rules in `src/question_types.py`.
- [x] Add notes for any weak or ambiguous examples.
- [x] Document image/data sources in `data/IMAGE_SOURCES.md`.
- [x] Document taxonomy and answer rules in `data/QUESTION_TAXONOMY.md`.
- [x] Generate dataset profile and charts for report evidence.
- [x] Complete Member 01 to Member 02 handoff.

### Member 01 Support in Later Phases
- [ ] Test at least 5 image-question pairs after Member 02 connects the model.
- [ ] Review predictions for dataset quality and answer normalization issues.
- [ ] Help classify at least 5 failure cases.
- [ ] Write report sections for dataset selection, question types, and preprocessing.
- [ ] Run the final app locally before submission.
- [ ] Be ready to explain dataset balance, preprocessing, and one failure case.

---

## Member 02 Checklist

### Phase 2: Model Loading and Inference Pipeline
- [x] Implement `src/model_loader.py` with processor and model loading and caching.
- [x] Implement `answer_question(image_path, question)` in `src/inference.py`.
- [x] Add model fallback config: BLIP-VQA primary, ViLT backup if memory fails.
- [x] Add answer normalization: lowercase, punctuation removal, yes/no standardization.
- [x] Add synonym normalization: bicycles/bikes, jumping/riding, and common variants.
- [x] Add `src/config.py` with central model names, device, and path settings.
- [x] Run batch inference on all 48 test cases with `run_batch_inference()`.
- [x] Save predictions to `data/predictions.csv`.
- [x] Run full evaluation and generate all metrics with `src/evaluation.py`.
- [x] Achieve overall accuracy of 93.8% (45/48 correct).
- [x] Generate `results/metrics_summary.csv`.
- [x] Generate `results/question_type_accuracy.png`.
- [x] Generate `results/inference_time_chart.png`.
- [x] Generate `results/failure_cases.csv` with 3 documented failure cases.
- [x] Complete Member 02 to Member 03 handoff.

### Phase 2 Results Summary
| Metric | Result |
|---|---|
| Overall Accuracy | 93.8% (45/48) |
| Average Inference Time | 0.1225s |
| action | 8/8 (100.0%) |
| yes_no | 8/8 (100.0%) |
| object | 8/8 (100.0%) |
| spatial_scene | 8/8 (100.0%) |
| color | 7/8 (87.5%) |
| counting | 6/8 (75.0%) |

### Documented Failure Cases
- Counting: predicted 3 giraffes, GT was 2 — partial occlusion
- Counting: predicted 8 pictures, GT was 7 — cluttered scene
- Color: predicted red church, GT was brown — lighting ambiguity

### Member 02 Support in Later Phases
- [ ] Test model stability and fallback model before final demo.
- [ ] Write report sections for model selection, inference pipeline, and results.
- [ ] Run the final app locally before submission.
- [ ] Be ready to explain BLIP/VQA model flow, processor logic, inference time, and limitations.

---

## Team Checklist

### Code
- [ ] `app.py` launches the Gradio app without errors.
- [ ] `requirements.txt` includes all required libraries.
- [x] Model loads once and is reused.
- [x] `answer_question()` works for a custom image and question.
- [ ] Invalid inputs are handled gracefully.
- [ ] README has install and run commands.

### Dataset
- [x] 48 image-question-answer pairs are present.
- [x] 6 question types are represented.
- [x] Every image path is valid.
- [x] Ground-truth answers are reviewed for short consistent wording.
- [x] Test set contains easy and challenging examples.

### Evaluation
- [x] Overall accuracy is calculated.
- [x] Per-question-type accuracy is calculated.
- [x] Average inference time is calculated.
- [x] Failure cases are saved and explained.
- [x] Charts are exported for the report.

### Report and Demo
- [ ] 3 to 5 page report is completed.
- [ ] Architecture diagram is included.
- [ ] Screenshots are included.
- [ ] 2-minute demo video is recorded.
- [ ] Live demo is tested from a fresh clone.
- [ ] Backup screenshots are ready in case model download or internet fails.
