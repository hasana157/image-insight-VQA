# Member 01 to Member 02 Handoff

## Completed Work

- [x] Selected official dataset source: VQA v2 + MS COCO 2014 images.
- [x] Added minimal downloader for VQA v2 validation metadata and selected COCO images.
- [x] Added generated COCO subset images to `data/sample_images/`.
- [x] Filled `data/vqa_test_set.csv` from VQA v2 annotations.
- [x] Balanced question types across the dataset.
- [x] Added hard examples and notes.
- [x] Verified image paths.
- [x] Updated `src/preprocessing.py`.
- [x] Updated `src/question_types.py`.
- [x] Added dataset validator in `scripts/validate_dataset.py`.

## How to Run Checks

```bash
python -m compileall src
python scripts/download_vqa_minimal.py --rows-per-type 5 --min-consensus 7 --max-rows-per-image 3 --clean-unused-images
python scripts/validate_dataset.py
```

## Changed Files

- `data/sample_images/`: generated COCO 2014 subset images.
- `data/vqa_test_set.csv`: ground-truth VQA v2 examples.
- `data/IMAGE_SOURCES.md`: image source list.
- `data/QUESTION_TAXONOMY.md`: question-type definitions and labeling rules.
- `report/member01_dataset_preprocessing.md`: report-ready dataset/preprocessing notes.
- `src/preprocessing.py`: image validation and RGB conversion.
- `src/question_types.py`: rule-based question taxonomy.
- `scripts/download_vqa_minimal.py`: recommended minimal official subset downloader.
- `scripts/validate_dataset.py`: dataset quality checker.

## Known Issues

- Raw VQA metadata files are ignored under `data/raw/`.
- The project uses a pre-trained VQA model, so no model training dataset is required.
- The committed evaluation subset is intentionally small and balanced for fast demo/evaluation.
- Some hard examples may produce weak model answers by design and should be used for failure analysis.

## Next Member Instructions

- Connect the model loader in `src/model_loader.py`.
- Implement `answer_question()` in `src/inference.py`.
- Run at least 10 mixed test cases from the CSV.
- Record model limitations and any answer normalization issues.

## Evidence

- `python scripts/validate_dataset.py` should report 30 rows and at least 5 rows for every question type.
