# Member 01 to Member 02 Handoff

## Completed Work

- [ ] Added curated images to `data/sample_images/`.
- [ ] Filled `data/vqa_test_set.csv`.
- [ ] Balanced question types across the dataset.
- [ ] Added hard examples and notes.
- [ ] Verified image paths.
- [ ] Updated `src/preprocessing.py`.
- [ ] Updated `src/question_types.py`.

## How to Run Checks

```bash
python -m compileall src
```

## Changed Files

- `data/sample_images/`: curated images.
- `data/vqa_test_set.csv`: ground-truth VQA examples.
- `src/preprocessing.py`: image validation and RGB conversion.
- `src/question_types.py`: rule-based question taxonomy.

## Known Issues

- Add any weak examples, uncertain labels, missing images, or ambiguous questions here.

## Next Member Instructions

- Connect the model loader in `src/model_loader.py`.
- Implement `answer_question()` in `src/inference.py`.
- Run at least 10 mixed test cases from the CSV.
- Record model limitations and any answer normalization issues.

## Evidence

- Add screenshot paths, terminal output, or CSV sample notes here.
