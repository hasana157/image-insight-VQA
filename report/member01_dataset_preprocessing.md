# Member 01 Dataset and Preprocessing Notes

## Dataset Source

The project uses the official VQA v2 validation split with MS COCO 2014 validation images. Because the model is pre-trained, the project does not download the full training image archive or train the model from scratch.

The local subset contains:

- 48 image-question-answer pairs
- 34 selected COCO `val2014` images
- 6 question categories
- 8 examples per category
- VQA v2 `multiple_choice_answer` as the ground-truth answer

## Question Categories

The test set is balanced across:

- `counting`
- `yes_no`
- `color`
- `object`
- `action`
- `spatial_scene`

This balance lets the evaluation report show which reasoning categories the model handles well or poorly.

## Preprocessing

Image preprocessing is implemented in `src/preprocessing.py`.

It checks:

- image file exists
- path points to a file
- extension is supported
- image can be opened by PIL
- image is converted to RGB
- image is resized only if it exceeds the configured maximum size

## Quality Checks

The dataset can be checked with:

```bash
python scripts/validate_dataset.py
python scripts/profile_dataset.py --charts
```

The validator checks row count, required columns, image paths, question type labels, official source labels, ground-truth answers, hard-example notes, and question-type balance.

Current expected result:

```text
Rows: 48
Unique images: 34
Hard examples: 24
counting: 8
yes_no: 8
color: 8
object: 8
action: 8
spatial_scene: 8
Dataset validation passed.
```
