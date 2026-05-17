# Member 01 Dataset and Preprocessing Notes

## Dataset Source

The project uses the official VQA v2 validation split with MS COCO 2014 validation images. Because the model is pre-trained, the project does not download the full training image archive or train the model from scratch.

The local subset contains:

- 30 image-question-answer pairs
- 20 selected COCO `val2014` images
- 6 question categories
- 5 examples per category
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
```

The validator checks row count, required columns, image paths, question type labels, official source labels, ground-truth answers, hard-example notes, and question-type balance.

Current expected result:

```text
Rows: 30
Unique images: 20
Hard examples: 15
counting: 5
yes_no: 5
color: 5
object: 5
action: 5
spatial_scene: 5
Dataset validation passed.
```
