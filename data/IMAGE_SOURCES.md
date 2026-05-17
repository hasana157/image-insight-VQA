# Official Dataset Sources

The project dataset source is VQA v2 with MS COCO 2014 images.

Raw VQA metadata downloads stay in `data/raw/`, which is ignored by Git. The selected COCO images used by the project are copied into `data/sample_images/` and committed with the repo.

## Minimal Official Download

For this lab project, we do not need the full COCO image ZIP files. The pre-trained model is not being trained from scratch, so the project uses a balanced official VQA v2 validation subset.

```bash
python scripts/download_vqa_minimal.py --rows-per-type 8 --min-consensus 7 --max-rows-per-image 3 --clean-unused-images
python scripts/validate_dataset.py
python scripts/profile_dataset.py --charts
```

This downloads:

- `data/raw/vqa/v2_Questions_Val_mscoco.zip`
- `data/raw/vqa/v2_Annotations_Val_mscoco.zip`
- only the selected COCO `val2014` image files in `data/sample_images/`

The final committed dataset contains 48 VQA rows across 34 selected COCO `val2014` images.
