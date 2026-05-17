# Dataset Profile

## Summary

| Metric | Value | Notes |
| --- | --- | --- |
| total_rows | 48 | VQA image-question-answer pairs |
| unique_images | 34 | Selected COCO val2014 images |
| hard_examples | 24 | Rows marked for reasoning/failure analysis |
| question_types | 6 | Balanced taxonomy categories |
| min_rows_per_type | 8 | Smallest category count |
| max_rows_per_image | 3 | Limits image repetition |
| duplicate_image_question_pairs | 0 | Should be 0 |
| missing_images | 0 | Should be 0 |
| unique_answers | 35 | Answer vocabulary size |
| avg_consensus | 9.00/10 | VQA annotator agreement |
| image_storage_mb | 5.24 | Committed selected images only |

## Question Type Distribution

| Type | Rows |
| --- | ---: |
| `action` | 8 |
| `color` | 8 |
| `counting` | 8 |
| `object` | 8 |
| `spatial_scene` | 8 |
| `yes_no` | 8 |

## VQA Answer Type Distribution

| Answer Type | Rows |
| --- | ---: |
| `number` | 8 |
| `other` | 32 |
| `yes/no` | 8 |

## Sample Rows

| ID | Type | Question | Answer |
| --- | --- | --- | --- |
| 1 | `spatial_scene` | Where is he looking? | down |
| 2 | `yes_no` | Is this a creamy soup? | no |
| 3 | `yes_no` | Is this rice noodle soup? | yes |
| 4 | `spatial_scene` | What is to the right of the soup? | chopsticks |
| 5 | `spatial_scene` | What does the truck on the left sell? | ice cream |
| 6 | `yes_no` | Is it daylight in this picture? | yes |
| 7 | `counting` | How many are playing ball? | 1 |
| 8 | `yes_no` | Is there a chain link fence in the image? | no |
