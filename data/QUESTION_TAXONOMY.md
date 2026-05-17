# Question Taxonomy

The test set uses 6 question types so evaluation can show where the VQA model performs well or poorly.

| Question Type | Meaning | Typical Pattern | Example |
| --- | --- | --- | --- |
| `counting` | Count visible objects or people. | `how many`, `number of`, `count` | How many cats are on the couch? |
| `yes_no` | Answer with yes or no. | starts with `is`, `are`, `does`, `do`, `can`, `has`, `have` | Is there a bus in the image? |
| `color` | Identify visible color or attribute. | `what color`, `colour`, `shape`, `material` | What color is the dog? |
| `object` | Identify an object, animal, vehicle, or item. | `what object`, `what animal`, `what vehicle`, fallback object question | What vehicle is in the image? |
| `action` | Describe what a person or animal is doing. | `doing`, `playing`, `riding`, `holding`, `eating`, `walking` | What is the dog doing? |
| `spatial_scene` | Describe position, relation, or scene context. | `where`, `behind`, `under`, `next to`, `on` | Where is the cat standing? |

## Ground-Truth Rules

- Use VQA v2 `multiple_choice_answer` as the default ground-truth answer.
- Keep answers short, usually 1 to 3 words.
- Use lowercase answers for consistency.
- Prefer simple labels such as `yes`, `no`, `bus`, `white`, `on snow`.
- Mark ambiguous or challenging rows in the `notes` column.
- Avoid questions that require private knowledge or identity recognition.
- Avoid OCR-heavy rows such as website, logo, license plate, or sign-reading questions because they distract from the main VQA goal.
