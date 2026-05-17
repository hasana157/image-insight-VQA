# Member 03 → Team Handoff

## Status: DONE

## Completed Work

### Gradio Application (`app.py`)
- Built a production-grade Gradio Blocks interface with custom CSS
- Implemented the full design guide specifications:
  - Professional Navy (#0f3460) + accent color palette
  - Custom typography with Segoe UI font stack
  - Animated result cards with fade-in effects
  - Color-coded question type badges for all 6 types
  - Stats dashboard showing accuracy, test cases, avg inference time, and model name
  - Styled error messages for invalid inputs (no image, empty question, inference failure)
- **Header section** with gradient title and subtitle
- **Input section** with image upload (drag & drop + clipboard), question textbox, submit + clear buttons
- **Results section** with styled answer display, question type badge, and inference time
- **Predefined examples** loaded from `data/vqa_test_set.csv` (one per question type)
- **Tabbed content**:
  - 📋 Dataset — Full 48-row test set table
  - 📝 Predictions Log — Auto-refreshing log of all submissions
  - 📊 Performance — Charts from `results/` (question type accuracy + inference time)
  - ℹ️ About — Model info, limitations, team credits
- Enter key submits the question
- Model loads once at startup (not on every inference)

### Logger Implementation (`src/logger.py`)
- Replaced `NotImplementedError` with working CSV append logic
- Auto-increments prediction IDs
- Logs: image_path, question, predicted_answer, question_type, inference_time_sec, timestamp
- Appends to `data/predictions.csv`

### Documentation
- Created `handoff/member03_to_team.md` (this file)
- Created `demo/demo_script.txt` with 2-3 minute demo flow
- Updated `handoff.txt` Member 03 section
- Updated `TASK_CHECKLIST.md` with completed items

## How to Run

### Install dependencies
```bash
pip install torch torchvision transformers accelerate pillow opencv-python gradio pandas matplotlib
```

### Launch the app
```bash
python app.py
```
The app will:
1. Load the BLIP-VQA model (first run downloads ~1GB)
2. Start a local Gradio server at `http://0.0.0.0:7860`
3. Open the UI in your default browser

### Run dataset validation
```bash
python scripts/validate_dataset.py
```

## Files Added / Modified

| File | Action | Purpose |
|---|---|---|
| `app.py` | Modified | Full Gradio application with custom CSS |
| `src/logger.py` | Modified | Working prediction logging (was NotImplementedError) |
| `handoff/member03_to_team.md` | New | This handoff document |
| `demo/demo_script.txt` | New | Live demo script (2-3 minutes) |

## Dependencies
All dependencies are listed in `requirements.txt`:
- `torch`, `transformers`, `accelerate` — Model loading and inference
- `pillow`, `opencv-python` — Image processing
- `gradio` — Web UI framework
- `pandas` — Data handling
- `matplotlib` — Chart rendering

## Known Issues
- Model download requires internet (~1GB on first run)
- Large images (>10MB) may slow inference slightly
- Counting accuracy is 75% — model tends to overcount clustered objects
- Color accuracy is 87.5% — subtle distinctions under varied lighting are challenging

## Tips for Team
- If Gradio imports fail, ensure `gradio>=4.0` is installed
- If model loading fails, it auto-falls back to `dandelin/vilt-b32-finetuned-vqa`
- The app logs every submission to `data/predictions.csv` — check this for demo history
- Screenshots for the report should be saved to `report/screenshots/`
