# 🔍 IMAGE INSIGHT: VQA

**Image Insight** is a production-grade, multimodal Visual Question Answering (VQA) system. It allows users to upload any image and ask natural-language questions about it, receiving highly accurate, real-time answers powered by state-of-the-art AI.

![VQA Concept](https://img.shields.io/badge/AI-Visual%20Question%20Answering-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 🔬 How It Works

Image Insight combines computer vision and natural language processing into a single pipeline:
1. **Image Upload**: A user uploads an image (JPG/PNG).
2. **Text Input**: The user asks a question (e.g., *"How many people are on the field?"* or *"What color is the sky?"*).
3. **Multimodal Inference**: The image is passed through a vision transformer, and the question is processed by a text encoder. The model fuses these modalities to predict the most probable answer.
4. **Classification & Output**: The system classifies the question into one of 6 semantic categories, measures the exact inference speed, and returns the normalized answer to an interactive UI.

### 🧠 Model Architecture

The application relies on Hugging Face Transformers and PyTorch, using a dual-model fallback mechanism to ensure reliability:
- **Primary Model**: `Salesforce/blip-vqa-base` (Requires ~1.5GB of RAM/VRAM for high-accuracy multimodal fusion).
- **Fallback Model**: `dandelin/vilt-b32-finetuned-vqa` (A lightweight ~470MB Vision-and-Language Transformer that auto-loads if the system lacks the memory for BLIP).

---

## 📊 Evaluation Metrics & Results

The system was rigorously evaluated against a curated 48-image test set sampled from **VQA v2 Validation** and **MS COCO 2014**. The dataset balances 6 core question types (8 examples each).

### Overall Performance
- **Global Accuracy:** `93.8%` (45 out of 48 correct)
- **Average Inference Time:** `0.122` seconds

### Per-Question-Type Accuracy
| Question Type | Accuracy | Score | Notes |
|---|---|---|---|
| **Action** | 100% | 8/8 | Flawless verb/activity recognition |
| **Object** | 100% | 8/8 | Perfect noun detection |
| **Yes / No** | 100% | 8/8 | Perfect binary logic reasoning |
| **Spatial** | 100% | 8/8 | Excellent contextual scene understanding |
| **Color** | 87.5% | 7/8 | Struggles slightly under ambiguous lighting |
| **Counting** | 75.0% | 6/8 | Tends to overcount clustered or occluded objects |

*(Note: The failure cases primarily involve severe visual clutter or partial occlusion, which are known limitations of current transformer architectures).*

---

## 👥 Team Contributions

This project was developed collaboratively in three distinct phases:

### Member 01: Dataset & Taxonomy
- **Data Curation:** Downloaded and filtered the official VQA v2 and MS COCO 2014 datasets.
- **Test Set Creation:** Authored the `vqa_test_set.csv` containing 48 balanced, manually-verified image-question-answer pairs.
- **Taxonomy Engine:** Engineered `src/question_types.py` to automatically classify user queries into 6 categories (counting, color, yes/no, object, action, spatial_scene).
- **Preprocessing:** Implemented robust image validation and preprocessing rules.

### Member 02: AI Integration & Evaluation
- **Pipeline Architecture:** Integrated Hugging Face Transformers and built the core inference engine (`src/inference.py`).
- **Memory Management:** Designed the graceful fallback architecture (`src/model_loader.py`) to prevent Out-Of-Memory crashes.
- **Answer Normalization:** Built robust text normalization (handling synonyms, casing, and punctuation) to align model outputs with ground truth.
- **Evaluation:** Conducted the batch evaluation script, achieving the final 93.8% accuracy and generating performance charts.

### Member 03: UI/UX & Production Deployment
- **Gradio Application:** Designed and developed the interactive `app.py` UI utilizing custom CSS and dynamic Gradio Blocks.
- **Design System:** Implemented a modern, professional "Navy & Accent" color palette with animated result cards and color-coded question-type badges.
- **Data Logging:** Engineered the `src/logger.py` module to persist all user interactions to a local CSV database.
- **Dashboards:** Built interactive tabs for viewing the test dataset, live prediction logs, and performance charts directly within the browser.

---

## 🚀 Setup & Usage Instructions

### Prerequisites
- Python 3.9+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hasana157/image-insight-VQA.git
   cd image-insight-VQA
   ```
2. Create and activate a virtual environment (Optional but recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the Gradio web interface by running:
```bash
python app.py
```
*Note: The first launch will automatically download the pre-trained AI models from Hugging Face. This may take a few minutes depending on your internet connection.*

Once the models are loaded, open the link provided in the terminal (usually `http://127.0.0.1:7860`) in your web browser. Upload an image, ask a question, and gain instant *Image Insight*!
