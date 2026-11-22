# NewsLens
### Intelligent Handwritten Answer Evaluation System

NewsLens is a teacher-facing application concept for digitising handwritten examination responses and supporting transparent, teacher-approved evaluation.

## Current demo milestone

- Teacher dashboard with an evaluation workflow overview.
- Image upload for handwritten answer samples (JPG, PNG, WEBP; 10 MB limit).
- Handwriting recognition using Microsoft's pretrained **TrOCR handwritten** checkpoint.
- Editable OCR output so a teacher can correct recognition mistakes before any downstream evaluation.

**Important scope note:** This first milestone implements handwriting text recognition only. Question segmentation, rubric matching, semantic answer evaluation, marks recommendation, and reporting are planned stages. The dashboard's sample activity and summary figures are illustrative demo data, not persisted student records or measured system performance.

## Technology

- Python 3.10+
- Django 5
- PyTorch
- Hugging Face Transformers
- Microsoft TrOCR: `microsoft/trocr-base-handwritten`

## Run locally

1. Clone the repository and enter the project folder.
2. Create and activate a virtual environment.
3. Install Python dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Apply migrations and start Django:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

5. Open http://127.0.0.1:8000/

The first OCR request downloads the pretrained model from Hugging Face and may take several minutes. A working internet connection is required for this initial download. PyTorch will use CUDA when a compatible CUDA-enabled installation is available; otherwise it runs on CPU.

## OCR input guidance

The current TrOCR checkpoint is line-oriented. For the most reliable first demo, upload a tightly cropped image containing one handwritten line or a small text region. Full-page layout analysis, automatic line detection, question-wise segmentation, and multi-column answer-sheet parsing are not implemented yet. Recognition quality depends on handwriting legibility, image quality, language, and layout. Always review and correct extracted text.

## Project structure

```
newslens/
  settings.py
  urls.py
evaluation/
  ocr.py
  views.py
  urls.py
templates/evaluation/
  dashboard.html
  upload.html
  review.html
static/css/
  app.css
manage.py
requirements.txt
```

## Responsible evaluation direction

The intended complete system will compare student responses against teacher-provided reference answers and marking rubrics. It should reward valid paraphrases and conceptually correct explanations, provide criterion-level reasons for suggested marks, and keep the teacher in control of final scores. Handwriting OCR errors must be reviewable before evaluation.

## Development status

| Capability | Status |
|---|---|
| Teacher dashboard | Demo UI implemented |
| Handwritten image upload | Implemented |
| TrOCR inference | Implemented, requires dependency/model setup |
| Editable extraction review | Implemented |
| Question segmentation | Planned |
| Rubric and semantic evaluation | Planned |
| Teacher approval and final marks | Planned |
| Persistent student/exam records | Planned |
