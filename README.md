# Simple Text Classification Web Application

This web application allows users to classify text data into one of three tasks: Sentiment Analysis, Spam Detection, or Topic Classification. It supports direct text input and batch file uploads (`.csv`, `.pdf`, `.txt`) for analysis.

---

## Features

- Choose between three NLP tasks:
  - Sentiment Analysis
  - Spam Detection
  - Topic Classification
- Input text manually or upload up to 4 files
- Displays prediction results in a table with:
  - Predicted label
  - Confidence score
  - Message status
- Supports `.csv`, `.pdf`, `.txt` file formats
- Includes REST API backend (FastAPI) and modern web UI
- Docker support (optional)

---

## Tech Stack

- Frontend: HTML/CSS or React (depending on version)
- Backend: Python (FastAPI)
- Models: Hugging Face Transformers
- PDF Parsing: `langchain_community`, `PyPDFLoader`
- CSV/Encoding Handling: `pandas`

---

## Setup Instructions

```bash
git clone https://github.com/thanphungdung/text-classification-app.git
cd text-classification-app
python -m venv venv
# Windows
venv\\Scripts\\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload