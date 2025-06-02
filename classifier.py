# classifier.py
import pandas as pd
from model_loader import get_pipeline
from data_loader import load_csv_safely, preprocess_text, load_pdf
import os


def validate_text(text: str, min_len: int = 5, max_len: int = 1000):
    """
    Validates a text input based on word count.

    Parameters:
    - text (str): The input text to validate.
    - min_len (int): Minimum number of words required.
    - max_len (int): Maximum number of words allowed.

    Returns:
    - (bool): Whether the text is valid
    - (int): Word count
    - (str): Validation message
    """
    if not text or not isinstance(text, str):
        return False, 0, "Text is empty or not a valid string."

    word_count = len(text.strip().split())

    if word_count < min_len:
        return False, word_count, f"Text too short (min {min_len} words)."
    elif word_count > max_len:
        return False, word_count, f"Text too long (max {max_len} words)."
    
    return True, word_count, "Valid"

def classify_text_input(text_input, task_name):
    classifier = get_pipeline(task_name)

    is_valid, word_count, status = validate_text(text_input)
    if not is_valid:
        return [{
            "source_file": "typed_input",
            "text": text_input,
            "label": "Invalid",
            "confidence": 0.0,
            "word_count": word_count,
            "message": status
        }]

    clean = preprocess_text(text_input)
    pred = classifier(clean)[0]

    return [{
        "source_file": "typed_input",
        "text": text_input,
        "label": pred["label"],
        "confidence": round(pred["score"] * 100, 2),
        "word_count": word_count,
        "message": "Success"
    }]

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df = load_csv_safely(file_path)
        return df["text"].dropna().tolist()

    elif ext == '.pdf':
        pdf_docs = load_pdf(file_path)
        return [doc.page_content for doc in pdf_docs if doc.page_content.strip() != ""]

    elif ext == '.txt':
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() != ""]

    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
def classify_text_and_files(text_input, file_paths, task_name):
    results = []

    # 1. Process typed input if present
    if text_input:
        results.extend(classify_text_input(text_input, task_name))

    # 2. Process files
    if file_paths:
        if len(file_paths) > 4:
            raise ValueError("Maximum 4 files allowed.")

        classifier = get_pipeline(task_name)

        for file_path in file_paths:
            try:
                texts = extract_text_from_file(file_path)
            except Exception as e:
                results.append({
                    "source_file": os.path.basename(file_path),
                    "text": None,
                    "label": "Error",
                    "confidence": 0.0,
                    "message": str(e)
                })
                continue

            for text in texts:
                is_valid, word_count, status = validate_text(text)
                if not is_valid:
                    results.append({
                        "source_file": os.path.basename(file_path),
                        "text": text,
                        "label": "Invalid",
                        "confidence": 0.0,
                        "word_count": word_count,
                        "message": status
                    })
                    continue

                clean = preprocess_text(text)
                pred = classifier(clean)[0]
                results.append({
                    "source_file": os.path.basename(file_path),
                    "text": text,
                    "label": pred["label"],
                    "confidence": round(pred["score"] * 100, 2),
                    "word_count": word_count,
                    "message": "Success"
                })

    return pd.DataFrame(results)