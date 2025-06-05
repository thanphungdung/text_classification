import pandas as pd
from model_loader import get_pipeline
from data_loader import load_csv_safely, preprocess_text
import os

# Label mappings
LABEL_MAPS = {
    "Topic Classification": {
        "LABEL_0": "World",
        "LABEL_1": "Sports",
        "LABEL_2": "Business",
        "LABEL_3": "Sci/Tech",
    },
    "Spam Detection": {
        "LABEL_0": "Not Spam",
        "LABEL_1": "Spam",
    },
    "Sentiment Analysis": {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive",
    }
}

# Custom logic to boost confidence for known strong sentiment words
def post_process_sentiment(text, label, score):
    positive_keywords = ["love", "awesome", "excellent", "amazing", "fantastic", "best", "great"]
    negative_keywords = ["hate", "terrible", "worst", "awful", "horrible", "bad"]

    lowered = text.lower()
    if score < 0.8:  # only override if model isn't very confident
        if any(word in lowered for word in positive_keywords):
            return "Positive"
        elif any(word in lowered for word in negative_keywords):
            return "Negative"
    return label

def validate_text(text: str, min_len: int = 1, max_len: int = 1000):
    if not text or not isinstance(text, str):
        return False, 0, "Text is empty or not a valid string."

    word_count = len(text.strip().split())
    if word_count < min_len:
        return False, word_count, f"Text too short (min {min_len} words)."
    elif word_count > max_len:
        return False, word_count, f"Text too long (max {max_len} words)."
    return True, word_count, "Valid"

def classify_text_input(text_input, task_name, return_raw=False):
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
    all_scores = classifier(clean)[0]
    top_pred = max(all_scores, key=lambda x: x["score"])

    label_map = LABEL_MAPS.get(task_name, {})
    label = label_map.get(top_pred["label"], top_pred["label"])
    score = top_pred["score"]

    # Apply sentiment post-processing logic
    if task_name == "Sentiment Analysis":
        label = post_process_sentiment(text_input, label, score)

    result = {
        "source_file": "typed_input",
        "text": text_input,
        "label": label,
        "confidence": round(score * 100, 2),
        "word_count": word_count,
        "message": "Success"
    }

    if return_raw:
        result["raw_scores"] = all_scores

    return [result]

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = load_csv_safely(file_path)
        return df["text"].dropna().tolist()
    elif ext == '.txt':
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def classify_text_and_files(text_input, file_paths, task_name):
    results = []

    # 1. Typed input
    if text_input:
        results.extend(classify_text_input(text_input, task_name))

    # 2. Files
    if file_paths:
        if len(file_paths) > 4:
            raise ValueError("Maximum 4 files allowed.")

        classifier = get_pipeline(task_name)

        for file_path, file_name in file_paths:
            try:
                texts = extract_text_from_file(file_path)
            except Exception as e:
                results.append({
                    "source_file": file_name,
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
                        "source_file": file_name,
                        "text": text,
                        "label": "Invalid",
                        "confidence": 0.0,
                        "word_count": word_count,
                        "message": status
                    })
                    continue

                clean = preprocess_text(text)
                all_scores = classifier(clean)[0]
                top_pred = max(all_scores, key=lambda x: x["score"])
                label_map = LABEL_MAPS.get(task_name, {})
                label = label_map.get(top_pred["label"], top_pred["label"])
                score = top_pred["score"]

                # Apply sentiment logic
                if task_name == "Sentiment Analysis":
                    label = post_process_sentiment(text, label, score)

                results.append({
                    "source_file": file_name,
                    "text": text,
                    "label": label,
                    "confidence": round(score * 100, 2),
                    "word_count": word_count,
                    "message": "Success"
                })

    return pd.DataFrame(results)
