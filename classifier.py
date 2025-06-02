# classifier.py
import pandas as pd
from model_loader import get_model

def classify_text(text, model_name):
    model = get_model(model_name)
    result = model(text)[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"] * 100, 2)
    }

def classify_csv(file_path, model_name):
    model = get_model(model_name)
    df = pd.read_csv(file_path)
    results = []

    for text in df["text"]:
        res = model(text)[0]
        results.append({
            "text": text,
            "label": res["label"],
            "confidence": round(res["score"] * 100, 2)
        })

    return pd.DataFrame(results)
print(classify_text("It is okay everything normal.", "DistilBERT (Sentiment)"))