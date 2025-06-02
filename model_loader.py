# model_loader.py
from transformers import pipeline

# Define available models
MODEL_OPTIONS = {
    "DistilBERT (Sentiment)": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa (Sentiment)": "cardiffnlp/twitter-roberta-base-sentiment",
    # Add more models if needed
}

# Cache loaded models
loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        model_id = MODEL_OPTIONS[model_name]
        loaded_models[model_name] = pipeline("sentiment-analysis", model=model_id)
    return loaded_models[model_name]

for model_name in MODEL_OPTIONS:
    get_model(model_name)  
    print(f"Model '{model_name}' loaded successfully.")