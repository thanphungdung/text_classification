# model_loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Define available models
MODEL_REGISTRY = {
    "Sentiment Analysis": "cardiffnlp/twitter-roberta-base-sentiment",
    "Spam Detection": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
    "Topic Classification": "textattack/distilbert-base-uncased-ag-news"
}

# Cache loaded models
loaded_pipeline = {}

def get_pipeline(task_name):
    if task_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{task_name}' is not available. Available models: {list(MODEL_REGISTRY.keys())}")
    if task_name in loaded_pipeline:
        return loaded_pipeline[task_name]
    model_id = MODEL_REGISTRY[task_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)
    loaded_pipeline[task_name] = pipe
    return pipe
    
def get_available_models():
    return list(MODEL_REGISTRY.keys())
