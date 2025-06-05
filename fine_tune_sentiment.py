from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import evaluate
# Load IMDB (50k reviews: 25k train, 25k test)
dataset = load_dataset("imdb")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english", num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_dataset["train"] = tokenized_dataset["train"].select(range(100))  
tokenized_dataset["test"] = tokenized_dataset["test"].select(range(10))  
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# DF: Batch size 8, epochs 3
training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model("./sentiment-finetuned-imdb")
tokenizer.save_pretrained("./sentiment-finetuned-imdb")
