import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate
from earlyBird import EarlyBERT

from agents import earlyBERTAgent


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2).cuda()

dataset = load_dataset("glue", "cola")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

earlyBird = EarlyBERT(0.5, 5, 0.1)

earlyBERTAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, 20, earlyBird)