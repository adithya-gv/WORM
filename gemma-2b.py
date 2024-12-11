import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate
from earlyBird import EarlyGemma

import time

import torch.nn.utils.prune as prune

import train
import sys

from agents import earlyGemmaAgent, wormGemmaAgent


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token="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")
model = AutoModelForSequenceClassification.from_pretrained("google/gemma-2b", token="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", num_labels=2, cache_dir="/home/hice1/avasudev8/scratch/").cuda()

dataset = load_dataset("nyu-mll/glue", "sst2", cache_dir="/home/hice1/avasudev8/scratch/")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Get random 1/4 of the dataset
first_quarter_size = len(train_dataset) // 40

# Extract the first quarter of the dataset
train_dataset = train_dataset.select(range(first_quarter_size))


def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

prune_rate = 0.5

def earlyGemma():
    earlyBird = EarlyGemma(prune_rate, 3, 0.01)

    start_time = time.time()

    earlyGemmaAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, 20, earlyBird)

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.L1Unstructured.apply(module, name="weight", amount=prune_rate)

    accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

    epoch = 0

    while epoch < 1:
        # Train New Model until accuracy is 80%
        print(epoch)
        model = train.train_one_epoch_transformer(model, train_dataset, tokenizer, compute_metrics)

        # Test New Model
        accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

def wormGemma():

    earlyBird = EarlyGemma(prune_rate, 3, 0.01)

    start_time = time.time()

    wormGemmaAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, 20, earlyBird)

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.L1Unstructured.apply(module, name="weight", amount=prune_rate)

    accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

    epoch = 0

    while epoch < 1:
        # Train New Model until accuracy is 80%
        print(epoch)
        model = train.train_one_epoch_transformer(model, train_dataset, tokenizer, compute_metrics)

        # Test New Model
        accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

if __name__ == "__main__":
    if sys.argv[1] == "standard":
        earlyGemma()
    elif sys.argv[1] == "worm":
        wormGemma()
    else:
        print("Invalid Argument")
        exit(1)