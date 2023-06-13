import argparse
import csv
import random

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from data.classification_dataset import ClassificationDataset
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

"""
------------------------------------------------------------------------------------------------------------------------

                                            PARSING ARGUMENTS

------------------------------------------------------------------------------------------------------------------------
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--model_name_or_path", type=str, default="bert-base-multilingual-cased"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_a/train_a.tsv",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_a/dev_a.tsv",
    )
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="ft_models_a/")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--warmup_percentage", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_file", type=str, default=None)

    return parser.parse_args()


args = parse_args()


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_all_seeds(args.seed)

"""
------------------------------------------------------------------------------------------------------------------------

                                            LOADING DATA

------------------------------------------------------------------------------------------------------------------------
"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# Load train data
train_df = pd.read_csv(args.train_file, sep="\t", quoting=csv.QUOTE_NONE)

train_texts = train_df["text"].tolist()
train_labels = train_df["region"].tolist()

# Load dev data
dev_df = pd.read_csv(args.dev_file, sep="\t", quoting=csv.QUOTE_NONE)
dev_texts = dev_df["text"].tolist()
dev_labels = dev_df["region"].tolist()

# map labels to integers
label2id = {label: i for i, label in enumerate(sorted(list(set(train_labels))))}
print("Label2id: {}".format(label2id))

train_labels = [label2id[label] for label in train_labels]
dev_labels = [label2id[label] for label in dev_labels]

# Create train dataset
train_dataset = ClassificationDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=args.max_seq_length,
)

# Create dev dataset
dev_dataset = ClassificationDataset(
    texts=dev_texts,
    labels=dev_labels,
    tokenizer=tokenizer,
    max_length=args.max_seq_length,
)


"""
------------------------------------------------------------------------------------------------------------------------

                                            PREPARING TRAINING

------------------------------------------------------------------------------------------------------------------------
"""

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path, num_labels=len(label2id),
    label2id=label2id,
    id2label={id: label for label, id in label2id.items()}
)
sanitized_model_name = args.model_name_or_path.replace("/", "-")


def compute_metrics(pred):
    labels = pred.label_ids
    print(pred)
    try:
        preds = pred.predictions.argmax(-1)
    except:
        preds = pred.predictions[0].argmax(-1)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        labels, preds, average="macro", labels=list(set(labels))
    )
    print(sklearn.metrics.classification_report(labels, preds, digits=4))
    acc = sklearn.metrics.accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


warmup_steps = int(
    args.warmup_percentage
    * (
        (args.num_train_epochs * len(train_dataset))
        / (args.per_device_train_batch_size * torch.cuda.device_count())
    )
)
print("Warmup steps: {}".format(warmup_steps))


# Create training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir + sanitized_model_name + "/",
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_steps=args.logging_steps,
    save_strategy="epoch",
    save_total_limit=3,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

"""
------------------------------------------------------------------------------------------------------------------------

                                            TRAINING    

------------------------------------------------------------------------------------------------------------------------
"""

# Train model
trainer.train()

# Save model
trainer.save_model(args.output_dir + sanitized_model_name + "/best_model/")

# Evaluate model
print(
    "\n ----------------- EVALUATION BEST MODEL ON VALIDATION SET ----------------- \n"
)
print(trainer.evaluate())

"""
model = bert-base-multilingual-cased
{
    'eval_loss': 1.0652551651000977, 
    'eval_accuracy': 0.7626811594202898, 
    'eval_f1': 0.756604662884207, 
    'eval_precision': 0.7701607177732873, 
    'eval_recall': 0.7626811594202898, 
    'eval_runtime': 1.1492, 
    'eval_samples_per_second': 480.33, 
    'eval_steps_per_second': 7.831, 
    'epoch': 10.0
}

model = dbmdz/bert-base-italian-cased
"""