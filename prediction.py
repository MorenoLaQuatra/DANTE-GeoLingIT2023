import argparse
import csv
import pathlib
from functools import partial

import numpy as np
import pandas as pd
import torch
from model.dual_regression_model import DualRegressionModel
from more_itertools import ichunked
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import pipeline


def main():
    args = _parse_args()
    # Load test data
    test_df = pd.read_csv(args.test_file, sep="\t", quoting=csv.QUOTE_NONE)

    if args.task == "a":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    elif args.task == "b":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = DualRegressionModel(args.model_name_or_path).eval().to(device)
        model.load_model(args.regression_model_weights)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        pipe = partial(_regression_model, tokenizer, model, device)

    test_texts = test_df["text"].tolist()
    predictions = pipe(test_texts)

    if args.task == "a":
        test_df["region"] = [p["label"] for p in predictions]
    elif args.task == "b":
        test_df["latitude"] = predictions["latitude"]
        test_df["longitude"] = predictions["longitude"]
    # Save predictions to csv
    output_file = (
        args.output_file
        or f"predictions_{pathlib.Path(args.model_name_or_path).stem}.tsv"
    )
    test_df.to_csv(output_file, sep="\t", index=False)


def _regression_model(tokenizer, model, device, x):
    preds = {"latitude": [], "longitude": []}
    chunck_size = 1 if device == "cpu" else 16
    for sample in ichunked(x, chunck_size):
        sample = list(sample)
        enc = tokenizer(
            sample,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        out = model(enc)
        preds["latitude"].append(out["latitude"].flatten().detach().cpu().numpy())
        preds["longitude"].append(out["longitude"].flatten().detach().cpu().numpy())
    preds["latitude"] = np.concatenate(preds["latitude"])
    preds["longitude"] = np.concatenate(preds["longitude"])
    return preds


def _parse_args():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test file")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="a",
        help="Task to train model for, either 'a' or 'b'",
    )
    parser.add_argument(
        "--regression_model_weights",
        type=str,
        default=None,
        help="Path to regression model weights",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
