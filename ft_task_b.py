import argparse
import transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from data.regression_dataset import RegressionDataset

from model.dual_regression_model import DualRegressionModel
from torch.utils.data import DataLoader

from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
import torch
import os
import csv
import random
from haversine import haversine, Unit

import geopy.distance

'''
------------------------------------------------------------------------------------------------------------------------

                                            PARSING ARGUMENTS

------------------------------------------------------------------------------------------------------------------------
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--train_file', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/train_b.tsv')
    parser.add_argument('--dev_file', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='ft_models_b/')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--warmup_percentage', type=float, default=0.1)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

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


'''
------------------------------------------------------------------------------------------------------------------------

                                            LOADING DATA

------------------------------------------------------------------------------------------------------------------------
'''

if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# Load train data
train_df = pd.read_csv(args.train_file, sep='\t', quoting=csv.QUOTE_NONE)
train_texts = train_df['text'].tolist()
train_latitudes = train_df['latitude'].tolist()
train_longitudes = train_df['longitude'].tolist()

# Load dev data
dev_df = pd.read_csv(args.dev_file, sep='\t', quoting=csv.QUOTE_NONE)
dev_texts = dev_df['text'].tolist()
dev_latitudes = dev_df['latitude'].tolist()
dev_longitudes = dev_df['longitude'].tolist()

# Create train dataset
train_dataset = RegressionDataset(
    texts=train_texts,
    latitudes=train_latitudes,
    longitudes=train_longitudes,
    tokenizer=tokenizer,
    max_length=args.max_seq_length,
)

# Create dev dataset
dev_dataset = RegressionDataset(
    texts=dev_texts,
    latitudes=dev_latitudes,
    longitudes=dev_longitudes,
    tokenizer=tokenizer,
    max_length=args.max_seq_length,
)

'''
------------------------------------------------------------------------------------------------------------------------

                                            PREPARING TRAINING

------------------------------------------------------------------------------------------------------------------------
'''

# Load model
model = DualRegressionModel(args.model_name_or_path)
sanitized_model_name = args.model_name_or_path.replace('/', '-')

train_dataloaders = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
dev_dataloaders = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True)

# Create optimizer and scheduler
warmup_steps = int(args.warmup_percentage * ((args.num_train_epochs * len(train_dataset)) / (args.per_device_train_batch_size)))
print("Warmup steps: {}".format(warmup_steps))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_train_epochs * len(train_dataset))
# set best eval mse to infinity
best_eval_mse = float('inf')
best_eval_distance = float('inf')
best_eval_epoch = 0
if device.type == 'cuda':
    model = model.to(device)


def compute_distance(latitudes, longitudes, batch):
    # compute distance between predicted and true latitudes and longitudes
    distances = []
    for i, (latitude, longitude) in enumerate(zip(latitudes, longitudes)):
        # distance += geopy.distance.distance((latitude, longitude), (batch['latitude'][i], batch['longitude'][i])).km
        distances.append(haversine((latitude.item(), longitude.item()), (batch['latitude'][i].item(), batch['longitude'][i].item()), unit=Unit.KILOMETERS))
    return distances

for epoch in tqdm(range(args.num_train_epochs), desc="Epoch"):
    # Train
    model.train()
    p_bar = tqdm(train_dataloaders, desc=f"Training epoch {epoch}")
    for step, batch in enumerate(p_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # model forward take as input the batch and the current step in training
        out = model(batch)
        loss = out['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        # set post fix for progress bar - loss and learning rate - distance between predicted and true latitudes and longitudes
        distances = compute_distance(out['latitude'], out['longitude'], batch)
        distance = sum(distances) / len(batch)
        p_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0], 'distance': distance})

    # Evaluate
    model.eval()
    eval_loss = 0
    eval_mse = 0
    eval_distances = []
    nb_eval_steps = 0
    for batch in dev_dataloaders:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            latitudes = outputs['latitude']
            longitudes = outputs['longitude']
            loss = outputs['loss']

        eval_loss += loss.mean().item()
        nb_eval_steps += 1
        # mse between predicted and true latitudes and longitudes
        eval_mse += torch.sqrt(torch.mean((latitudes - batch['latitude']) ** 2) + torch.mean((longitudes - batch['longitude']) ** 2))
        # distance between predicted and true latitudes and longitudes
        eval_distances.extend(compute_distance(latitudes, longitudes, batch))

    eval_loss = eval_loss / (nb_eval_steps)
    eval_mse = eval_mse / (nb_eval_steps)
    eval_distance = sum(eval_distances) / len (eval_distances)

    print(f"Epoch {epoch} - eval_loss: {eval_loss} - eval_mse: {eval_mse} - eval_distance: {eval_distance}")

    # create output directory if it doesn't exist - and subdirectories
    # epoch directory
    if not os.path.exists(args.output_dir + sanitized_model_name + f"/epoch_{epoch}"):
        os.makedirs(args.output_dir + sanitized_model_name + f"/epoch_{epoch}")
    # best model directory
    if not os.path.exists(args.output_dir + sanitized_model_name + "/best_model"):
        os.makedirs(args.output_dir + sanitized_model_name + "/best_model")


    # save model
    model.save_model(args.output_dir + sanitized_model_name + f"/epoch_{epoch}/regression_model.pt")
    tokenizer.save_pretrained(args.output_dir + sanitized_model_name + f"/epoch_{epoch}/")

    if best_eval_distance > eval_distance:
        best_eval_distance = eval_distance
        best_eval_epoch = epoch
        model.save_model(args.output_dir + sanitized_model_name + "/best_model/regression_model.pt")
        tokenizer.save_pretrained(args.output_dir + sanitized_model_name + "/best_model/")

print(f"Best eval distance: {best_eval_distance} - Best eval epoch: {best_eval_epoch}")

