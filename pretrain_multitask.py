from data.dc_dataset import DCDataset
from data.mlm_dataset import MLMDataset
from data.nsp_dataset import NSPDataset
from data.tc_dataset import TCDataset

import random

from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pandas as pd

import argparse
import torch
import numpy as np

from model.encoder_model_wrapper import ModelForPretraining
from torch import nn
from tqdm import tqdm

'''
------------------------------------------------------------------------------------------------------------------------

                                            PARSING ARGUMENTS

------------------------------------------------------------------------------------------------------------------------
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--dialettando_data_folder', type=str, default='/home/mlaquatra/geolingit/geolingit-data-collection/dialettando/dialettando_data/')
    parser.add_argument('--wikipedia_data_folder', type=str, default='/home/mlaquatra/geolingit/geolingit-data-collection/wikipedia_extracted/')
    parser.add_argument('--output_dir', type=str, default='pretrained_models/')
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--nsp', action='store_true')
    parser.add_argument('--dc', action='store_true')
    parser.add_argument('--tc', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--warmup_percentage', type=float, default=0.1)
    parser.add_argument('--task_best_model_checkpoint', type=str, default='dc')
    parser.add_argument('--logging_steps', type=int, default=100)

    return parser.parse_args()

args = parse_args()


'''
------------------------------------------------------------------------------------------------------------------------

                                                PREPARING DATA

------------------------------------------------------------------------------------------------------------------------
'''

# read all data for all regions
regions = [
    "abruzzo", "basilicata", "calabria", "campania", "emilia_romagna", "friuli_venezia_giulia", "lazio",
    "liguria", "lombardia", "marche", "molise", "piemonte", "puglia", "sardegna", "sicilia", "toscana",
    "trentino_alto_adige", "umbria", "valle_daosta", "veneto",
]
regions.append("italian") # add italian as a region for DC task
regions.append("corsica") # add corsica as a region for DC task

wikipedia_mapping = {
    "co": "corsica",
    "eml": "emilia_romagna",
    "fur": "friuli_venezia_giulia",
    "lij": "liguria",
    "lmo": "lombardia",
    "nap": "campania",
    "pms": "piemonte",
    "scn": "sicilia",
    "roa_tara": "puglia",
    "sc": "sardegna",
    "vec": "veneto",
}

text_types = ["racconti", "poesie", "proverbi"]
# map regions to integers
region_to_int = {}
for i, region in enumerate(regions):
    region_to_int[region] = i

texts = []
labels = []
label_italian = region_to_int["italian"]
for region in regions:
    label_region = region_to_int[region]
    for text_type in text_types:
        try:
            df = pd.read_csv(args.dialettando_data_folder + region + "/" + text_type + ".csv")
            df.dropna(inplace=True)
            text_dialect = df["dialect"].tolist()
            text_italian = df["italian"].tolist()
            texts.extend(text_dialect)
            texts.extend(text_italian)
            labels.extend([label_region] * len(text_dialect))
            labels.extend([label_italian] * len(text_italian))
        except:
            print(f"No data for {region} and {text_type}")

for wiki_lang in wikipedia_mapping.keys():
    try:
        df = pd.read_csv(args.wikipedia_data_folder + wiki_lang + ".csv")
        df.dropna(inplace=True)
        text = df["dialect"].tolist()
        texts.extend(text)
        labels.extend([region_to_int[wikipedia_mapping[wiki_lang]]] * len(text))
    except:
        print(f"No data for {wiki_lang}")

# shuffle texts and labels
c = list(zip(texts, labels))
random.shuffle(c)
texts, labels = zip(*c)
# convert to list
texts = list(texts)
labels = list(labels)

# split into train and validation
train_texts = texts[:int(0.9 * len(texts))]
train_labels = labels[:int(0.9 * len(texts))]
val_texts = texts[int(0.9 * len(texts)):]
val_labels = labels[int(0.9 * len(texts)):]

print ("[DIALECT] Total number of training examples: ", len(train_texts))
print ("[DIALECT] Total number of validation examples: ", len(val_texts))

# print the distributions of the labels in the training and validation sets - one per region
print ("[DIALECT] Training set distribution:")
for region in regions:
    print (f"{region}: {train_labels.count(region_to_int[region])}")
print ("[DIALECT] Validation set distribution:")
for region in regions:
    print (f"{region}: {val_labels.count(region_to_int[region])}")

import matplotlib.pyplot as plt

# plot the distribution of the labels in the training and validation sets in a pie chart
def plot_pie_chart(labels, title, filename):
    # reset the plot
    plt.clf()
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    # use region name instead of integer
    unique = [regions[i] for i in unique]
    plt.pie(counts, labels=unique, autopct='%1.1f%%')
    plt.title(title)
    plt.savefig(filename)

plot_pie_chart(train_labels, "Training set distribution", "stats/train_labels_distribution.png")
plot_pie_chart(val_labels, "Validation set distribution", "stats/val_labels_distribution.png")

# create datasets
train_dataloaders = []
val_dataloaders = []

if args.mlm:
    train_mlm_dataset = MLMDataset(
        texts = train_texts,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )
    train_mlm_dataloader = torch.utils.data.DataLoader(
        train_mlm_dataset,
        batch_size = args.per_device_train_batch_size,
        shuffle = True,
        num_workers=4,
    )

    val_mlm_dataset = MLMDataset(
        texts = val_texts,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )
    val_mlm_dataloader = torch.utils.data.DataLoader(
        val_mlm_dataset,
        batch_size = args.per_device_eval_batch_size,
        shuffle = False,
        num_workers=4,
    )

    train_dataloaders.append(train_mlm_dataloader)
    val_dataloaders.append(val_mlm_dataloader)

if args.nsp:
    train_nsp_dataset = NSPDataset(
        texts = train_texts,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )
    train_nsp_dataloader = torch.utils.data.DataLoader(
        train_nsp_dataset,
        batch_size = args.per_device_train_batch_size,
        shuffle = True,
        num_workers=4,
    )

    val_nsp_dataset = NSPDataset(
        texts = val_texts,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )
    val_nsp_dataloader = torch.utils.data.DataLoader(
        val_nsp_dataset,
        batch_size = args.per_device_eval_batch_size,
        shuffle = False,
        num_workers=4,
    )

    train_dataloaders.append(train_nsp_dataloader)
    val_dataloaders.append(val_nsp_dataloader)

if args.dc:
    train_dc_dataset = DCDataset(
        texts = train_texts,
        labels = train_labels,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
        italian_label_id = region_to_int["italian"],
    )
    train_dc_dataloader = torch.utils.data.DataLoader(
        train_dc_dataset,
        batch_size = args.per_device_train_batch_size,
        shuffle = True,
        num_workers=4,
    )

    val_dc_dataset = DCDataset(
        texts = val_texts,
        labels = val_labels,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
        italian_label_id = region_to_int["italian"],
    )
    val_dc_dataloader = torch.utils.data.DataLoader(
        val_dc_dataset,
        batch_size = args.per_device_eval_batch_size,
        shuffle = False,
        num_workers=4,
    )

    train_dataloaders.append(train_dc_dataloader)
    val_dataloaders.append(val_dc_dataloader)

if args.tc:
    train_tc_dataset = TCDataset(
        texts = train_texts,
        labels = train_labels,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )
    train_tc_dataloader = torch.utils.data.DataLoader(
        train_tc_dataset,
        batch_size = args.per_device_train_batch_size,
        shuffle = True,
        num_workers=0,
    )

    val_tc_dataset = TCDataset(
        texts = val_texts,
        labels = val_labels,
        tokenizer_name_or_path = args.model_name_or_path,
        max_length=args.max_seq_length,
    )

    val_tc_dataloader = torch.utils.data.DataLoader(
        val_tc_dataset,
        batch_size = args.per_device_eval_batch_size,
        shuffle = False,
        num_workers=0,
    )

    train_dataloaders.append(train_tc_dataloader)
    val_dataloaders.append(val_tc_dataloader)




class ListDataloadersWrapper:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.num_dataloaders = len(dataloaders)
        self.counters = [0] * self.num_dataloaders
        self.num_samples = [len(dataloader) for dataloader in dataloaders]
        self.num_samples_total = sum(self.num_samples)
        self.iterators = [iter(dataloader) for dataloader in dataloaders]

        print(f"num_dataloaders: {self.num_dataloaders}")
        print(f"num_samples_total: {self.num_samples_total}")
        print(f"num_samples: {self.num_samples}")

    def __iter__(self):
        return self

    def __next__(self):
        # randomly choose a dataloader among the ones that have not finished
        try:
            dataloader_index = random.choice([i for i in range(self.num_dataloaders) if self.counters[i] < self.num_samples[i]])
        except IndexError:
            print ("The dataloaders are empty, restarting them")
            self.counters = [0] * self.num_dataloaders
            self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
            raise StopIteration
        self.counters[dataloader_index] += 1
        return next(self.iterators[dataloader_index])

    def __len__(self):
        return self.num_samples_total

train_dataloader = ListDataloadersWrapper(train_dataloaders)
val_dataloader = ListDataloadersWrapper(val_dataloaders)

print ("train_dataloader length: ", len(train_dataloader))
print ("val_dataloader length: ", len(val_dataloader))

current_step = 0

global_steps = int(args.num_train_epochs * len(train_dataloader))
print(f"global_steps: {global_steps}")

model = ModelForPretraining(
    model_name_or_path = args.model_name_or_path,
    mlm = args.mlm,
    nsp = args.nsp,
    dc = args.dc,
    num_labels_dc = len(regions),
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
lr_lambda = lambda step: max(1e-9, min(1.0, step / (global_steps * args.warmup_percentage)) * (1.0 - (step - global_steps * args.warmup_percentage) / (global_steps * (1 - args.warmup_percentage))) ** 0.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
remaining_epochs = args.num_train_epochs

device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Using %d GPUs" % num_gpus)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

model.to(device)

current_step = 0
# store losses separately for each task
losses = {task: [] for task in [0, 1, 2, 3]} # 0: mlm, 1: nsp, 2: dc
# use tqdm to show progress bar

# val loss to decide when to save the model
best_val_loss = float("inf")

for epoch in range(remaining_epochs):

    print(f"Epoch {epoch + 1}/{remaining_epochs}")

    try:
        model.module.train()
    except AttributeError:
        model.train()

    p_bar = tqdm(train_dataloader, desc="Training", disable=not(args.verbose))

    for batch in p_bar:
        task_index = int(batch["task_index"][0].item())
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**dict(batch = batch, current_step = current_step))
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        current_step += 1
        losses[task_index].append(loss.item())
        #postfix in tqdm
        if args.verbose:
            s_postfix = ""
            if args.mlm:
                s_postfix += f"MLM: {np.mean(losses[0])}  "
            if args.nsp:
                s_postfix += f"NSP: {np.mean(losses[1])}  "
            if args.dc:
                s_postfix += f"DC: {np.mean(losses[2])}  "
            if args.tc:
                s_postfix += f"TC: {np.mean(losses[3])}  "
            p_bar.set_postfix_str(s_postfix)

        if current_step % args.logging_steps == 0:
            s_print = ""
            if args.mlm:
                s_print += f"MLM: {np.mean(losses[0])}  \t"
            if args.nsp:
                s_print += f"NSP: {np.mean(losses[1])}  \t"
            if args.dc:
                s_print += f"DC: {np.mean(losses[2])}  \t"
            if args.tc:
                s_print += f"TC: {np.mean(losses[3])}  \t"

            print(s_print)
            
            losses = {task: [] for task in [0, 1, 2, 3]} # 0: mlm, 1: nsp, 2: dc, 3: tc

    # validation
    try:
        model.module.eval()
    except AttributeError:
        model.eval()
        
    p_bar = tqdm(val_dataloader, desc="Validation", disable=not(args.verbose))
    with torch.no_grad():
        for batch in p_bar:
            task_index = int(batch["task_index"][0].item())
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**dict(batch = batch, current_step = current_step))
            loss = loss.mean()
            losses[task_index].append(loss.item())
            #postfix in tqdm
            if args.verbose:
                s_postfix = ""
                if args.mlm:
                    s_postfix += f"MLM: {np.mean(losses[0])}  "
                if args.nsp:
                    s_postfix += f"NSP: {np.mean(losses[1])}  "
                if args.dc:
                    s_postfix += f"DC: {np.mean(losses[2])}  "
                if args.tc:
                    s_postfix += f"TC: {np.mean(losses[3])}  "
                p_bar.set_postfix_str(s_postfix)

        # print losses separately for each task
        if args.mlm:
            print(f"[VALIDATION] MLM: {np.mean(losses[0])}")
        if args.nsp:
            print(f"[VALIDATION] NSP: {np.mean(losses[1])}")
        if args.dc:
            print(f"[VALIDATION] DC: {np.mean(losses[2])}")
        if args.tc:
            print(f"[VALIDATION] TC: {np.mean(losses[3])}")
        
        val_mlm_loss = np.mean(losses[0])
        val_nsp_loss = np.mean(losses[1])
        val_dc_loss = np.mean(losses[2])
        val_tc_loss = np.mean(losses[3])

    if args.task_best_model_checkpoint == "mlm":
        val_loss = val_mlm_loss
    elif args.task_best_model_checkpoint == "nsp":
        val_loss = val_nsp_loss
    elif args.task_best_model_checkpoint == "dc":
        val_loss = val_dc_loss
    elif args.task_best_model_checkpoint == "tc":
        val_loss = val_tc_loss
    else:
        val_loss = (val_mlm_loss + val_nsp_loss + val_dc_loss + val_tc_loss) / 4
    

    # save model
    sanitized_model_name = args.model_name_or_path.replace("/", "-")
    print (f"Saving model to {args.output_dir + sanitized_model_name + '/model_' + str(epoch) + '/'}")
    try:
        model.module.save_model(args.output_dir + sanitized_model_name + "/model_" + str(epoch) + "/")
    except AttributeError:
        model.save_model(args.output_dir + sanitized_model_name + "/model_" + str(epoch) + "/")
    # save tokenizer
    tokenizer.save_pretrained(args.output_dir + sanitized_model_name + "/model_" + str(epoch) + "/")

    if val_loss < best_val_loss:
        # save model
        print(f"Saving best model so far - val loss: {val_loss}")
        try:
            model.module.save_model(args.output_dir + sanitized_model_name + "/best_model/")
        except AttributeError:
            model.save_model(args.output_dir + sanitized_model_name + "/best_model/")
            # save tokenizer
        tokenizer.save_pretrained(args.output_dir + sanitized_model_name + "/best_model/")
        best_val_loss = val_loss

print ("Training complete!")


