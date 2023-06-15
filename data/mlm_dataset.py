import random
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer


class MLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer_name_or_path: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        task_index: int = 0,
        **kwargs,
    ):
        """
        This class instantiates the pre-training dataset for the masked language modeling task.
        :param texts: List of texts to be tokenized.
        :param tokenizer_name_or_path: The tokenizer name or path to be used for tokenizing the texts.
        :param max_length: The maximum length of the tokenized text.
        :param padding: The padding strategy to be used. Available options are available in the transformers library.
        :param truncation: Whether to truncate the text or not.
        :param task_index: The index of the task in the pre-training pipeline.
        """

        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.task_index = task_index

    def __len__(self):
        """
        This function is called to get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ):
        """
        This function is called to get the tokenized text and the label for a given index.
        :param index: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized text (with attention mask) and the labels.
        """
        enc = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding=False,
            truncation=self.truncation,
            return_tensors="pt",
        )

        labels = enc["input_ids"][0].clone()
        input_ids = enc["input_ids"][0].clone()
        attention_mask = enc["attention_mask"][0].clone()
        # randomly mask 15% of the tokens
        # - in the 80% of the cases, replace the masked token with [MASK]
        # - in the 10% of the cases, replace the masked token with a random token
        # - in the 10% of the cases, keep the masked token unchanged - need to be predicted (loss is computed only on the masked tokens)
        probability_matrix = torch.full(labels.shape, 0.15)
        masked_token_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_token_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_token_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_token_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) keep the masked input tokens unchanged

        # padding
        input_ids = torch.nn.functional.pad(
            input_ids,
            (0, self.max_length - input_ids.shape[0]),
            value=self.tokenizer.pad_token_id,
        )

        labels = torch.nn.functional.pad(
            labels,
            (0, self.max_length - labels.shape[0]),
            value=-100,
        )

        attention_mask = torch.nn.functional.pad(
            attention_mask,
            (0, self.max_length - attention_mask.shape[0]),
            value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_index": torch.tensor(self.task_index, dtype=torch.long),
        }
