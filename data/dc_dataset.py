import random
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer


class DCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name_or_path: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        task_index: int = 2,
        prob_italian_injection: float = 0.5,
        italian_label_id: int = None,
        **kwargs,
    ):
        """
        This class instantiates the pre-training dataset for the dialect classification task.
        :param texts: List of texts to be tokenized.
        :param labels: List of labels for each text. It corresponds to the class of the text (dialect + italian).
        :param tokenizer_name_or_path: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
        :param max_length: The maximum length of the tokenized text.
        :param padding: The padding strategy to be used. Available options are available in the transformers library.
        :param truncation: Whether to truncate the text or not.
        :param task_index: The index of the task in the pre-training pipeline.
        """

        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.task_index = task_index
        self.prob_italian_injection = prob_italian_injection
        self.italian_label_id = italian_label_id

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

        # with random probability (prob_italian_injection) we inject additional italian text but we keep the same label
        if random.random() < self.prob_italian_injection:
            # get a random text having italian label
            italian_texts = [
                text
                for text, label in zip(self.texts, self.labels)
                if label == self.italian_label_id
            ]
            italian_text = random.choice(italian_texts)
            # randomly mix before or after the text
            if random.random() < 0.5:
                text = italian_text + self.texts[index]
            else:
                text = self.texts[index] + italian_text
        else:
            text = self.texts[index]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "task_index": torch.tensor(self.task_index, dtype=torch.long),
        }
