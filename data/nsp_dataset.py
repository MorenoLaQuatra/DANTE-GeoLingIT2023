import random
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer


class NSPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer_name_or_path: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        task_index: int = 1,
        **kwargs,
    ):
        """
        This class instantiates the pre-training dataset for the next sentence prediction task.
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
        :param index: The index of the text in the corpus.
        :return: A dictionary containing the tokenized text (with attention mask) and the label.
        """
        # select another sentence from the corpus
        # - 50% of the time, the sentence B is the next sentence
        # - 50% of the time, the sentence B is a random sentence from the corpus

        is_really_next = random.random() < 0.5

        # split the text into sentences
        sentences = self.texts[index].split(".")
        # select a random sentence from the text
        sentence_index = random.randint(0, len(sentences) - 1)
        sentence_a = sentences[sentence_index]

        if is_really_next:
            # select the next sentence
            if sentence_index + 1 >= len(sentences):  # edge case
                sentence_a = sentences[sentence_index - 1]
                sentence_b = sentences[sentence_index]
            else:
                sentence_b = sentences[sentence_index + 1]
        else:
            # select a random sentence from the corpus
            random_index = random.randint(0, len(self.texts) - 1)
            random_sentences = self.texts[random_index].split(".")
            random_sentence_index = random.randint(0, len(random_sentences) - 1)
            sentence_b = random_sentences[random_sentence_index]

        # tokenize the text
        enc = self.tokenizer(
            sentence_a,
            sentence_b,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # convert bool to int
        is_really_next = int(is_really_next)

        # convert all dtypes
        input_ids = enc["input_ids"][0].long()
        attention_mask = enc["attention_mask"][0].long()
        labels = torch.tensor(is_really_next, dtype=torch.long).float()
        task_index = torch.tensor(self.task_index, dtype=torch.long).long()

        # assert dtype
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long
        assert labels.dtype == torch.float
        assert task_index.dtype == torch.long

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_index": task_index,
        }
