import random
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer


class TCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name_or_path: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        task_index: int = 3,
        **kwargs,
    ):
        """
        This class instantiates the pre-training dataset for the token classification task.
        :param texts: List of texts to be tokenized.
        :param tokenizer_name_or_path: The tokenizer name or path to be used for tokenizing the texts.
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

    def __len__(self):
        """
        This function is called to get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, index):

        # randomly select the number of sentences to use for augmentation (between 2 and 10)
        n_sentences = random.randint(2, 10)
        # randomly select the number of tokens to use for each sentence (between 10 and max_length//n_sentences)
        n_tokens = [
            random.randint(10, self.max_length // n_sentences + 1)
            for _ in range(n_sentences + 1)
        ]  # +1 because we add the original sentence
        # randomly select the sentences to use for augmentation
        indexes = random.sample(range(len(self.texts)), n_sentences)
        sentences = [self.texts[i] for i in indexes]
        labels = [self.labels[i] for i in indexes]

        # add the original sentence to the list of sentences (at the beginning)
        sentences = [self.texts[index]] + sentences
        sentences_labels = [self.labels[index]] + labels

        input_ids = []
        attention_mask = []
        labels = []

        for i in range(len(sentences)):
            current_token_ids = self.tokenizer(
                sentences[i], add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            if current_token_ids.shape[0] > n_tokens[i]:
                # select a random starting position and get n_tokens[i] tokens
                start = random.randint(0, current_token_ids.shape[0] - n_tokens[i])
                current_token_ids = current_token_ids[start : start + n_tokens[i]]
                current_labels = torch.tensor(
                    [sentences_labels[i]] * n_tokens[i], dtype=torch.long
                )  # the label is the same for all tokens
                current_attention_mask = torch.tensor(
                    [1] * n_tokens[i], dtype=torch.long
                )
            else:
                # pad the token ids with the pad token id
                pad_token_id = self.tokenizer.pad_token_id
                current_attention_mask = torch.tensor(
                    [1] * current_token_ids.shape[0], dtype=torch.long
                )
                current_labels = torch.tensor(
                    [sentences_labels[i]] * current_token_ids.shape[0], dtype=torch.long
                )  # the label is the same for all tokens
                current_token_ids = torch.cat(
                    (
                        current_token_ids,
                        torch.tensor(
                            [pad_token_id] * (n_tokens[i] - current_token_ids.shape[0]),
                            dtype=torch.long,
                        ),
                    ),
                    dim=0,
                )
                current_attention_mask = torch.cat(
                    (
                        current_attention_mask,
                        torch.tensor(
                            [0] * (n_tokens[i] - current_attention_mask.shape[0]),
                            dtype=torch.long,
                        ),
                    ),
                    dim=0,
                )
                current_labels = torch.cat(
                    (
                        current_labels,
                        torch.tensor(
                            [-100] * (n_tokens[i] - current_labels.shape[0]),
                            dtype=torch.long,
                        ),
                    ),
                    dim=0,
                )

            input_ids.append(current_token_ids)
            attention_mask.append(current_attention_mask)
            labels.append(current_labels)

        # concatenate the token ids, attention masks and labels
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        labels = torch.cat(labels, dim=0)

        # check if trimming is needed
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        # check if padding is needed - use the pad function
        if input_ids.shape[0] < self.max_length:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, self.max_length - input_ids.shape[0]),
                value=self.tokenizer.pad_token_id,
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0
            )
            labels = torch.nn.functional.pad(
                labels, (0, self.max_length - labels.shape[0]), value=-100
            )

        # convert all dtype to torch.long
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        labels = labels.long()

        # check if any of the elements is of incorrect dtype
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long
        assert labels.dtype == torch.long

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_index": torch.tensor(self.task_index),
        }
