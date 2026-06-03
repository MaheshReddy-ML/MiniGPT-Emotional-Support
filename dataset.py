"""
Dataset utilities for supervised emotional-support language modeling.

EmotionalSupportDataset converts prompt/target JSON samples into fixed-size
token tensors. Prompt tokens and padding tokens are masked with -100 so the
loss is calculated only on the assistant response.
"""

import json

import torch

from torch.utils.data import Dataset

from tokenizers import Tokenizer


class EmotionalSupportDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer_path,
        max_length=512
    ):

        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )

        with open(
            data_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.samples = json.load(f)

        self.max_length = max_length

        self.pad_id = self.tokenizer.token_to_id(
            "[PAD]"
        )

        self.bos_id = self.tokenizer.token_to_id(
            "[BOS]"
        )

        self.eos_id = self.tokenizer.token_to_id(
            "[EOS]"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        prompt = sample["prompt"]

        target = sample["target"]

        prompt_text = (
            prompt
            + " <RESPONSE> "
        ).replace(
            "\n",
            " "
        )

        prompt_ids = self.tokenizer.encode(
            prompt_text
        ).ids

        target_ids = (
            self.tokenizer.encode(
                target
            ).ids
            + [self.eos_id]
        )

        # Reserve room for BOS and EOS

        if len(prompt_ids) >= self.max_length - 2:

            prompt_ids = prompt_ids[
                : self.max_length - 2
            ]

        available_space = (
            self.max_length
            - len(prompt_ids)
            - 1
        )

        target_ids = target_ids[
            :available_space
        ]

        full_ids = (
            [self.bos_id]
            + prompt_ids
            + target_ids
        )

        attention_mask = (
            [1]
            * len(full_ids)
        )

        padding_length = (
            self.max_length
            - len(full_ids)
        )

        full_ids += (
            [self.pad_id]
            * padding_length
        )

        attention_mask += (
            [0]
            * padding_length
        )

        input_ids = torch.tensor(
            full_ids,
            dtype=torch.long
        )

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.long
        )

        labels = input_ids.clone()

        # Mask BOS + prompt tokens

        prompt_length = (
            len(prompt_ids)
            + 1
        )

        labels[
            :prompt_length
        ] = -100

        # Mask padding

        labels[
            attention_mask == 0
        ] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


if __name__ == "__main__":

    dataset = EmotionalSupportDataset(
        data_path="data/train_pairs.json",
        tokenizer_path="tokenizer.json",
        max_length=128
    )

    sample = dataset[0]

    print(
        sample["input_ids"].shape
    )

    print(
        sample["attention_mask"].shape
    )

    print(
        sample["labels"].shape
    )

    print(
        sample["input_ids"][:40]
    )

    print(
        sample["labels"][:40]
    )

    bad_samples = 0

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        valid_targets = (
            sample["labels"] != -100
        ).sum().item()

        if valid_targets == 0:

            bad_samples += 1

    print(
        f"\nBad Samples: {bad_samples}"
    )