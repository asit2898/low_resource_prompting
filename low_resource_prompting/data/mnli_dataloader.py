from multiprocessing.spawn import prepare

import lightning as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch


class DictDataset(Dataset):
    """
    Tokenizer output is dict with keys dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'label', 'idx'])
    This makes the dict into a pytorch Dataset
    """

    def __init__(self, dict):
        self.dict = dict

    def __getitem__(self, index):
        d = {k: v[index] for k, v in self.dict.items()}
        return d

    def __len__(self):
        return len(self.dict["input_ids"])


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size: int = 32, val_batch_size: int = 32):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        # happens once on single CPU
        # do downloading, tokenizing, etc...
        dataset = load_dataset("glue", "mnli")

        # TODO make sure we dont need mismatched evaluation
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation_matched"]
        self.test_dataset = dataset["test_matched"]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.train_tokenized = self.tokenize_function(self.train_dataset)
        self.val_tokenized = self.tokenize_function(self.val_dataset)
        self.test_tokenized = self.tokenize_function(self.test_dataset)

    def tokenize_function(self, dataset):
        tokenized_dataset = self.tokenizer(
            text=dataset["premise"],
            text_pair=dataset["hypothesis"],
            add_special_tokens=True,
            padding="max_length",
            max_length=50,
            truncation=True,
            return_tensors="pt",
        ).data
        tokenized_dataset["label"] = torch.tensor(dataset["label"])
        tokenized_dataset["idx"] = torch.tensor(dataset["idx"])

        return tokenized_dataset

    def setup(self, stage: str = None):
        # happens on every GPU
        # do train/val/test splits here
        if stage == "fit" or stage is None:
            self.train_tokenized_dataset = DictDataset(self.train_tokenized)
            self.val_tokenized_dataset = DictDataset(self.val_tokenized)

        if stage == "test" or stage is None:
            self.test_tokenized_dataset = DictDataset(self.test_tokenized)

    def train_dataloader(self):
        return DataLoader(
            self.train_tokenized_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_tokenized_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_tokenized_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )
