import lightning as pl
from datasets import load_dataset


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size: int = 32, eval_batch_size: int = 32):
        super().__init__()
        print("MNLIDataModule init")
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def prepare_data(self):
        # happens once on single CPU
        # do downloading, tokenizing, etc...
        dataset = load_dataset("glue", "mnli")
