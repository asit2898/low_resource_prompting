import lightning as pl
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class BertTrainingModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
    def model_step(self,batch):
        x = batch[0]
        y = batch[1]
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        return

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.lr)
