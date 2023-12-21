import lightning as pl
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
from torchmetrics.classification import Accuracy
import torch


class BertTrainingModule(pl.LightningModule):
    def __init__(self, n_classes, lr):
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=self.n_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # Scalar Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_classes)

    def forward(self, x):
        return self.model(**x)

    def model_step(self, batch):
        x = {
            "ids": batch["ids"],
            "mask": batch["mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        y = batch["labels"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, axis=1)

        return {"preds": preds, "logits": logits, "loss": loss}

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs["loss"]
        preds = outputs["preds"]

        self.train_acc(preds, batch["labels"])

        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs["loss"]
        preds = outputs["preds"]

        self.val_acc(preds, batch["labels"])

        self.log(
            "val_acc",
            self.val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
        # TODO implement test step
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
