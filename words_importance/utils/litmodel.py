import torch

import pytorch_lightning as pl


class litDistillBERT(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix="val")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=targets
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

        preds = logits.argmax(dim=1, keepdim=True)
        acc = preds.eq(targets.view_as(preds)).sum().item() / len(preds)
        # log both training loss and accuracy at the end of each epoch, and show them with progress bar
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_acc",
            acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
