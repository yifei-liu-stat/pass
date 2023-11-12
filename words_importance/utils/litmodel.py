import numpy as np

import torch
from torch import nn


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, BasePredictionWriter


import pyro.distributions as dist
import pyro.distributions.transforms as T

import os

try:
    from visualization import pairwise_bertembeddings
except:
    from utils.visualization import pairwise_bertembeddings


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


class DistillBertEmd(litDistillBERT):
    def __init__(self, model=None, lr=0.001):
        super().__init__(model, lr)

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        targets = batch["targets"]
        all_embeddings = self.model.distilbert(
            input_ids=input_ids, attention_mask=attention_masks
        )["last_hidden_state"]

        # embedding of the [CLS] token
        return all_embeddings[:, 0, :], targets


class NF_affine_cond(nn.Module):
    def __init__(self, input_dim, context_dim, num_flow_layers, num_hidden_units):
        super().__init__()
        # Use conditional affine coupling layer to incorporate class label infomation
        self.spline_coupling = nn.ModuleList(
            [
                T.conditional_affine_coupling(
                    input_dim=input_dim,
                    context_dim=context_dim,
                    hidden_dims=num_hidden_units,
                )
                for _ in range(num_flow_layers)
            ]
        )

        self.batch_norm = nn.ModuleList(
            [T.batchnorm(input_dim=input_dim) for _ in range(num_flow_layers)]
        )

        self.reverse_tensor = nn.Parameter(
            torch.tensor(range(input_dim - 1, -1, -1)), requires_grad=False
        )
        self.reverse = [
            T.permute(input_dim=input_dim, permutation=self.reverse_tensor)
            for _ in range(num_flow_layers)
        ]

        self.transformlist = []
        for i in range(num_flow_layers):
            self.transformlist.extend(
                [self.spline_coupling[i], self.batch_norm[i], self.reverse[i]]
            )
        self.transformlist.pop()  # get rid of the last permutation layer

        self.base_dist_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.base_dist_scale = nn.Parameter(torch.ones(input_dim), requires_grad=False)
        self.base_dist = dist.Normal(
            loc=self.base_dist_mean, scale=self.base_dist_scale
        )

        self.target_dist_giveny = dist.ConditionalTransformedDistribution(
            self.base_dist, self.transformlist
        )

    def nll(self, x_batch, y_batch):
        # prior distribution of the target is 0.5/0.5
        return (
            -0.5 * self.target_dist_giveny.condition(y_batch).log_prob(x_batch).mean()
        )

    def sample_joint(self, sample_size=1, one_hot=False):
        # sample joint distribution of (txt_embedding, sentiment), the prior of positive sentiment is 0.5
        targets = torch.tensor(torch.rand(sample_size) < 0.5, dtype=torch.int)
        targets_onehot = torch.tensor(
            [[1, 0] if t else [0, 1] for t in targets], dtype=torch.float
        )
        embeddings = self.target_dist_giveny.condition(targets_onehot).sample(
            [sample_size]
        )
        if one_hot:
            return embeddings, targets_onehot
        else:
            return embeddings, targets


class litNFAffine(pl.LightningModule):
    def __init__(
        self,
        model=None,
        lr=None,
        scheduler_total_steps=None,
        embeddings=None,
        targets=None,
        log_epoch_intervals=10,
        plot_samples=1000,
        pairwise_list=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.scheduler_total_steps = scheduler_total_steps
        self.embeddings = embeddings
        self.targets = targets
        self.log_epoch_intervals = log_epoch_intervals
        self.plot_samples = plot_samples
        self.pairwise_list = pairwise_list

    def on_train_batch_start(self, batch, batch_idx):
        self.model.target_dist_giveny.clear_cache()

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        loss = self.model.nll(x_batch, y_batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.log_epoch_intervals == 0:
            tensorboard_logger = self.logger.experiment
            fig_pos, fig_neg = self.plot_true_fake_samples()
            tensorboard_logger.add_figure(
                tag=f"True-versus-fake-embeddings-pos-{self.current_epoch}",
                figure=fig_pos,
            )
            tensorboard_logger.add_figure(
                tag=f"True-versus-fake-embeddings-neg-{self.current_epoch}",
                figure=fig_neg,
            )

    # # Deprecated in v2.0.0.
    # def training_epoch_end(self, outputs):
    #     if (self.current_epoch + 1) % self.log_epoch_intervals == 0:
    #         tensorboard_logger = self.logger.experiment
    #         fig_pos, fig_neg = self.plot_true_fake_samples()
    #         tensorboard_logger.add_figure(
    #             tag=f"True-versus-fake-embeddings-pos-{self.current_epoch}",
    #             figure=fig_pos,
    #         )
    #         tensorboard_logger.add_figure(
    #             tag=f"True-versus-fake-embeddings-neg-{self.current_epoch}",
    #             figure=fig_neg,
    #         )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # ceiling value of lr
            total_steps=self.scheduler_total_steps,
            pct_start=0.3,  # defaults, percentage of increasing part
            anneal_strategy="linear",  # alternative: 'cos'
            cycle_momentum=True,  # defaults, set to False if optimizer does not have moment argument
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def plot_true_fake_samples(self):
        # generate plots for comparing distributions
        idx = np.random.choice(
            range(len(self.embeddings)), size=self.plot_samples, replace=False
        )
        true_embeddings = self.embeddings[idx, :]
        true_targets = self.targets[idx]
        true_embeddings_pos = true_embeddings[true_targets == 1]
        true_embeddings_neg = true_embeddings[true_targets == 0]

        fake_embeddings_pos = (
            self.model.target_dist_giveny.condition(
                torch.tensor(
                    [[1, 0] for _ in range(len(true_embeddings_pos))],
                    dtype=torch.float,
                    device=self.device,
                )
            )
            .sample([len(true_embeddings_pos)])
            .cpu()
        )
        fake_embeddings_neg = (
            self.model.target_dist_giveny.condition(
                torch.tensor(
                    [[0, 1] for _ in range(len(true_embeddings_neg))],
                    dtype=torch.float,
                    device=self.device,
                )
            )
            .sample([len(true_embeddings_neg)])
            .cpu()
        )

        fig_pos = pairwise_bertembeddings(
            bert_embeddings=true_embeddings_pos,
            pairlist=self.pairwise_list,
            nrows=2,
            ncols=2,
            fake_embeddings=fake_embeddings_pos,
        )

        fig_neg = pairwise_bertembeddings(
            bert_embeddings=true_embeddings_neg,
            pairlist=self.pairwise_list,
            nrows=2,
            ncols=2,
            fake_embeddings=fake_embeddings_neg,
        )

        return fig_pos, fig_neg


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )
