from transformers import (
    BertModel,
)
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl


import pyro.distributions as dist
import pyro.distributions.transforms as T


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

import os
import pickle


# load indices of embeddings that corresponding to training flow, training classifier and inference
def load_embd_idx(save_path):
    flow_idx = pickle.load(open(save_path + "flow_idx.pkl", "rb"))
    cls_idx = pickle.load(open(save_path + "cls_idx.pkl", "rb"))
    inf_idx = pickle.load(open(save_path + "inf_idx.pkl", "rb"))
    return flow_idx, cls_idx, inf_idx


# Return the top K sentiment words based on contributions to sentiment classes. based on a freq dict generated based on imdb_get_att_mats.py
def topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer):
    word_list_dict = {}
    for keyword in ["pos_freq", "neg_freq", "others_freq"]:
        keyword_sort = 2 if keyword == "pos_freq" else 1
        senti_dist = senti_dist_dict[keyword]
        K = min([K, len(senti_dist)])
        temp_list = [[k] + v for k, v in senti_dist.items()]
        if keyword == "others_freq":
            # 'others_freq'
            temp_list.sort(key=lambda x: (x[1] * x[2], -x[3]), reverse=True)
            # sort critiria: most impure (large gini index, in descending order) -> number of apprearance (in ascending order)
        else:
            # 'pos_freq' or 'neg_freq'
            temp_list.sort(key=lambda x: x[keyword_sort], reverse=True)

        topK_target_token_list = [l[0] for l in temp_list[:K]]
        topK_target_ids = tokenizer.convert_tokens_to_ids(topK_target_token_list)
        word_list_dict["_".join(["W", keyword.split("_")[0], "id"])] = topK_target_ids
    return word_list_dict


# Return the top K frequent sentiment tokens based on a freq dictionary generated based on words_list.py
def topK_from_freq_lists(freq_dict, K, tokenizer):
    pos_freq = freq_dict["pos_freq"]
    neg_freq = freq_dict["neg_freq"]
    others_freq = freq_dict["others_freq"]

    K1 = min([K, len(pos_freq)])
    K2 = min([K, len(neg_freq)])
    K3 = min([K, len(others_freq)])
    W_pos = [word for word, _ in pos_freq.most_common(K1)]
    W_neg = [word for word, _ in neg_freq.most_common(K2)]
    W_others = [word for word, _ in others_freq.most_common(K3)]
    W_pos_id, W_neg_id, W_others_id = (
        tokenizer.convert_tokens_to_ids(W_pos),
        tokenizer.convert_tokens_to_ids(W_neg),
        tokenizer.convert_tokens_to_ids(W_others),
    )

    word_list_dict = {
        "W_pos": W_pos,
        "W_pos_id": W_pos_id,
        "W_neg": W_neg,
        "W_neg_id": W_neg_id,
        "W_others": W_others,
        "W_others_id": W_others_id,
    }
    return word_list_dict


# Add new masking based on id in word_ids; returns the new masking
# IMPORTANT: the following discussion indicates that attention_mask argument is not used at all when training:
# (cont.) https://stackoverflow.com/questions/60397610/use-of-attention-mask-during-the-forward-pass-in-lm-finetuning
def wordmask(input_ids, attention_mask, word_list_ids):
    word_list_ids = set(word_list_ids)
    temp = attention_mask.clone().detach()
    for i, id in enumerate(input_ids):
        if id.item() in word_list_ids:
            temp[i] = 0
    return temp


# e.g. attention_mask_new = wordmask(encoding["input_ids"], encoding['attention_mask'], W_pos_id)


# Convet review score to sentiment classification
def switcher(score):
    if score <= 2:
        return 0  # negative
    elif score == 3:
        return 1  # neutral
    else:
        return 2  # positive


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

    def training_epoch_end(self, outputs):
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


class litBERT(pl.LightningModule):
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs, targets)

        preds = outputs.argmax(dim=1, keepdim=True)
        acc = preds.eq(targets.view_as(preds)).sum().item() / len(preds)
        # log both training loss and accuracy at the end of each epoch, and show them with progress bar
        self.log(f"{prefix}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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
        self.log(f"{prefix}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Prepare IMDB dataset
class IMDBdataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(target, dtype=torch.long),
        }


# prepare IMDB dataset with sentiment-word screening using replacement strategy
class IMDBdataset_replace(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len, word_list_ids=None):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word_list_ids = set(word_list_ids) if word_list_ids is not None else None

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # replace the sentiment words
        if self.word_list_ids is not None:
            for i, token_id in enumerate(input_ids):
                token_id = token_id.item()
                if token_id in self.word_list_ids:
                    input_ids[i] = 0
                    attention_mask[i] = 0

        return {
            "review_text": review,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(target, dtype=torch.long),
        }


# prepare IMDB review dataset, replace sentiment words (along with its neighbours) with non-sense token such as 'x' or '[PAD]' based on ATTENTION
# make sure that bert_model is sent to the same device
class IMDBdataset_att(Dataset):
    def __init__(
        self,
        reviews,
        targets,
        tokenizer,
        max_len,
        att_mats,
        word_list_ids=None,
        threshold_val=0.05,
        threshold_prop=0.1,
    ):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.att_mats = att_mats  # attention matrices of the first layer
        self.word_list_ids = word_list_ids
        self.threshold_val = threshold_val
        self.threshold_prop = threshold_prop

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        att_mat = self.att_mats[idx] if self.word_list_ids is not None else None

        input_ids_new, attention_mask_new = deepcopy(input_ids), deepcopy(
            attention_mask
        )

        # replace instead of removing (SUBJECT TO CHANGE)
        if self.word_list_ids is not None:
            # mask words from word_list_ids, it seems that only input_ids are useful
            # (cont.) see this discussion: https://stackoverflow.com/questions/60397610/use-of-attention-mask-during-the-forward-pass-in-lm-finetuning
            # (cont.) the implementation by HuggingFace does not seem to put much efforts in attention mask?
            for i, token_id in enumerate(input_ids):
                token_id = token_id.item()
                if token_id in self.word_list_ids:
                    att_weights = att_mat[i]  # attention paid by the token of interest
                    if self.threshold_prop is not None:
                        threshold_val = torch.quantile(
                            att_weights, 1 - self.threshold_prop
                        )
                    else:
                        threshold_val = (
                            self.threshold_val
                        )  # threshold value to screen out the high-attention contexts

                    input_ids_new[att_weights > threshold_val] = 0
                    attention_mask_new[att_weights > threshold_val] = 0

        return {
            "review_text": review,
            "input_ids": input_ids_new,
            "attention_mask": attention_mask_new,
            "targets": torch.tensor(target, dtype=torch.long),
        }


# Prepare IMDB embedding dataset
class IMDBembedding(Dataset):
    def __init__(self, embeddings, sentiments, onehot=False):
        self.embeddings = embeddings
        self.sentiments = sentiments
        self.onehot = onehot

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        if self.onehot:
            y = [1, 0] if self.sentiments[idx] else [0, 1]
            y = torch.tensor(y, dtype=torch.float)
        else:
            y = torch.tensor(int(self.sentiments[idx]), dtype=torch.long)
        return x, y


# process distillbert embeddings generated from if-else part of embd_affine_flow.py
def process_distillbert_embedding(save_path):
    predicts = []
    indices = []
    for f in os.listdir(save_path):
        if f.startswith("predictions"):
            predicts.append(torch.load(os.path.join(save_path, f)))
        if f.startswith("batch_indices"):
            indices.append(torch.load(os.path.join(save_path, f)))

    embeddings, targets, idxs = [], [], []
    for process_idx in range(len(predicts)):
        temp = predicts[process_idx][0]
        embeddings.extend([temp[i][0] for i in range(len(temp))])
        targets.extend([temp[i][1] for i in range(len(temp))])

        temp_idx = indices[process_idx][0]
        idxs.extend(
            [torch.tensor(temp_idx[i], dtype=torch.int) for i in range(len(temp_idx))]
        )

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)
    idxs = torch.cat(idxs, dim=0)
    return embeddings, targets, idxs


# Sentiment classifier with tunable BERT and classification layer
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, n_classes, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).values()
        output = self.drop(pooled_output)
        return self.out(output)


# Sentiment classifier, trained help function, on training dataset
def train_epoch(
    model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch, EPOCHS
):
    model = model.train()
    losses = []
    correct_predictions = 0
    loop = tqdm(
        data_loader, leave=False, dynamic_ncols=True
    )  # use leave = False to print progress bar on the same line
    for d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        correct_predictions += torch.sum(preds == targets)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        # update progress bar
        loop.set_description(f"Training: Epoch [{epoch}/{EPOCHS}]")

    return correct_predictions.double() / n_examples, np.mean(losses)


# Sentiment classifier, evaluation help function, on validation dataset
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# BERT embeddings. Returns the pooler_output (output of a dense layer applied on the classification head)
class BertEmbedding(nn.Module):
    def __init__(self, from_pretrained="bert-base-cased", checkpoint=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(from_pretrained)
        if checkpoint is not None:
            saved_bert_state_dict = torch.load(checkpoint, map_location="cpu")
            self.bert.load_state_dict(saved_bert_state_dict)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)["pooler_output"]


# Get BERT embeddings based on the dataloader
def get_bert_embeddings(model, dl, device, target=False):
    model.eval()
    embeddings = []
    targets = []
    for encodings in tqdm(dl, leave=False):
        input_ids = encodings["input_ids"].to(device)
        attention_masks = encodings["attention_mask"].to(device)
        with torch.no_grad():
            pooler_output = model(input_ids, attention_masks)
        embeddings.append(pooler_output.cpu())
        if target:
            targets.extend(encodings["targets"])

    bert_embeddings = torch.concat(embeddings)
    targets = torch.tensor(targets)
    if target:
        return bert_embeddings, targets
    else:
        return bert_embeddings


# plot pairwise scatter plots of bert embeddings
def pairwise_bertembeddings(
    bert_embeddings,
    pairlist,
    nrows,
    ncols,
    savepath=None,
    fake_embeddings=None,
    kde=False,
):
    p = len(bert_embeddings[0])
    if pairlist is None:
        pairlist = [np.random.choice(range(p), 2, False) for _ in range(nrows * ncols)]

    assert (
        len(pairlist) == nrows * ncols
    ), "Length of pairlist does not mathc specivied nrows and ncols"

    if fake_embeddings is not None:
        assert len(bert_embeddings[0]) == len(
            fake_embeddings[0]
        ), "Dimension mismatch between bert_embeddings and fake_embeddings"

    fig = plt.figure(figsize=(12 * ncols, 10 * nrows))
    for i in range(nrows * ncols):
        c1, c2 = pairlist[i]
        plt.subplot(nrows, ncols, 1 + i)
        plt.title(f"x_{c1 + 1} Versus x_{c2 + 1}", fontsize=20)
        plt.xlabel(f"x_{c1 + 1}", fontsize=16)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.ylabel(f"x_{c2 + 1}", fontsize=16)
        if kde:
            sns.kdeplot(
                x=bert_embeddings[:, c1],
                y=bert_embeddings[:, c2],
                cmap="Blues",
                shade=True,
            )
            plt.scatter([], [], s=20, alpha=0.3, label="True", color="C0")
        else:
            plt.scatter(
                bert_embeddings[:, c1],
                bert_embeddings[:, c2],
                s=20,
                alpha=0.3,
                label="True",
            )

        if fake_embeddings is not None:
            plt.scatter(
                fake_embeddings[:, c1],
                fake_embeddings[:, c2],
                s=20,
                alpha=0.3,
                color="firebrick",
                label="Flow",
            )

            plt.legend(fontsize=18)
    if savepath is not None:
        plt.savefig(savepath)
    plt.close()
    return fig


# RealNVP flow: AC + BN + PERT + AC + ... + AC + BN + PERT + AC
# Returns a list of transforms
def NF_RealNVP(input_dim, num_flows, num_layers, num_hidden_units, device="cpu"):
    realnvp = []
    for _ in range(num_flows - 1):
        # Affine coupling layer with dense NN
        realnvp.append(
            T.affine_coupling(
                input_dim=input_dim, hidden_dims=[num_hidden_units] * num_layers
            )
        )
        # Learnable batch normalization layer for stabalization
        realnvp.append(T.batchnorm(input_dim=input_dim))
        # Reverse permutation transform for variability
        realnvp.append(
            T.permute(
                input_dim=input_dim,
                permutation=torch.tensor(range(input_dim - 1, -1, -1)),
            )
        )

    realnvp.append(
        T.affine_coupling(
            input_dim=input_dim, hidden_dims=[num_hidden_units] * num_layers
        )
    )

    return move_realnvp(realnvp, device)


# Move the affine coupling layer and the batch norm layer to designated device
def move_realnvp(realnvp, device):
    if device != "cpu":
        for i in range(len(realnvp)):
            if hasattr(realnvp[i], "to"):
                # the affine coupling layer and batchnorm layer
                realnvp[i] = realnvp[i].to(device)
            if hasattr(realnvp[i], "permutation"):
                # the reverse permutation layer
                realnvp[i].permutation = realnvp[i].permutation.to(device)

    return realnvp


# save checkpoint with device tag the one that was originally trained on
def realnvp_cpsave(epochs, realnvp, optimizer, scheduler, cp_save_path):
    checkpoint_dict = defaultdict(list)
    # Trained number of epochs
    checkpoint_dict["epochs"] = epochs
    # State dictionaries of all parametrized layers of RealNVP
    for i, transform in enumerate(realnvp):
        if hasattr(transform, "state_dict"):
            checkpoint_dict["realnvp"].append((i, transform.state_dict()))

    # State dictionaries of optimizer and scheduler
    checkpoint_dict["optimizer"] = optimizer.state_dict()
    checkpoint_dict["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint_dict, cp_save_path)


# load saved checkpoint of realnvp
# MAKE SURE that BOTH realnvp and state_dicts are initialized first on CPU to avoid GPU OOM error
def realnvp_cpload(p, num_flows, num_layers, num_hidden_units, state_dicts, device):
    # initialize a realnvp transform list on CPU
    realnvp = NF_RealNVP(p, num_flows, num_layers, num_hidden_units, "cpu")
    # load state_dict of affine coupling layer and batchnorm layer
    for i, state_dict in state_dicts["realnvp"]:
        realnvp[i].load_state_dict(state_dict)
    realnvp = move_realnvp(realnvp, device)

    # load state_dict of optimizer
    params = NF_params(realnvp)
    optimizer = torch.optim.Adam(params)
    optimizer.load_state_dict(state_dicts["optimizer"])

    # load state_dict of scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler.load_state_dict(state_dicts["scheduler"])
    return realnvp, optimizer, scheduler


# Get all learnable parameters in a NF list of transforms
def NF_params(transformlist):
    params = []
    for tr in transformlist:
        if hasattr(tr, "parameters"):
            params += list(tr.parameters())
    print(
        f"Total number of learnable parameters: {sum([len(para.flatten()) for para in params])}"
    )
    return params


# Set realnvp at train or evaluation mode
def realnvp_mode(realnvp, mode):
    for i, transform in enumerate(realnvp):
        if mode == "train":
            if hasattr(transform, "train"):
                realnvp[i].train()
        if mode == "eval":
            if hasattr(transform, "eval"):
                realnvp[i].eval()


def train_realNVP_oneepoch(
    epoch, dataloader, transformed_distribution, optimizer, losslog, device, writer
):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for batch_idx, x_batch in loop:
        x_batch = x_batch.to(device)
        loss = -transformed_distribution.log_prob(x_batch).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        transformed_distribution.clear_cache()

        # Monitor the training curve with running loss on Tensorboard
        lossval = loss.detach().cpu()
        losslog.append(lossval)
        writer.add_scalar(
            "Training running loss", lossval, epoch * len(dataloader) + batch_idx + 1
        )
        loop.set_description(f"Epoch [{epoch + 1} / {len(dataloader)}]")
        loop.set_postfix(est_loss=lossval.item())


def eval_realNVP(epoch, dataloader, transformed_distribution, device, writer):
    loss_list = []
    for x_batch in tqdm(dataloader, leave=False):
        x_batch = x_batch.to(device)
        with torch.no_grad():
            loss = -transformed_distribution.log_prob(x_batch).mean()
            loss_list.append(loss.cpu())
        transformed_distribution.clear_cache()

    lossval = torch.mean(torch.tensor(loss_list))
    # Monitor the training curve with total loss on Tensorboard
    writer.add_scalar("Training all loss", lossval.item(), epoch + 1)
    return lossval


# get predictions and true labels from a data_loader based on a model
def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values


# # example of using get_predictions
# print(classification_report(y_test, y_pred, target_names=class_names))


# plot the confusion matrix, note that input is actually a dataframe, see the example
def show_confusion_matrix(confusion_matrix, path):
    plt.figure(figsize=(8, 7))
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")
    plt.savefig(path)
    plt.close()


# # example of using show_confusion_matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
# show_confusion_matrix(df_cm)
