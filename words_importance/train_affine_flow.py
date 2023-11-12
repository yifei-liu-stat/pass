"""
Train an affine coupling flow for sentiment-masked text embeddings.
- ./results/tensorboard_logging_affine/: training curves including callbacks for comparing synthetic embeddings and true embeddings
- ./ckpt/ckpt_affineflow/: checkpoints for the flow model, followed by the naming convention of "imdb-distillbert-{mask_keyword}-top{topK}-tp{threshold_proportion}-train_loss...".
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Subset

import argparse
import os

from utils.data import (
    IMDBembedding,
    load_embd_idx,
    process_distillbert_embedding,
)

from utils.litmodel import (
    NF_affine_cond,
    litNFAffine,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--mask-keyword",
    type=str,
    default="W_others_id",
    choices=[None, "W_pos_id", "W_neg_id", "W_others_id"],
    help="Keyword for specifying the mask",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=600,
    help="top K sentiment words will be used for masking",
)
parser.add_argument(
    "-tp",
    "--threshold-proportion",
    type=float,
    default=0.02,
    help="Threshold proportions to screen out high-attention tokens",
)
parser.add_argument(
    "--device_ids",
    type=int,
    nargs="+",
    default=[4, 5, 6, 7],
    help="List of device IDs for multi-GPU training",
)

args = parser.parse_args()


# Set up file system and structure
device_ids = args.device_ids
pretrained_model_name = "distilbert-base-uncased"

data_dir = "./data"
result_dir = "./results"
flow_ckpt_dir = "./ckpt/ckpt_affineflow"

word_list_keyword = args.mask_keyword
K, threshold_proportion = args.topk, args.threshold_proportion
print(
    f"Masking sentiment keyword: {word_list_keyword}; top K: {K}; threshold_proportion: {threshold_proportion}"
)
temp = "none" if word_list_keyword is None else word_list_keyword.split("_")[1]
temp2 = (
    f"tp{0:0>3n}"
    if threshold_proportion is None
    else f"tp{100 * threshold_proportion:0>3n}"
)
imdb_ebd_dir = os.path.join(
    data_dir, "-".join(["imdb-distillbert", temp, f"top{K}", temp2])
)
ckpt_prefix = "-".join(["imdb-distillbert", temp, f"top{K}", temp2])


# Load indices for splitting the IMDB text embeddings (holdout + training + inference)
flow_idx, cls_idx, inf_idx = load_embd_idx(os.path.join(data_dir, "ebd_split_idx"))

# Process loaded/generated embeddings
print("Loading embeddings ...")
embeddings, targets, idxs = process_distillbert_embedding(imdb_ebd_dir)


SEED = 2023
seed_everything(SEED, workers=True)


# 2. Design an affine coupling flow
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 7e-5
TFPLOT_LOG_EPOCH_INTERVALS = 10
TFPLOT_NSAMPLES = 1000
PAIRWISE_LIST = [(494, 253), (416, 592), (759, 516), (538, 687)]

ds = IMDBembedding(embeddings=embeddings, sentiments=targets, onehot=True)

ds_flow = Subset(ds, flow_idx)
dl_flow = DataLoader(ds_flow, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)


nfmodel = NF_affine_cond(embeddings.shape[1], 2, 10, [8192, 4096, 2048])
lit_nfmodel = litNFAffine(
    model=nfmodel,
    lr=LEARNING_RATE,
    scheduler_total_steps=EPOCHS * len(dl_flow),
    embeddings=embeddings,
    targets=targets,
    log_epoch_intervals=TFPLOT_LOG_EPOCH_INTERVALS,
    plot_samples=TFPLOT_NSAMPLES,
    pairwise_list=PAIRWISE_LIST,
)


tensorboard = pl.loggers.TensorBoardLogger(
    save_dir=os.path.join(result_dir, "tensorboard_logging_affine"), name=ckpt_prefix
)

flow_checkpoint_name = ckpt_prefix + "-flow-{train_loss:.2f}"
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    mode="min",
    monitor="train_loss",
    dirpath=flow_ckpt_dir,
    filename=flow_checkpoint_name,
)

trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=device_ids,
    strategy="ddp",
    precision=16,
    # gradient_clip_val = 1,
    # deterministic = True,
    callbacks=[checkpoint_callback],
    log_every_n_steps=20,
    logger=tensorboard,
    # auto_lr_find = True
)


trainer.fit(model=lit_nfmodel, train_dataloaders=dl_flow)
