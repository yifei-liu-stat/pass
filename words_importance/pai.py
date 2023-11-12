"""
Calcuate test statistic and obtain p-value using estimated null distribution.
"""

import numpy as np

from pytorch_lightning import seed_everything

import pickle
import argparse
import os

from utils.data import (
    load_embd_idx,
    process_distillbert_embedding,
    get_cls_inf_dls,
    get_test_stat,
)




parser = argparse.ArgumentParser(description = 'Perform MC inference on collections of sentiment words using PAI')
parser.add_argument(
    "-m",
    "--mask-keyword",
    type=str,
    default=None,
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
    default=[0],
    help="List of device IDs for multi-GPU training",
)

args = parser.parse_args()


# Set up file system and structure
device_ids = args.device_ids


data_dir = "./data"
nulldist_dir = "./results/null_dist"
if not os.path.exists(nulldist_dir):
    os.makedirs(nulldist_dir)

# temporary checkpoint folder for classification heads
temp_ckpt_dir = "./ckpt/ckpt_clshead"

flow_ckpt_dir = "./ckpt/ckpt_affineflow"



word_list_keyword = args.mask_keyword
K, threshold_proportion = args.topk, args.threshold_proportion
print(
    f"Masking sentiment keyword: {word_list_keyword}; top K: {K}; threshold_proportion: {threshold_proportion}"
)
senti_word = "none" if word_list_keyword is None else word_list_keyword.split("_")[1]
temp2 = (
    f"tp{0:0>3n}"
    if threshold_proportion is None
    else f"tp{100 * threshold_proportion:0>3n}"
)
imdb_ebd_dir = os.path.join(
    data_dir, "-".join(["imdb-distillbert", senti_word, f"top{K}", temp2])
)
imdb_ebd_dir_none = os.path.join(
    data_dir, "-".join(["imdb-distillbert-none", f"top{K}", temp2])
)
ckpt_prefix = "-".join(["imdb-distillbert", senti_word, f"top{K}", temp2])
for ckpt_flow in os.listdir(flow_ckpt_dir):
    if ckpt_flow.startswith(ckpt_prefix):
        break





ebd_dimension = 768
flow_idx, cls_idx, inf_idx = load_embd_idx(os.path.join(data_dir, "ebd_split_idx"))
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)


SEED = 2023
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 5e-5




seed_everything(SEED, workers = True)

# 1. Get the test statistics from the original sample/embeddings
print("Calculating test statistics from the original sample/embeddings ...")
embeddings_senti, targets_senti, _ = process_distillbert_embedding(imdb_ebd_dir)
dl_cls_pos, dl_inf_pos = get_cls_inf_dls(embeddings_senti, targets_senti, cls_idx, inf_idx, cls_bs=BATCH_SIZE)

embeddings_none, targets_none, _ = process_distillbert_embedding(imdb_ebd_dir_none)
dl_cls_none, dl_inf_none = get_cls_inf_dls(embeddings_none, targets_none, cls_idx, inf_idx, cls_bs=BATCH_SIZE)


clshead_train_hparams = {
    "temp_ckpt_dir": temp_ckpt_dir,
    "ebd_dimension": ebd_dimension,
    "lr": LEARNING_RATE,
    "epochs": EPOCHS,
    "device_ids": device_ids
}

test_stat = get_test_stat(dl_cls_none, dl_inf_none, dl_cls_pos, dl_inf_pos, n_inf, senti_word=senti_word, **clshead_train_hparams)
print(f"Test statistic for testing {senti_word}:", test_stat.item())


# 2. Get the null distribution of the test statistics
print(f"Getting the null distribution of the test statistics for {senti_word} ...")
null_test_stat_list = pickle.load(open(os.path.join(nulldist_dir, senti_word + '.pkl'), "rb"))


# 3. Calculate the p-value
null_array = np.array(null_test_stat_list)
pvalue = np.mean(null_array >= test_stat.item())
print(f"P-value for testing {senti_word}:", pvalue)


