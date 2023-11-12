"""
Estimate null distribution of the test statistics using MC inference.
- Estimated null distribution will be saved in ./results/null_dist/{mask_keyword}.pkl.
"""

from pytorch_lightning import seed_everything

from tqdm import tqdm
import pickle
import argparse
import os

from utils.data import (
    load_embd_idx,
    get_test_stat,
    sample_cls_inf_dls_from_flow,
)

from utils.litmodel import (
    NF_affine_cond,
    litNFAffine,
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
ckpt_prefix = "-".join(["imdb-distillbert", senti_word, f"top{K}", temp2])
for ckptname_flow in os.listdir(flow_ckpt_dir):
    if ckptname_flow.startswith(ckpt_prefix):
        break





ebd_dimension = 768
flow_idx, cls_idx, inf_idx = load_embd_idx(os.path.join(data_dir, "ebd_split_idx"))
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)


BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 5e-5
clshead_train_hparams = {
    "temp_ckpt_dir": temp_ckpt_dir,
    "ebd_dimension": ebd_dimension,
    "lr": LEARNING_RATE,
    "epochs": EPOCHS,
    "device_ids": device_ids
}



SEED = 2023
seed_everything(SEED, workers = True)


# Get the null distribution of the test statistics
print("Estimating the null distribution of the test statistics using MC inference...")
nfmodel = NF_affine_cond(ebd_dimension, 2, 10, [8192, 4096, 2048])

# lit_nfmodel = litNFAffine()
lit_nfmodel = litNFAffine.load_from_checkpoint(os.path.join(flow_ckpt_dir, ckptname_flow), map_location = "cpu", model = nfmodel)

lit_nfmodel.eval()
null_test_stat_list = []
B = 200

for _ in tqdm(range(B), leave = False, dynamic_ncols = True):
    dl_cls_0, dl_inf_0 = sample_cls_inf_dls_from_flow(nfmodel, n_cls, n_inf, cls_bs=BATCH_SIZE)
    dl_cls_1, dl_inf_1 = sample_cls_inf_dls_from_flow(nfmodel, n_cls, n_inf, cls_bs=BATCH_SIZE)
    null_test_stat = get_test_stat(dl_cls_0, dl_inf_0, dl_cls_1, dl_inf_1, n_inf, senti_word=senti_word, **clshead_train_hparams)
    null_test_stat_list.append(null_test_stat)
    pickle.dump(null_test_stat_list, open(os.path.join(nulldist_dir, senti_word + '.pkl'), "wb"))




