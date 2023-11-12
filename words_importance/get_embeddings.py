"""
Get embeddings from fine-tuned DistilBERT with sentiment masking.
- Will use checkpoints in ./ckpt/ckpt_distilbert_att/.
- Embeddings are saved in ./data/ followed by the naming convention of "imdb-distillbert-{mask_keyword}-top{topK}-tp{threshold_proportion}".
"""


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from pytorch_lightning import Trainer, seed_everything

from torch.utils.data import DataLoader

import pandas as pd


import pickle
import argparse
import os

from utils.data import (
    IMDBdataset_att,
    load_embd_idx,
    topK_from_senti_dist_dict,
)

from utils.litmodel import (
    DistillBertEmd,
    CustomWriter,
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
    default=[0, 1],
    help="List of device IDs for multi-GPU training",
)

args = parser.parse_args()


# Set up file system and structure
device_ids = args.device_ids
pretrained_model_name = "distilbert-base-uncased"


data_dir = "./data"
imdb_data_path = os.path.join(data_dir, "imdb_reviews.csv")
senti_dist_path = os.path.join(data_dir, "senti_dist_dict.pkl")
att_mat_path = os.path.join(data_dir, "att_mats_pooled.pt")

ckpt_dir = "./ckpt/ckpt_distilbert_att"


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
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)


SEED = 2023
seed_everything(SEED, workers=True)


print("Preparing IMDB embeddings ...")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

## choose sentiment word collection
senti_dist_dict = pickle.load(open(senti_dist_path, "rb"))
word_lists = topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer)
word_list_ids = word_lists[word_list_keyword] if word_list_keyword is not None else None

print("Loading IMDB dataset ...")
df = pd.read_csv(imdb_data_path)
df["sentiment"] = df.sentiment.apply(lambda s: 0 if s == "negative" else 1)

print("Loading pooled attention matrices from BERT...")
att_mats_pooled = (
    pickle.load(open(att_mat_path, "rb")) if word_list_keyword is not None else None
)

ds = IMDBdataset_att(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=512,
    att_mats=att_mats_pooled,
    word_list_ids=word_list_ids,
    threshold_val=0,
    threshold_prop=threshold_proportion,
)

dl = DataLoader(ds, batch_size=16, num_workers=4, pin_memory=True, shuffle=False)

pred_writer = CustomWriter(output_dir=imdb_ebd_dir, write_interval="epoch")

trainer = Trainer(
    accelerator="gpu",
    devices=device_ids,
    strategy="ddp",
    precision="16",
    callbacks=[pred_writer],
    deterministic=True,
    logger=False,
)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name, num_labels=2, output_attentions=True
)
litmodel = DistillBertEmd(model=model)

## Get the checkpoint model for evaluating masked embeddings
for checkpoint_name in os.listdir(ckpt_dir):
    if checkpoint_name.startswith(ckpt_prefix):
        break

trainer.predict(
    model=litmodel,
    dataloaders=dl,
    ckpt_path=os.path.join(ckpt_dir, checkpoint_name),
    return_predictions=False,
)
