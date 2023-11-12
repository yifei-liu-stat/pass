"""
Fine-tune DistilBERT to perform sentiment classification on IMDB review dataset with sentiment word masking. The following files are needed:
- imdb_reviews.csv
- senti_dist_dict.pkl
- att_mats_pooled.pt
which can all be obtained by running `python prepare.py` in the `words_importance` directory.
"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader

import pandas as pd

import pickle
import argparse
import os

from data import IMDBdataset_att, topK_from_senti_dist_dict
from litmodel import litDistillBERT


parser = argparse.ArgumentParser(
    description="Fine-tune DistilBERT to perform sentiment classification on IMDB review dataset with sentiment word masking."
)

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
    "-tv",
    "--threshold-value",
    type=float,
    default=0.05,
    help="Threshold absolute value to screen out high-attention tokens",
)
parser.add_argument(
    "-tp",
    "--threshold-proportion",
    type=float,
    default=None,
    help="Threshold proportions to screen out high-attention tokens",
)
parser.add_argument(
    "--device_ids",
    type=int,
    nargs="+",
    default=[0, 1, 2, 3],
    help="List of device IDs for multi-GPU training",
)
parser.add_argument(
    "--data_folder",
    type=str,
    default="../data/",
    help="Path to the data folder where imdb_reviews.csv, senti_dist_dict.pkl and att_mats_pooled.pt are stored",
)
parser.add_argument(
    "--ckpt_folder",
    type=str,
    default="../ckpt/ckpt_distilbert_att/",
    help="Path to the checkpoint folder where the trained model will be saved",
)


group = parser.add_mutually_exclusive_group()
group.add_argument("-t", "--train", action="store_true")
group.add_argument("-i", "--infer", action="store_true")

args = parser.parse_args()


# Set up file system and structure
data_dir = args.data_folder
imdb_data_path = os.path.join(data_dir, "imdb_reviews.csv")
senti_dist_path = os.path.join(data_dir, "senti_dist_dict.pkl")
att_mat_path = os.path.join(data_dir, "att_mats_pooled.pt")


ckpt_dir = args.ckpt_folder
threshold_value = args.threshold_value
threshold_proportion = args.threshold_proportion

device_list = args.device_ids

word_list_keyword = None  # None (no maksing), 'W_pos_id', 'W_neg_id', 'W_others_id', or None (no masking)
K = args.topk  # top K sentiment words will be used for masking
pretrained_model_name = "distilbert-base-uncased"  # cased tokenization ("BAD" conveys more than "bad") # bert-based-uncased, bert-large-cased, bert-large-uncased


word_list_keyword = args.mask_keyword
temp = "none" if word_list_keyword is None else word_list_keyword.split("_")[1]
ckpt_name = "-".join(
    [
        "imdb-distillbert",
        temp,
        f"top{K}",
        f"tv{100 * threshold_proportion:0>3n}",
        "{val_loss:.4f}",
    ]
)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
MAX_LEN = 512
SEED = 2023


if __name__ == "__main__":
    seed_everything(SEED, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # use most sentiment words (according to corresponding sentiment classes) as topping criteria
    senti_dist_dict = pickle.load(open(senti_dist_path, "rb"))
    word_lists = topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer)
    word_list_ids = (
        word_lists[word_list_keyword] if word_list_keyword is not None else None
    )

    print("Loading IMDB dataset ...")
    df = pd.read_csv(imdb_data_path)
    df["sentiment"] = df.sentiment.apply(lambda s: 0 if s == "negative" else 1)

    print("Loading pooled attention matrices from BERT...")
    att_mats_pooled = pickle.load(open(att_mat_path, "rb"))
    ds = IMDBdataset_att(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        att_mats=att_mats_pooled,
        word_list_ids=word_list_ids,
        threshold_val=threshold_value,
        threshold_prop=threshold_proportion,
    )

    ds_train, ds_val, ds_inf = torch.utils.data.random_split(ds, [40000, 5000, 5000])
    dl_train = DataLoader(
        ds_train, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False
    )
    dl_inf = DataLoader(
        ds_inf, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=2, output_attentions=True
    )
    litmodel = litDistillBERT(model=model, lr=LEARNING_RATE)

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="val_loss",
            dirpath=ckpt_dir,
            filename=ckpt_name,
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=device_list,
            strategy="ddp",
            precision="16",
            deterministic=True,
            callbacks=[checkpoint_callback],
            default_root_dir="./",
            logger=False,
        )

        trainer.fit(litmodel, dl_train, dl_val)

    if args.infer:
        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=[7],
            precision="16",
            deterministic=True,
        )

        senti_keyword = (
            "none" if word_list_keyword is None else word_list_keyword.split("_")[1]
        )
        for checkpoint_name in os.listdir(ckpt_dir):
            if (
                checkpoint_name.__contains__(senti_keyword)
                and checkpoint_name.__contains__(f"top{K}")
                and checkpoint_name.__contains__(f"tv{100 * threshold_proportion:0>3n}")
            ):
                break

        trainer.validate(
            model=litmodel,
            dataloaders=dl_inf,
            ckpt_path=ckpt_dir + checkpoint_name,
        )
