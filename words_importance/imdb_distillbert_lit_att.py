# Use attention weights to threshold/screen out the context

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from tqdm import tqdm
from copy import deepcopy
import pickle
import argparse
import os


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

group = parser.add_mutually_exclusive_group()
group.add_argument("-t", "--train", action="store_true")
group.add_argument("-i", "--infer", action="store_true")

args = parser.parse_args()


WORKING_DIRECTORY = "/home/liu00980/Documents/mvdp/nlp_task/imdb_distillbert/"
DATA_PATH = "/home/liu00980/Documents/mvdp/nlp_task/data/imdb/"
PLOT_PATH = "/home/liu00980/Documents/mvdp/nlp_task/pyplot.png"
CHECKPOINT_FOLDER = (
    "/home/liu00980/Documents/mvdp/nlp_task/checkpoints_distillbert_att/"
)
WORD_LIST_SAVE_PATH = "/home/liu00980/Documents/mvdp/nlp_task/data/imdb/"
THRESHOLD_VALUE = (
    args.threshold_value
)  # absolute threshold value above which the attention weights will be eliminated
THRESHOLD_PROPORTION = (
    args.threshold_proportion
)  # based on attention weights distribution, this proportion of weights (from above) will be eliminated


WORD_LIST_KEYWORD = None  # None (no maksing), 'W_pos_id', 'W_neg_id', 'W_others_id', or None (no masking)
K = args.topk  # top K sentiment words will be used for masking
PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"  # cased tokenization ("BAD" conveys more than "bad") # bert-based-uncased, bert-large-cased, bert-large-uncased


WORD_LIST_KEYWORD = args.mask_keyword
temp = "none" if WORD_LIST_KEYWORD is None else WORD_LIST_KEYWORD.split("_")[1]
CHECKPOINT_NAME = "-".join(
    [
        "imdb-distillbert",
        temp,
        f"top{K}",
        f"tv{100 * THRESHOLD_PROPORTION:0>3n}",
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

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # use most sentiment words (according to corresponding sentiment classes) as topping criteria
    senti_dist_dict = pickle.load(
        open(WORD_LIST_SAVE_PATH + "senti_dist_dict.pkl", "rb")
    )
    word_lists = topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer)
    word_list_ids = (
        word_lists[WORD_LIST_KEYWORD] if WORD_LIST_KEYWORD is not None else None
    )

    df = pd.read_csv(DATA_PATH + "imdb_reviews.csv")
    df["sentiment"] = df.sentiment.apply(lambda s: 0 if s == "negative" else 1)

    att_mats_pooled = pickle.load(
        open(DATA_PATH + "/att_mats_pooled.pkl", "rb")
    )  # took some time to load, too large
    ds = IMDBdataset_att(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        att_mats=att_mats_pooled,
        word_list_ids=word_list_ids,
        threshold_val=THRESHOLD_VALUE,
        threshold_prop=THRESHOLD_PROPORTION,
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
        PRE_TRAINED_MODEL_NAME, num_labels=2, output_attentions=True
    )
    litmodel = litDistillBERT(model=model, lr=LEARNING_RATE)

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="val_loss",
            dirpath=CHECKPOINT_FOLDER,
            filename=CHECKPOINT_NAME,
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=[3, 5, 6, 7],
            strategy="ddp",
            precision="16",
            deterministic=True,
            callbacks=[checkpoint_callback],
            default_root_dir=WORKING_DIRECTORY,
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
            "none" if WORD_LIST_KEYWORD is None else WORD_LIST_KEYWORD.split("_")[1]
        )
        for checkpoint_name in os.listdir(CHECKPOINT_FOLDER):
            if (
                checkpoint_name.__contains__(senti_keyword)
                and checkpoint_name.__contains__(f"top{K}")
                and checkpoint_name.__contains__(f"tv{100 * THRESHOLD_PROPORTION:0>3n}")
            ):
                break

        trainer.validate(
            model=litmodel,
            dataloaders=dl_inf,
            ckpt_path=CHECKPOINT_FOLDER + checkpoint_name,
        )


# # get some example words from the sentiment collections
# temp_keyword = 'W_others_id'
# temp_start = 50
# temp_end = temp_start + 20
# tokens = tokenizer.convert_ids_to_tokens(word_lists[temp_keyword])
# ", ".join(tokens[temp_start:temp_end])
