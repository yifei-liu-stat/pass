"""
Prepare IMDB review dataset: get the dataset, sentiment words list and attention weights from pretrained BERT model.
- data will be downloaded to `./data/imdb_reviews.csv`
- sentiment words list (ranked by appearance frequency) will be saved to `./data/freq_lists.pkl`
- sentiment words list (with sentiment distribution) will be saved to `./data/senti_dist_dict.pkl`
- pooled attention matrices will be saved to `./data/att_mats_pooled.pt`
"""
import torch
from transformers import BertModel, BertTokenizer

import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import opinion_lexicon

from collections import Counter

import pickle
import string
import copy

import os
import gdown


from utils.data import create_data_loader_imdb


data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# 1. Download IMDB review dataset
imdb_data_path = os.path.join(data_dir, "imdb_reviews.csv")
if not os.path.exists(imdb_data_path):
    print("Downloading IMDB review dataset ...")

    url = "https://drive.google.com/file/d/188kXbMMJdP26A4_b_aH_iXHNoB45DZmW/view?usp=drive_link"
    gdown.download(url, quiet=False, output=imdb_data_path, fuzzy=True)

    print("Download finished.")
else:
    print(f"IMDB review dataset already exists in path {imdb_data_path}.")


df = pd.read_csv(imdb_data_path)
df["sentiment"] = df.sentiment.apply(
    lambda s: 0 if s == "negative" else 1
)  # this is series.apply() method

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 2. Prepare sentiment words list
freq_list_path = os.path.join(data_dir, "freq_lists.pkl")
if not os.path.exists(freq_list_path):
    print("Preparing sentiment words list ...")

    ## positive words and negative words from opinion lexicon
    pos_words = opinion_lexicon.positive()
    neg_words = opinion_lexicon.negative()

    ## english stop words set, provided by https://github.com/stopwords-iso/stopwords-en/tree/master
    stop_words_set = set([])
    with open("./data/stopwords-en.txt") as file:
        for line in file:
            stop_words_set.add(line.rstrip())

    # 2'. Prepare sentimet words list, ranked by appearance frequency in imdb review dataset
    print(
        "Preparing sentiment words list, sorted by appearance frequency in imdb review dataset..."
    )

    ## counts of each tokenized review and union them all together
    freq_tokens = Counter()
    for review in tqdm(df.review):
        tokenized_review = tokenizer.tokenize(review)
        freq_tokens += Counter(tokenized_review)

    ## cross check with pos/neg words
    pos_words_set, neg_words_set = set(pos_words), set(neg_words)
    pos_freq = {
        token: cnts for token, cnts in freq_tokens.items() if token in pos_words_set
    }
    neg_freq = {
        token: cnts for token, cnts in freq_tokens.items() if token in neg_words_set
    }

    ## criteria for tokens to be in this others_freq (neutral word list)
    # 1. neither in pos_freq nor neg_freq
    # 2. not include stopwords (too frequent such as 'a', 'an', 'another', 'and'...)
    # 3. not include punctuations
    # 4. not include special tokens such as [UNK], [PAD], [CLS], [SEP] and etc.
    # 5. not include stuff such as '#...' (subsequent words)
    # 6. should be either adjective (JJ), adverb (RB) or verb (VB): nouns are not important
    def neutral_rule(token):
        return (
            False
            if token in pos_words_set.union(neg_words_set)
            or token in stop_words_set
            or token in string.punctuation
            or token in ["[UNK]", "[PAD]", "[SEP]", "[CLS]"]
            or token.startswith("#")
            or not nltk.tag.pos_tag([token])[0][1].startswith(tuple(["JJ", "RB", "VB"]))
            else True
        )

    others_freq = {
        token: cnts for token, cnts in freq_tokens.items() if neutral_rule(token)
    }

    pos_freq, neg_freq, others_freq = (
        Counter(pos_freq),
        Counter(neg_freq),
        Counter(others_freq),
    )

    ## frequency of sentiment (pos/neg/others) words appeared in the inquired dataset
    print("Saving sentiment words list (sort by appearance frequency) ...")
    freq_dict = {"pos_freq": pos_freq, "neg_freq": neg_freq, "others_freq": others_freq}
    pickle.dump(freq_dict, open(freq_list_path, "wb"))

else:
    print(f"Sentiment words freq list already exists in path {freq_list_path}.")


# 2''. Prepare sentimet words list, creating dictionaries of sentiment class distribution for each sentiment word
senti_dist_path = os.path.join(data_dir, "senti_dist_dict.pkl")
if not os.path.exists(senti_dist_path):
    print(
        "Preparing sentiment words list, with meta information of sentiment distribution."
    )

    freq_dict = pickle.load(open(freq_list_path, "rb"))

    senti_dist_dict = {}
    for keyword in freq_dict.keys():  # 'pos_freq', 'neg_freq' and 'others_freq'
        senti_set = set(freq_dict[keyword].keys())
        senti_dist = {senti_word: [0, 0] for senti_word in senti_set}

        print(f"Calculating sentiment distribution for {keyword} ...")
        for idx in tqdm(range(len(df))):
            input_ids = tokenizer.encode_plus(df.review[idx], add_special_tokens=True)[
                "input_ids"
            ]
            input_tokens_set = set(tokenizer.convert_ids_to_tokens(input_ids))
            sentiment_id = df.sentiment[idx]
            target_set = input_tokens_set.intersection(senti_set)
            for senti_word in target_set:
                senti_dist[senti_word][sentiment_id] += 1

        for k, v in senti_dist.items():
            total = sum(v)
            v[0] = round(v[0] / total, 4)
            v[1] = 1 - v[0]
            v.append(total)

        senti_dist_dict[keyword] = copy.deepcopy(senti_dist)

    print("Saving sentiment words list (with sentiment distribution) ...")
    pickle.dump(senti_dist_dict, open(senti_dist_path, "wb"))
else:
    print(
        f"Sentiment words list with sentiment distribution already exists in {senti_dist_path}"
    )

# 3. Get attention matrices from pretrained BERT model

att_mat_path = os.path.join(data_dir, "att_mats_pooled.pt")


if not os.path.exists(att_mat_path):
    print("Preparing attention matrices from pretrained BERT model ...")
    device = "cuda:0"

    # to output attentions: output_bert = bert_model(input_ids = ..., attention_mask = ...), output_bert.attentions (batchsize, nheads, n, n)
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

    dl = create_data_loader_imdb(df, tokenizer, max_len=512, batch_size=16)
    bert_model = bert_model.to(device)
    bert_model.eval()

    att_mats = []
    for d in tqdm(dl):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        with torch.no_grad():
            bert_outputs = bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            att_mats_batch = (
                bert_outputs.attentions[0].mean(1).cpu()
            )  # 1st layer / 12 layers, average over 12 heads

        att_mats.append(att_mats_batch)

    att_mats = torch.cat(tuple(att_mats), 0)

    print("Saving pooled attention matrices ...")
    pickle.dump(att_mats, open(att_mat_path, "wb"))
else:
    print(f"Pooled attention matrices already exist in {att_mat_path}.")
