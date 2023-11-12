import torch
from torch.utils.data import Dataset, DataLoader

import os
from copy import deepcopy
import pickle


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
            "review_text": review,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(target, dtype=torch.long),
        }


class IMDBdataset_att(Dataset):
    """Prepare IMDB review dataset, replace sentiment words (along with its neighbours) with non-sense token such as 'x' or '[PAD]' based on ATTENTION"""

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

        # replace instead of removing
        if self.word_list_ids is not None:
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


class IMDBembedding(Dataset):
    """Prepare IMDB embedding dataset"""

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


def create_data_loader_imdb(df, tokenizer, max_len, batch_size):
    """Create dataloader from IMDB movie review dataset"""
    dataset = IMDBdataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)


def topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer):
    """Return the top K sentiment words based on contributions to sentiment classes. based on a freq dict generated based on imdb_get_att_mats.py"""
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


def load_embd_idx(save_path):
    """Load indices of embeddings that corresponding to training flow, training classifier and inference"""
    flow_idx = pickle.load(open(os.path.join(save_path, "flow_idx.pkl"), "rb"))
    cls_idx = pickle.load(open(os.path.join(save_path, "cls_idx.pkl"), "rb"))
    inf_idx = pickle.load(open(os.path.join(save_path, "inf_idx.pkl"), "rb"))
    return flow_idx, cls_idx, inf_idx


def process_distillbert_embedding(save_path):
    """Process distillbert embeddings."""
    predicts = []
    indices = []
    for f in os.listdir(save_path):
        if f.startswith("predictions"):
            predicts.append(torch.load(os.path.join(save_path, f)))
        if f.startswith("batch_indices"):
            indices.append(torch.load(os.path.join(save_path, f)))

    embeddings, targets, idxs = [], [], []
    for process_idx in range(len(predicts)):
        temp = predicts[process_idx]
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
