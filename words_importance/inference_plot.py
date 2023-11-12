# NOT MODIFIED YET

# Inference, generate fake embeddings, compare with true embeddings

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, BasePredictionWriter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd


import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN


from tqdm import tqdm
from copy import deepcopy
import pickle
import argparse
import os

import sys
sys.path.insert(0, '/home/liu00980/Documents/mvdp/nlp_task/')
from utils_nlp import *


parser = argparse.ArgumentParser(description = 'Fine-tune BERT to perform sentiment classification on IMDB review dataset with sentiment word masking.')
parser.add_argument('-m', '--mask-keyword', type = str, default = 'W_pos_id', choices = [None, 'W_pos_id', 'W_neg_id', 'W_others_id'], help = 'Keyword for specifying the mask')

args = parser.parse_args()



WORKING_DIRECTORY = "/home/liu00980/Documents/mvdp/nlp_task/imdb_distillbert/"
DATA_PATH = "/home/liu00980/Documents/mvdp/nlp_task/data/imdb/"
PLOT_FOLDER = "/home/liu00980/Documents/mvdp/nlp_task/imdb_distillbert/"
CHECKPOINT_FOLDER = '/home/liu00980/Documents/mvdp/nlp_task/checkpoints_distillbert_att/'
FLOW_CHECKPOINT_FOLDER = "/home/liu00980/Documents/mvdp/nlp_task/checkpoints_affineflow/"
WORD_LIST_SAVE_PATH = "/home/liu00980/Documents/mvdp/nlp_task/data/imdb/"
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased' # cased tokenization ("BAD" conveys more than "bad") # bert-based-uncased, bert-large-cased, bert-large-uncased
IDX_SAVE_PATH = '/home/liu00980/Documents/mvdp/nlp_task/data/imdb/ebd_split_idx/'


WORD_LIST_KEYWORD = args.mask_keyword # None (no maksing), 'W_pos_id', 'W_neg_id', 'W_others_id', or None (no masking)
K = 600 # top K sentiment words will be used for masking
THRESHOLD_PROPORTION = 0.02 # based on attention weights distribution, this proportion of weights (from above) will be eliminated


temp = "none" if WORD_LIST_KEYWORD is None else WORD_LIST_KEYWORD.split('_')[1]
temp2 = f'tv{0:0>3n}' if THRESHOLD_PROPORTION is None else f'tv{100 * THRESHOLD_PROPORTION:0>3n}'
EMBEDDING_FOLDER = '-'.join(['imdb-distillbert', temp, f'top{K}', temp2])
CHECKPOINT_NAME = '-'.join(['imdb-distillbert', temp, f'top{K}', temp2])

for file in os.listdir(FLOW_CHECKPOINT_FOLDER):
    if file.startswith(CHECKPOINT_NAME):
        break

flow_infer_checkpoint_path = os.path.join(FLOW_CHECKPOINT_FOLDER, file)


MAX_LEN = 512
SEED = 2023


flow_idx, cls_idx, inf_idx = load_embd_idx(IDX_SAVE_PATH)
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)


seed_everything(SEED, workers = True)

# 1. choose a checkpoint (the 600-002 seems like a decent choice) and get embeddings
# Load the embeddings if it already exists. Otherwise, run a forward pass and get the embeddings.
if not os.path.isdir(DATA_PATH + EMBEDDING_FOLDER):
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    # use most sentiment words (according to corresponding sentiment classes) as topping criteria
    senti_dist_dict = pickle.load(open(WORD_LIST_SAVE_PATH + 'senti_dist_dict.pkl', "rb"))
    word_lists = topK_from_senti_dist_dict(senti_dist_dict, K, tokenizer)
    word_list_ids = word_lists[WORD_LIST_KEYWORD] if WORD_LIST_KEYWORD is not None else None
    
    df = pd.read_csv(DATA_PATH + 'imdb_reviews.csv')
    df["sentiment"] = df.sentiment.apply(lambda s: 0 if s == 'negative' else 1)
    
    att_mats_pooled = pickle.load(open(DATA_PATH + "/att_mats_pooled.pkl", "rb")) if WORD_LIST_KEYWORD is not None else None
    
    ds = IMDBdataset_att(
        reviews = df.review.to_numpy(),
        targets = df.sentiment.to_numpy(),
        tokenizer = tokenizer,
        max_len = MAX_LEN,
        att_mats = att_mats_pooled,
        word_list_ids = word_list_ids,
        threshold_val = 0,
        threshold_prop = THRESHOLD_PROPORTION
        )
    
    dl = DataLoader(ds, batch_size = 16, num_workers = 4, pin_memory = True, shuffle = False)
    
    class DistillBertEmd(litDistillBERT):
        def __init__(lit_nfmodel, model = None, lr = 0.001):
            super().__init__(model, lr)
        
        def predict_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_masks = batch['attention_mask']
            targets = batch['targets']
            all_embeddings = self.model.distilbert(
                input_ids = input_ids,
                attention_mask = attention_masks
            )['last_hidden_state']
            
            # embedding of the [CLS] token
            return all_embeddings[:, 0, :], targets
    
    class CustomWriter(BasePredictionWriter):
        def __init__(self, output_dir, write_interval):
            super().__init__(write_interval)
            
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            
            self.output_dir = output_dir
        
        def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
            torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
            torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))
    
    pred_writer = CustomWriter(output_dir = DATA_PATH + EMBEDDING_FOLDER , write_interval = "epoch")
    
    trainer = Trainer(
        accelerator = 'gpu',
        devices = [5, 7],           # [1, 2, 3, 5]
        strategy = 'ddp_notebook',
        precision = "16",
        callbacks = [pred_writer],
        deterministic = True,
        logger = False
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2, output_attentions = True)
    litmodel = DistillBertEmd(model = model)
    
    # Get the checkpoint model for evaluating masked embeddings
    if WORD_LIST_KEYWORD is None:
        checkpoint_name = 'imdb-distillbert-none-val_loss=0.1878.ckpt'
    else:
        for checkpoint_name in os.listdir(CHECKPOINT_FOLDER):
            if checkpoint_name.startswith(CHECKPOINT_NAME):
                break
    
    trainer.predict(model = litmodel, dataloaders = dl, ckpt_path = CHECKPOINT_FOLDER + checkpoint_name, return_predictions = False)

embeddings, targets, idxs = process_distillbert_embedding(DATA_PATH + EMBEDDING_FOLDER)



# issues: nan entries for negative embeddings
# e = np.array(embeddings)
# np.argwhere(np.isnan(e)) # row 14460 (of dim 768) is all null
# e[14460]
# targets[14460]
# idxs[14460]  # data entry 28291 has issues
if temp == 'neg':
    col_mean = torch.cat((embeddings[:14460, :], embeddings[14461:, :])).mean(dim = 0)
    embeddings[14460, :] = col_mean




# 2. design an affine flow
# (still have issues with trianing spline flows: large loss, striped learned distribution, cuda out-of-memory error)
# HOWEVER, the trained affine flow seems to be good. Will use it for now

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 7e-5
TFPLOT_LOG_EPOCH_INTERVALS = 10


ds = IMDBembedding(embeddings = embeddings, sentiments = targets, onehot = True)

ds_flow = Subset(ds, flow_idx)
dl_flow = DataLoader(ds_flow, batch_size = BATCH_SIZE, pin_memory = True, shuffle = True)

TFPLOT_NSAMPLES = 3000
PAIRWISE_LIST = [(494, 253), (345, 601)]
kwargs = {
    "model": NF_affine_cond(embeddings.shape[1], 2, 10, [8192, 4096, 2048]),
    "embeddings": embeddings,
    "targets": targets,
    "pairwise_list": PAIRWISE_LIST,
    "lr": LEARNING_RATE,
    "plot_samples": TFPLOT_NSAMPLES,
}


lit_nfmodel = litNFAffine()
lit_nfmodel = lit_nfmodel.load_from_checkpoint(flow_infer_checkpoint_path, **kwargs)


lit_nfmodel.eval()		# put model to evaluation mode



# generate plots for comparing distributions
idx = np.random.choice(range(len(lit_nfmodel.embeddings)), size = lit_nfmodel.plot_samples, replace = False)
true_embeddings = lit_nfmodel.embeddings[idx, :]
true_targets = lit_nfmodel.targets[idx]
true_embeddings_pos = true_embeddings[true_targets == 1]
true_embeddings_neg = true_embeddings[true_targets == 0]

fake_embeddings_pos = lit_nfmodel.model.target_dist_giveny.condition(
    torch.tensor([[1, 0] for _ in range(len(true_embeddings_pos))], dtype = torch.float, device = lit_nfmodel.device)
).sample([len(true_embeddings_pos)]).cpu()
fake_embeddings_neg = lit_nfmodel.model.target_dist_giveny.condition(
    torch.tensor([[0, 1] for _ in range(len(true_embeddings_neg))], dtype = torch.float, device = lit_nfmodel.device)
).sample([len(true_embeddings_neg)]).cpu()


fake_embeddings = torch.cat((fake_embeddings_pos, fake_embeddings_neg))
random_idxs = np.random.choice(range(len(fake_embeddings)), size = len(fake_embeddings_pos), replace = False)


fig = pairwise_bertembeddings(
    bert_embeddings = true_embeddings[random_idxs, :],
    pairlist = lit_nfmodel.pairwise_list,
    nrows = 2,
    ncols = 1,
    fake_embeddings = fake_embeddings[random_idxs, :],
    kde = True
)


fig_pos = pairwise_bertembeddings(
    bert_embeddings = true_embeddings_pos,
    pairlist = lit_nfmodel.pairwise_list,
    nrows = 2,
    ncols = 1,
    fake_embeddings = fake_embeddings_pos
)          

fig_neg = pairwise_bertembeddings(
    bert_embeddings = true_embeddings_neg,
    pairlist = lit_nfmodel.pairwise_list,
    nrows = 2,
    ncols = 1,
    fake_embeddings = fake_embeddings_neg
)

for i in range(len(fig.get_axes())):
    overall_xlim, overall_ylim = fig.get_axes()[i].get_xlim(), fig.get_axes()[i].get_ylim()
    fig_pos.get_axes()[i].set_xlim(overall_xlim)
    fig_pos.get_axes()[i].set_ylim(overall_ylim)
    fig_neg.get_axes()[i].set_xlim(overall_xlim)
    fig_neg.get_axes()[i].set_ylim(overall_ylim)

fig.suptitle("Unconditional DistilBERT embeddings \n(Positive & Negative)", fontsize = 27, weight = 'bold')
fig_pos.suptitle("Conditional embeddings (Positive)", fontsize = 27, weight = 'bold')
fig_neg.suptitle("Conditional embeddings (Negative)", fontsize = 27, weight = 'bold')

fig.savefig(os.path.join(PLOT_FOLDER, "true_fake_samples.png"))
fig_pos.savefig(os.path.join(PLOT_FOLDER, "true_fake_samples_pos.png"))
fig_neg.savefig(os.path.join(PLOT_FOLDER, "true_fake_samples_neg.png"))



