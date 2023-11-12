# NOT MODIFIED YET

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

from tqdm import tqdm
from copy import deepcopy
import pickle
import argparse
import os
import shutil

import sys
sys.path.insert(0, '/home/liu00980/Documents/mvdp/nlp_task/')
from utils_nlp import *


parser = argparse.ArgumentParser(description = 'Perform MC inference on collections of sentiment words using DPG')
parser.add_argument('-m', '--mask-keyword', type = str, default = "W_others_id", choices = [None, 'W_pos_id', 'W_neg_id', 'W_others_id'], help = 'Keyword for specifying the mask')
args = parser.parse_args()



WORKING_DIRECTORY = "/home/liu00980/Documents/mvdp/nlp_task/imdb_distillbert/"
DATA_PATH = "/home/liu00980/Documents/mvdp/nlp_task/data/imdb/"
PLOT_PATH = "/home/liu00980/Documents/mvdp/nlp_task/imdb_distillbert/pyplot.png"
IDX_SAVE_PATH = '/home/liu00980/Documents/mvdp/nlp_task/data/imdb/ebd_split_idx/'
CHECKPOINT_FOLDER = "/home/liu00980/Documents/mvdp/nlp_task/checkpoints_clshead/"
FLOW_CHECKPOINT_FOLDER = "/home/liu00980/Documents/mvdp/nlp_task/checkpoints_affineflow/"

WORD_LIST_KEYWORD = args.mask_keyword # None (no maksing), 'W_pos_id', 'W_neg_id', 'W_others_id', or None (no masking)
K = 600 # top K sentiment words will be used for masking
THRESHOLD_PROPORTION = 0.02 # based on attention weights distribution, this proportion of weights (from above) will be eliminated

senti_word = "none" if WORD_LIST_KEYWORD is None else WORD_LIST_KEYWORD.split('_')[1]
temp2 = f'tv{0:0>3n}' if THRESHOLD_PROPORTION is None else f'tv{100 * THRESHOLD_PROPORTION:0>3n}'
EMBEDDING_DIMENSION = 768
EMBEDDING_FOLDER = '-'.join(['imdb-distillbert', senti_word, f'top{K}', temp2])
EMBEDDING_FOLDER_NONE = 'imdb-distillbert-none-top600-tv002'
CHECKPOINT_NAME = '-'.join(['imdb-distillbert', senti_word, f'top{K}', temp2])

for FLOW_CHECKPOINT in os.listdir(FLOW_CHECKPOINT_FOLDER):
    if FLOW_CHECKPOINT.startswith(CHECKPOINT_NAME):
        break


MAX_LEN = 512
SEED = 2023


BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 5e-5

flow_idx, cls_idx, inf_idx = load_embd_idx(IDX_SAVE_PATH)
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)
DEVICES = [0]   # currently use one GPU for getting the result

class ClassificationHead(nn.Module):
    def __init__(self, input_dim = 768, n_classes = 2, hidden_units = [1024, 512], dropout_rate = 0.3):
        super().__init__()
        layers = []
        for output_dim in hidden_units:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout_rate))
            input_dim = output_dim
        
        layers.append(nn.Linear(input_dim, n_classes))
        layers.append(nn.LogSoftmax(dim = 1))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, batch):
        for layer in self.layers:
            batch = layer(batch)
        return batch



class litCLSHead(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        embeddings, targets = batch
        outputs = self.model(embeddings)
        loss = F.nll_loss(outputs, targets)
        
        preds = outputs.argmax(dim = 1, keepdim = True)
        acc = preds.eq(targets.view_as(preds)).sum().item() / len(preds)
        self.log('train_loss', loss, on_epoch = True, on_step = False, prog_bar = True)
        self.log('train_acc', acc, on_epoch = True, on_step = False, prog_bar = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        embeddings, targets = batch
        outputs = self.model(embeddings)
        loss = F.nll_loss(outputs, targets)
        
        preds = outputs.argmax(dim = 1, keepdim = True)
        correct_preds = preds.eq(targets.view_as(preds)).sum().item()
        acc = correct_preds / len(preds)
        self.log('val_loss', loss, on_epoch = True, on_step = False, prog_bar = True)
        self.log('val_acc', acc, on_epoch = True, on_step = False, prog_bar = True)
    
    def predict_step(self, batch, batch_idx):
        embeddings, targets = batch
        outputs = self.model(embeddings)
        losses = F.nll_loss(outputs, targets, reduction = "none") # get the nll losses for each sample
        return losses
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer



# get cls and inf dataloaders from a full-size embeddings-targets (50,000), based on pre-specified split indices
def get_cls_inf_dls(embeddings, targets):
    ds = IMDBembedding(embeddings = embeddings, sentiments = targets, onehot = False)    
    ds_cls, ds_inf = Subset(ds, cls_idx), Subset(ds, inf_idx)
    
    dl_cls = DataLoader(ds_cls, batch_size = BATCH_SIZE, pin_memory = True, shuffle = True)
    dl_inf = DataLoader(ds_inf, batch_size = n_inf, pin_memory = True, shuffle = False)
    return dl_cls, dl_inf


# sample a cls dl of size cls_size and a inf dl of size inf_size from a trained flow model
def sample_cls_inf_dls_from_flow(model, cls_size, inf_size):
    total_size = cls_size + inf_size
    embeddings, targets = model.sample_joint(sample_size = total_size)
    ds = IMDBembedding(embeddings = embeddings, sentiments = targets, onehot = False)    
    ds_cls, ds_inf = random_split(ds, [cls_size, inf_size])
    
    dl_cls = DataLoader(ds_cls, batch_size = BATCH_SIZE, pin_memory = True, shuffle = True)
    dl_inf = DataLoader(ds_inf, batch_size = n_inf, pin_memory = True, shuffle = False)
    return dl_cls, dl_inf


# get CE loss on each instance of inference sample, based on classifier trained on cls sample
# 1. both embeddings and targets will be splitted correspondingly based on global varialbes flow_idx, cls_idx, inf_idx
# 2. classification heads will be trained on cls_sample, and test statistic will be calculated on inf_sample
def get_nll_losses(dl_cls, dl_inf):
    model = ClassificationHead(
        input_dim = EMBEDDING_DIMENSION,
        n_classes = 2,
        hidden_units = [1024, 512],
        dropout_rate = 0.3
    )
    litmodel = litCLSHead(model, LEARNING_RATE)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(CHECKPOINT_FOLDER, senti_word),
        filename = 'checkpointname-{val_loss:.4f}.ckpt',
        save_top_k = 1,
        mode = 'min',
        monitor = 'val_loss'
    )
    
    trainer = pl.Trainer(
        max_epochs = EPOCHS, 
        accelerator = 'gpu',
        devices = DEVICES,
        callbacks = [checkpoint_callback],
        logger = False
    )
    
    trainer.fit(litmodel, dl_cls, dl_inf)
    nll_losses = trainer.predict(litmodel, dl_inf)
    
    shutil.rmtree(CHECKPOINT_FOLDER) # remove the checkpoint folder to reduce memory and avoid future confusion
    return nll_losses[0]


# get test statistic based on alternative data (1) and null data (0)
# 1. if test statistic is significantly > 0, it means that we should reject the null, meaning that colelction of sentiment matters signifcantly
def get_test_stat(dl_cls0, dl_inf0, dl_cls1, dl_inf1, inf_size):
    losses_alternative = get_nll_losses(dl_cls1, dl_inf1)
    losses_none = get_nll_losses(dl_cls0, dl_inf0)
    diff = losses_alternative - losses_none
    test_stat = np.sqrt(inf_size) * diff.mean() / diff.std()
    return test_stat








seed_everything(SEED, workers = True)

# get the test statistics from the original sample/embeddings
embeddings_pos, targets_pos, _ = process_distillbert_embedding(DATA_PATH + EMBEDDING_FOLDER)
dl_cls_pos, dl_inf_pos = get_cls_inf_dls(embeddings_pos, targets_pos)

embeddings_none, targets_none, _ = process_distillbert_embedding(DATA_PATH + EMBEDDING_FOLDER_NONE)
dl_cls_none, dl_inf_none = get_cls_inf_dls(embeddings_none, targets_none)

test_stat = get_test_stat(dl_cls_none, dl_inf_none, dl_cls_pos, dl_inf_pos, n_inf)
test_stat



# get the null distribution of the test statistics
nfmodel = NF_affine_cond(EMBEDDING_DIMENSION, 2, 10, [8192, 4096, 2048])
lit_nfmodel = litNFAffine()
lit_nfmodel = lit_nfmodel.load_from_checkpoint(os.path.join(FLOW_CHECKPOINT_FOLDER, FLOW_CHECKPOINT), model = nfmodel)


lit_nfmodel.eval()


null_test_stat_list = []
B = 200

for _ in tqdm(range(B), leave = False, dynamic_ncols = True):
    dl_cls_0, dl_inf_0 = sample_cls_inf_dls_from_flow(nfmodel, n_cls, n_inf)
    dl_cls_1, dl_inf_1 = sample_cls_inf_dls_from_flow(nfmodel, n_cls, n_inf)
    null_test_stat = get_test_stat(dl_cls_0, dl_inf_0, dl_cls_1, dl_inf_1, n_inf)
    null_test_stat_list.append(null_test_stat)
    pickle.dump(null_test_stat_list, open(os.path.join(DATA_PATH, 'null_dist', senti_word + '.pkl'), "wb"))



null_array = np.array(null_test_stat_list)

np.mean(null_array >= test_stat.item()) # 0.045 for positive collection, 0.015 for negative, 0.715 for others.


# (Guideline) check results for pos, neg, others first, then think about acceleration (sample via GPU(s), DDP for replications, still train cls on single GPU [smaller network])
# Questions to be solved:
# (maybe we shouldn't apply to different pseudo inference data? need to think about the framework, hypothesis, algorithm REALLY CAREFULLY!)
# 1. when training classification heads on generated samples, the results are not good (acc is about 88, loss is about 0.3, and may continue to improve)
# (cont. it seems that 50 epochs help. 0.95 acc and good loss! so the trained flow seems good! need to continue on fine tuning! lr_scheduler! more epochs!)
# 2. implement sample on GPU with batch operation for speed? (might be a good idea to implement it in LightningModule with self.device, just saying)
# 3. think about the algorithm twice (I guess not significant if use the two (conditionally) independent cls loaders and inf loaders)
# 4. distributed through multiple GPUs for repetitions (to use, for example, total B = 2,000)




# visualization of comparing these three null distributions

pos = pickle.load(open(os.path.join(DATA_PATH, 'null_dist', 'pos' + '.pkl'), "rb"))
neg = pickle.load(open(os.path.join(DATA_PATH, 'null_dist', 'neg' + '.pkl'), "rb"))
others = pickle.load(open(os.path.join(DATA_PATH, 'null_dist', 'others' + '.pkl'), "rb"))

pos, neg, others = np.array(pos), np.array(neg), np.array(others)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

subtitles = ['Positive', 'Negative', 'Neutral']
subcolors = ['C0', 'C1', 'C2']
subpoints = [2.5994, 3.7996, -1.1296] # test statistic values, used for marking points
subpvalues = [0.045, 0.015, 0.715]

# Set up plot layout
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

# Plot histograms and kernel density curves for each distribution
for i, (dist, ax) in enumerate(zip([pos, neg, others], axs)):
        
    # Compute histogram
    counts, bins, _ = ax.hist(dist, bins = 15, density = True, alpha = 0.7, color = subcolors[i])
    
    # Fit kernel density estimation
    kde = KernelDensity(bandwidth = 0.4).fit(dist[:, np.newaxis])
    x = np.linspace(min(dist), max(dist), 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x)
    ax.plot(x[:, 0], np.exp(log_dens), '-', linewidth = 1.5, color = 'black', alpha = 0.7, label = "KDE")
    
    # Add mean and standard deviation to plot
    mean = np.mean(dist)
    std = np.std(dist)
    ax.axvline(mean, color = 'black', linestyle = 'dashed', linewidth = 1, label = 'Mean')
    ax.axvline(mean + std, color = 'black', linestyle = 'dotted', linewidth = 1, label = 'Std Dev')
    ax.axvline(mean - std, color = 'black', linestyle = 'dotted', linewidth = 1)
    
    # Add an 'x' marker
    ax.scatter(x = subpoints[i], y = 0, s = 100, marker = 'x', color = 'C3', zorder=10, clip_on=False, label = 'Test statistic')
    # ax.annotate('x', xy = (subpoints[i], 0), ha='center', va='bottom', color='red', label = "Test statistic")
    
    # Add title and axis labels
    ax.set_title(f'({subtitles[i]}) p value = {subpvalues[i]:.3f}')
    ax.set_xlabel('Test statistic')
    ax.set_ylabel('Density')


handles, labels = ax.get_legend_handles_labels()

# # Add legend to last plot
# axs[2].legend(['Kernel Density Estimate', 'Mean', 'Std Dev'])
fig.suptitle('Comparison of Three Null Distributions', fontsize=16, fontweight='bold', x = 0.25)
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=[0.75, 0.905])

# # Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(PLOT_PATH)





# KS test against standard normal distribution
import scipy.stats as stats

# Calculate means and std devs
for sample_data in [pos, neg, others]:
    print('Mean:', sample_data.mean())
    print('Std Dev:', sample_data.std())



# Generate some sample data
for sample_data in [pos, neg, others]:
    # Perform the KS test against a standard normal distribution
    ks_statistic, p_value = stats.kstest(sample_data, 'norm', method = 'exact')
    # Print the results
    print('KS statistic:', ks_statistic)
    print('p-value:', p_value)


