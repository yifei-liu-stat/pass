"""
Evaluations of the inference, including:
- True vs. fake embeddings -> results/plots/true_fake_samples*.png
- Visualize null distributions -> results/plots/null_comparison.png
- KS test against standard normal distribution -> results will be printed out
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy.stats as stats



import torch
from torch.utils.data import (
    DataLoader,
    Subset,
)
from pytorch_lightning import seed_everything


import argparse
import os
import pickle

from utils.data import (
    load_embd_idx,
    process_distillbert_embedding,
    IMDBembedding,
)

from utils.litmodel import (
    NF_affine_cond,
    litNFAffine,
)

from utils.visualization import pairwise_bertembeddings




parser = argparse.ArgumentParser(description = 'Perform MC inference on collections of sentiment words using PAI')
parser.add_argument(
    "-m",
    "--mask-keyword",
    type=str,
    default="W_pos_id",
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

data_dir = "./data"
nulldist_dir = "./results/null_dist"
if not os.path.exists(nulldist_dir):
    os.makedirs(nulldist_dir)

plot_dir = "./results/plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

imdb_data_path = os.path.join(data_dir, "imdb_reviews.csv")
senti_dist_path = os.path.join(data_dir, "senti_dist_dict.pkl")
att_mat_path = os.path.join(data_dir, "att_mats_pooled.pt")


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
for ckptname_flow in os.listdir(flow_ckpt_dir):
    if ckptname_flow.startswith(ckpt_prefix):
        break
ckpt_flow = os.path.join(flow_ckpt_dir, ckptname_flow)


flow_idx, cls_idx, inf_idx = load_embd_idx(os.path.join(data_dir, "ebd_split_idx"))
n_flow, n_cls, n_inf = len(flow_idx), len(cls_idx), len(inf_idx)


SEED = 2023
seed_everything(SEED, workers = True)


print(f"Load true IMDB embeddings with {senti_word} masking ...")
embeddings, targets, idxs = process_distillbert_embedding(imdb_ebd_dir)

ds = IMDBembedding(embeddings = embeddings, sentiments = targets, onehot = True)

ds_flow = Subset(ds, flow_idx)
dl_flow = DataLoader(ds_flow, batch_size = 64, pin_memory = True, shuffle = True)

kwargs = {
    "model": NF_affine_cond(768, 2, 10, [8192, 4096, 2048]),
    "embeddings": embeddings,
    "targets": targets,
    "pairwise_list": [(494, 253), (345, 601)],
    "lr": 5e-5,
    "plot_samples": 3000,
}



print(f"Loading normalization flow checkpoint {ckpt_flow} ...")
lit_nfmodel = litNFAffine.load_from_checkpoint(ckpt_flow, **kwargs)


lit_nfmodel.eval()		# put model to evaluation mode






# 1. True vs. fake embeddings

print(f"Generating true and fake embeddings on coordinate pairs {kwargs['pairwise_list']} ...")
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

fig.savefig(os.path.join(plot_dir, "true_fake_samples.png"))
fig_pos.savefig(os.path.join(plot_dir, "true_fake_samples_pos.png"))
fig_neg.savefig(os.path.join(plot_dir, "true_fake_samples_neg.png"))

print(f"Visualization of the comparison between true and fake embeddings saved to {plot_dir} ...")









# 2. visualization of the null distributions
print(f"Loading null distributions from {nulldist_dir} ...")

pos = pickle.load(open(os.path.join(nulldist_dir, 'pos' + '.pkl'), "rb"))
neg = pickle.load(open(os.path.join(nulldist_dir, 'neg' + '.pkl'), "rb"))
others = pickle.load(open(os.path.join(nulldist_dir, 'others' + '.pkl'), "rb"))

pos, neg, others = np.array(pos), np.array(neg), np.array(others)


subtitles = ['Positive', 'Negative', 'Neutral']
subcolors = ['C0', 'C1', 'C2']

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
    

    # Add title and axis labels
    ax.set_xlabel(f'Test statistic ({subtitles[i]})')
    ax.set_ylabel('Density')


handles, labels = ax.get_legend_handles_labels()

# # Add legend to last plot
# axs[2].legend(['Kernel Density Estimate', 'Mean', 'Std Dev'])
fig.suptitle('Comparison of Three Null Distributions', fontsize=16, fontweight='bold', x = 0.25)
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=[0.75, 0.905])

# # Adjust layout
plt.tight_layout()

# Show plot
null_comparison_path = os.path.join(plot_dir, "null_comparison.png")
plt.savefig(null_comparison_path)

print(f"Visualization of the null distributions saved to {null_comparison_path} ...")



# 3. KS test against standard normal distribution
sample_data_dict = {
    "positive": pos,
    "negative": neg,
    "neutral": others
}
for kw, sample_data in sample_data_dict.items():
    print(f"KS test against standard normal distribution for {kw} null distribution ...")
    
    # Calculate means and std devs
    print('Mean:', sample_data.mean())
    print('Std Dev:', sample_data.std())
    
    # Perform the KS test against a standard normal distribution
    ks_statistic, p_value = stats.kstest(sample_data, 'norm', method = 'exact')    
    print('KS statistic:', ks_statistic)
    print('p-value:', p_value)

