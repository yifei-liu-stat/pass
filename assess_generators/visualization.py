"""
Visualize the null distribution of FID scores and test statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

import pickle

fid_null_dist_dict = pickle.load(open("./results/fid_null_dist_dict.pkl", "rb"))
competitor_score_dict = pickle.load(open("./results/competitor_score_dict.pkl", "rb"))
competitor_pvalue_dict = pickle.load(open("./results/competitor_pvalue_dict.pkl", "rb"))


para_dict = {
    "alpha": 0.7,
    "density": True,
    # "edgecolor": "black",
    "bins": np.linspace(30, 90, 180),
}

para_dict_marker = {"zorder": 10, "clip_on": False, "s": 200, "y": 0.005, "alpha": 0.8}


# Change the overall text size
overallsize = 15
plt.rcParams["font.size"] = overallsize
plt.rcParams["axes.titlesize"] = overallsize
plt.rcParams["axes.labelsize"] = overallsize
plt.rcParams["xtick.labelsize"] = overallsize
plt.rcParams["ytick.labelsize"] = overallsize
plt.rcParams["legend.fontsize"] = overallsize


fig, ax = plt.subplots(figsize=(16, 7))

for i, n in enumerate([2050, 5000, 10000]):
    ax.hist(fid_null_dist_dict[str(n)], label=f"n = {n}", **para_dict)
    ax.scatter(
        x=competitor_score_dict["ddpm"][i],
        color=f"C{i}",
        marker="*",
        label="Test-stat-DDPM",
        **para_dict_marker,
    )
    ax.scatter(
        x=competitor_score_dict["dcgan"][i],
        color=f"C{i}",
        marker="x",
        label="Test-stat-DCGAN",
        **para_dict_marker,
    )
    ax.scatter(
        x=competitor_score_dict["glow"][i],
        color=f"C{i}",
        marker="+",
        label="Test-stat-GLOW",
        **para_dict_marker,
    )


handles, labels = ax.get_legend_handles_labels()


temp_legend = ax.legend(
    handles=sum(
        [
            [handles[i]] + [(handles[i], handles[i + k]) for k in range(1, 4)]
            for i in [0, 4, 8]
        ],
        [],
    ),
    labels=labels,
    markerscale=0.8,
    loc="upper left",
    bbox_to_anchor=(0.75, 0.9),
    handler_map={tuple: HandlerTuple(ndivide=None, pad=1.0)},
)

sublegendsize = 12
for i in range(len(labels)):
    if i not in [0, 4, 8]:
        temp_legend.get_texts()[i].set_fontsize(sublegendsize)


plt.xlabel("FID score")
plt.ylabel("Density")
plt.title(
    "Estimated null distribution of FID scores using reference DDPM",
    weight="bold",
    size=18,
)
# plt.savefig(plotpath)
plot_path = "./results/null_distributions.png"
plt.savefig(plot_path)
plt.close()

print(f"Visualization of FID null distributions saved to {plot_path}.")
