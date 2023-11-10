"""
Display images for different prompts for a visual comparison (replicate Fig 6 from the paper)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils.utils import image_grid


PLOT_PATH = "./results/prompt_images.png"
IMAGE_FOLDER = "./data/"


comparison_keywords = ["sunset1", "sunset2-copy2", "stars"]
num_images_per_class, nrows, ncols = 25, 5, 5
subtitles = [
    'Prompts 1 and 2:\n "The sun sets behind the mountains."',
    'Prompt 3:\n "The mountains with sunset behind."',
    'Prompt 4:\n "The mountains with a night sky full of shining stars."',
]


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 18))

for i in range(len(axes)):
    temp_folder = os.path.join(IMAGE_FOLDER, comparison_keywords[i])
    img_names = np.array(os.listdir(temp_folder))
    idxs = np.random.choice(len(img_names), num_images_per_class, replace=False)
    temp_imgs = [
        Image.open(os.path.join(temp_folder, temp_img_name))
        for temp_img_name in img_names[idxs]
    ]
    temp_grid = image_grid(temp_imgs, nrows, ncols)

    axes[i].imshow(temp_grid)
    axes[i].axis("off")
    axes[i].set_title(subtitles[i], y=1.02, fontsize=45)

# fig.suptitle("Images generated from different prompts", fontsize = 18, fontweight = 'bold', y = 0.1)
fig.subplots_adjust(wspace=0.22)
# fig.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Saved plot to {PLOT_PATH}")

plt.close()
