import os
import sys
import pickle
from contextlib import contextmanager

import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F


class ImageFoldersDataset(Dataset):
    """Pair two image folders together as a dataset."""

    def __init__(self, image_folder_one, image_folder_two, transform=None):
        self.image_folder_one = image_folder_one
        self.image_filenames_one = os.listdir(image_folder_one)
        self.image_folder_two = image_folder_two
        self.image_filenames_two = os.listdir(image_folder_two)
        self.transform = transform

    def __len__(self):
        return min(len(self.image_filenames_one), len(self.image_filenames_two))

    def __getitem__(self, idx):
        image_path_one = os.path.join(
            self.image_folder_one, self.image_filenames_one[idx]
        )
        image_one = Image.open(image_path_one).convert("RGB")
        image_path_two = os.path.join(
            self.image_folder_two, self.image_filenames_two[idx]
        )
        image_two = Image.open(image_path_two).convert("RGB")
        if self.transform:
            image_one = self.transform(image_one)
            image_two = self.transform(image_two)

        return image_one, image_two


# set random seed for all possible sources of randomness
def set_random_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def suppress_stderr():
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


def load_fids_from_pkls(keyword, fids_save_folder_overall):
    fids_save_folder = os.path.join(fids_save_folder_overall, keyword)
    fids = []
    for pklfile in os.listdir(fids_save_folder):
        pklpath = os.path.join(fids_save_folder, pklfile)
        fids.extend(pickle.load(open(pklpath, "rb")))
    return fids


def generate_embedding(pipe, prompt, device):
    """
    generate embeddings for a prompt based on a stable diffusion text encoder
    - pipe: a stable diffusion pipeline
    """
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(input_ids)["last_hidden_state"]
    return text_embeddings


def get_cos_similarity(e1, e2):
    return F.cosine_similarity(e1.flatten(), e2.flatten(), dim=0).item()


def append_save(path, data):
    """Save the data to the path: append if exists, create if not"""
    if os.path.exists(path):
        temp = pickle.load(open(path, "rb"))
        temp.append(data)
        pickle.dump(temp, open(path, "wb"))
    else:
        pickle.dump([data], open(path, "wb"))


def image_grid(imgs, rows, cols):
    """Convert a list of PIL images to image grid by creating a big new image and pasting from left to right, top to bottom

    Args:
        imgs (PIL): A list of images in PIL format
        rows (int): Number of rows to be displayed
        cols (int): Number of columns to be displayed

    Returns:
        PIL: A large image that displays `len(imgs)` images in a grid with `rows` rows and `cols` columns.
    """
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
