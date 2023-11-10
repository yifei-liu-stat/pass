"""
Get FID score from two sets of images of the same size.
"""


import torch, torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor, Lambda
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline

import numpy as np
from scipy.stats import f

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle


from utils import (
    set_random_seed,
    load_fids_from_pkls,
    ImageFoldersDataset,
    generate_embedding,
    get_cos_similarity,
    append_save,
)

parser = argparse.ArgumentParser(
    description="Perform two-sample Hotelling's test on Inception-V3 extracted features of two sets of images of the same size"
)

parser.add_argument("--device_id", type=int, default=0, help="Device ID for CUDA.")
parser.add_argument(
    "--feature_dimension",
    type=int,
    default=192,
    choices=[64, 192, 768, 2048],
    help="Dimension of features extracted from Inception-V3 model.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for computing the test statistic.",
)
parser.add_argument(
    "--image_folder_one",
    type=Path,
    default="./imgs/sunset1",
    help="Path to the first image folder.",
)
parser.add_argument(
    "--image_folder_two",
    type=Path,
    default="./imgs/sunset2",
    help="Path to the second image folder.",
)

parser.add_argument(
    "--fid_folder",
    type=Path,
    default="./fid_by_prompts",
    help="Path to simulated FID folder.",
)

parser.add_argument(
    "--sample_size", type=int, default=200, help="Sample size for each set of images."
)


args = parser.parse_args()

DEVICE_ID = args.device_id
FEATURE_DIMENSION = args.feature_dimension
BATCH_SIZE = args.batch_size
IMAGE_FOLDER_ONE = args.image_folder_one
IMAGE_FOLDER_TWO = args.image_folder_two
SAMPLE_SIZE = args.sample_size
FIDS_SAVE_FOLDER_OVERALL = args.fid_folder

SEED = 2023
set_random_seed(SEED)


MODEL_ID = "CompVis/stable-diffusion-v1-4"
PROMPTS_DICT = {
    "sunset1": "The sun sets behind the mountains.",
    "sunset1-copy": "The sun sets behind the mountains.",
    "sunset2-copy2": "The mountains with sunset behind.",
    "stars": "The mountains with a night sky full of shining stars.",
}


if __name__ == "__main__":
    device = "cuda:" + str(DEVICE_ID)
    transform = torchvision.transforms.Compose([ToTensor()])
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    fid = FrechetInceptionDistance(feature=FEATURE_DIMENSION, normalize=True).to(device)

    # prepare dataset
    ds = ImageFoldersDataset(IMAGE_FOLDER_ONE, IMAGE_FOLDER_TWO, transform=transform)
    idx_list = np.random.choice(len(ds), size=min(SAMPLE_SIZE, len(ds)), replace=False)
    ds = Subset(ds, idx_list)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # calculate fid score
    with torch.no_grad():
        for imgs1, imgs2 in tqdm(dl, leave=False, dynamic_ncols=True):
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            fid.update(imgs1, real=True)
            fid.update(imgs2, real=False)

    test_stat = fid.compute()

    # retrieve simulated fid distribution and get the estimated null distribution
    keyword1, keyword2 = (
        str(args.image_folder_one).split("/")[-1],
        str(args.image_folder_two).split("/")[-1],
    )
    keyword1_dir, keyword2_dir = keyword1, keyword2

    ## use mixed distribution of fid scores from keyword1 and keyword2 as the null distribution
    if keyword1 not in os.listdir(FIDS_SAVE_FOLDER_OVERALL):
        keyword1_dir = "sunset1"
        print(
            f"Keyword {keyword1} not found in {FIDS_SAVE_FOLDER_OVERALL}, using {keyword1_dir} instead."
        )

    if keyword2 not in os.listdir(FIDS_SAVE_FOLDER_OVERALL):
        keyword2_dir = "sunset1-copy"
        print(
            f"Keyword {keyword2} not found in {FIDS_SAVE_FOLDER_OVERALL}, using {keyword2_dir} instead."
        )

    fids1 = load_fids_from_pkls(keyword1_dir, FIDS_SAVE_FOLDER_OVERALL)
    fids2 = load_fids_from_pkls(keyword2_dir, FIDS_SAVE_FOLDER_OVERALL)
    fids_mixed = np.array(fids1 + fids2)

    pvalue = np.mean(fids_mixed >= test_stat.item())

    # cosine similarity
    embedding1 = generate_embedding(pipe, PROMPTS_DICT[keyword1], device)
    embedding2 = generate_embedding(pipe, PROMPTS_DICT[keyword2], device)
    cos_similarity = get_cos_similarity(embedding1, embedding2)

    print(f"Comparison between {keyword1} and {keyword2}:")
    print(f"Cosine similarity between {keyword1} and {keyword2} is: {cos_similarity}")
    print(f"The fid score between {keyword1} and {keyword2} is {test_stat.item():.6f}")
    print(f"The p value for the test is {pvalue:.6f}")

    # # save the results, append if exists, create if not

    # # save the test statistic
    # test_stat_save_path = os.path.join(os.getcwd(), "test_stat.pkl")
    # append_save(test_stat_save_path, test_stat.item())
    # print(f"Test statistic saved to {test_stat_save_path}.")

    # # save the p value
    # p_value_save_path = os.path.join(os.getcwd(), "p_value.pkl")
    # append_save(p_value_save_path, pvalue)
    # print(f"P value saved to {p_value_save_path}.")

    # # save the cosine similarity
    # cosine_similarity_save_path = os.path.join(os.getcwd(), "cosine_similarity.pkl")
    # append_save(cosine_similarity_save_path, cos_similarity)
    # print(f"Cosine similarity saved to {cosine_similarity_save_path}.")
