"""
Get the null distribution of FID scores estimated from PASS-DDPM.
- cd to assess_generators before running this script.
- will take a long time to finish. For quick exploration, one can use results/fid_null_dist_dict.pkl for inference/comparison first.
"""

import numpy as np
import torch

import os
import random
import pickle
import argparse
import matplotlib.pyplot as plt

from utils.data import (
    get_cifar10,
    get_images,
    from_images_to_dataset,
)

from utils.fid import (
    calculate_fretchet,
    InceptionV3,
)

from utils.ddpm import load_diffusion_cifar10

from utils.inference import set_random_seed


data_dir = "./data"
null_ckpt_path = "./ckpt/ddpm_pass.pt"


set_random_seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--cuda-device", default=0, type=int)
    args = parser.parse_args()

    gpu_device = args.cuda_device
    # - will be replaced with _ when naming
    torch.cuda.set_device(gpu_device)

    print(f"cuda device {gpu_device} is activated.")

    D = 500  # Monte Carlo Size
    n_list = [
        2050,
        5000,
        10000,
    ]  # should be multiple of batch size, which is 50 by default
    latent_dim = 2048
    fid_dist = np.zeros((len(n_list), D))
    _, test_set = get_cifar10(data_dir)  # range [-1, 1]

    _, diffusion = load_diffusion_cifar10(0, ckpt_path=null_ckpt_path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[latent_dim]
    model = InceptionV3([block_idx], normalize_input=False)  # on GPU

    for i, n in enumerate(n_list):
        for j in range(D):
            # get a random subset of true images
            true_imgs = get_images(test_set, n)
            true_dataset = from_images_to_dataset(true_imgs)

            # get fake images of same size
            diffusion = diffusion.cuda()

            # if more than 2500 (around), GPU will be out of memory, so split it
            nc = n // 2500
            nr = n % 2500
            n_split = [2500] * nc if nc else []
            n_split = n_split + [nr] if nr else n_split

            fake_imgs = []
            for m in n_split:
                temp_imgs = diffusion.sample(m)  # on cuda device by default
                temp_imgs = temp_imgs.cpu()
                fake_imgs.append(temp_imgs)
            diffusion = diffusion.cpu()

            fake_imgs = torch.cat(fake_imgs, 0)
            fake_imgs = fake_imgs * 2 - 1  # range [-1, 1]
            fake_dataset = from_images_to_dataset(fake_imgs)

            score = calculate_fretchet(
                true_dataset,
                fake_dataset,
                batch_size=50,
                device=gpu_device,
                model=model,
                dims=latent_dim,
            )
            fid_dist[i][j] = score

            torch.cuda.empty_cache()
            print(
                f"Progress: {i + 1}/{len(n_list)} -> {j + 1}/{D}: FID score is {score}."
            )

            filename = "fiddist" + "-" + str(gpu_device)
            if os.path.exists(filename):
                filename = filename + "-" + str(random.randint(0, 100000))
            pickle.dump(fid_dist, open(filename + ".pkl", "wb"))


print("Null distribution of FID scores saved to ./results/ folder.")
