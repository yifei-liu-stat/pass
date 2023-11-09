"""
PAI for assessing GLOW, DCGAN and DDPM on CIFAR-10, using FID score.
"""


import numpy as np
import torch
import torchvision


from utils.inference import set_random_seed, pvalue
from utils.fid import calculate_fretchet, InceptionV3
from utils.data import (
    get_cifar10,
    show,
    get_images,
    from_images_to_dataset,
)
from utils.dcgan import load_generator_dcgan
from utils.glow import load_glow, generate_glow
from utils.ddpm import load_diffusion_cifar10

import os
import pickle
import gdown


latent_dim = 2048
cifar10_dir = "./data"
fid_null_dist_path = "./results/fid_null_dist_dict.pkl"

ddpm_ckpt_path = "./ckpt/ddpm_competitor.pt"
dcgan_ckpt_path = "./ckpt/dcgan_generator.pt"
glow_ckpt_dir = "./ckpt/glow/"
fake_img_dir = "./results/fake_images/"

if not os.path.exists(fake_img_dir):
    os.makedirs(fake_img_dir)

# Retrieve checkpoints from Google Drive if not exist
if not os.path.exists("./ckpt"):
    print("Downloading checkpoints ...")

    url = "https://drive.google.com/drive/folders/1jrOyIxNsc4Wtdsv3LcJXLb8pF-VVBt6X?usp=drive_link"
    gdown.download_folder(url, quiet=False, use_cookies=False)

    print("Download finished.")
else:
    print("Checkpoints already exist.")

cuda_device_id = 0
cuda_device = f"cuda:{cuda_device_id}"
torch.cuda.set_device(cuda_device_id)
set_random_seed(2023)

# Load the null distribution of FID scores, estimated by PASS-DDPM
fid_null_dist_dict = pickle.load(open(fid_null_dist_path, "rb"))

# Prepare dataset and inception model for FID calculation
train_set, test_set = get_cifar10(cifar10_dir)
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx], normalize_input=False)


# 1. DCGAN
print("PAI for assessing DCGAN started.")
generator_dcgan = load_generator_dcgan(dcgan_ckpt_path)


## Display fake images of DCGAN
fake_imgs_dcgan = generator_dcgan(torch.randn((25, 1024, 1, 1), dtype=torch.float32))
grid = torchvision.utils.make_grid(
    fake_imgs_dcgan, nrow=5, normalize=True, value_range=(-1, 1)
)
show(grid, os.path.join(fake_img_dir, "dcgan.png"))


## Uncertainty quantification on generation quality of DCGAN
res_dcgan = []
score_dcgan = []
for i, n in enumerate([2050, 5000, 10000]):
    print(f"Inference size n = {n}:")

    true_imgs = get_images(test_set, n)
    true_dataset = from_images_to_dataset(true_imgs)

    fake_imgs_dcgan = generator_dcgan(torch.randn((n, 1024, 1, 1), dtype=torch.float32))
    fake_imgs_dcgan = fake_imgs_dcgan.detach()
    fake_dataset_dcgan = from_images_to_dataset(fake_imgs_dcgan)
    score = calculate_fretchet(
        true_dataset,
        fake_dataset_dcgan,
        batch_size=50,
        device="cuda",
        model=model,
        dims=latent_dim,
    )
    res_dcgan.append(pvalue(fid_null_dist_dict[str(n)], score))
    score_dcgan.append(score)

print("PAI with DCGAN finished, results are (n = 2050, 5000, 10000):")
print(f"(FID scores) {score_dcgan}")
print(f"(P-values) {res_dcgan}")


# 2. GLOW
print("PAI for assessing GLOW started.")
generator_glow = load_glow(glow_ckpt_dir, "ckpt.pt")


## Display fake images of GLOW
fake_imgs_glow = generate_glow(generator_glow, 25)
grid = torchvision.utils.make_grid(
    fake_imgs_glow, nrow=5, normalize=True, value_range=(-1, 1)
)
show(grid, os.path.join(fake_img_dir, "glow.png"))


## Uncertainty quantification on generation quality of GLOW
res_glow = []
score_glow = []
for i, n in enumerate([2050, 5000, 10000]):
    print(f"Inference size n = {n}:")

    true_imgs = get_images(test_set, n)
    true_dataset = from_images_to_dataset(true_imgs)

    # glow
    fake_imgs_glow = generate_glow(generator_glow, n)
    fake_imgs_glow = fake_imgs_glow.detach()
    fake_dataset_glow = from_images_to_dataset(fake_imgs_glow)
    score = calculate_fretchet(
        true_dataset,
        fake_dataset_glow,
        batch_size=50,
        device="cuda",
        model=model,
        dims=latent_dim,
    )
    res_glow.append(pvalue(fid_null_dist_dict[str(n)], score))
    score_glow.append(score)

print("PAI with GLOW finished, results are (n = 2050, 5000, 10000):")
print(f"(FID scores) {score_glow}")
print(f"(P-values) {res_glow}")


# 3. DDPM

print("PAI for assessing GLOW started.")
_, diffusion = load_diffusion_cifar10(0, ddpm_ckpt_path)
diffusion = diffusion.cuda()

## Display fake images of DDPM
fake_imgs_ddpm = diffusion.sample(25).cpu()
fake_imgs_ddpm = 2 * fake_imgs_ddpm - 1
grid = torchvision.utils.make_grid(
    fake_imgs_ddpm, nrow=5, normalize=True, value_range=(-1, 1)
)
show(grid, os.path.join(fake_img_dir, "ddpm.png"))


res_ddpm = []
score_ddpm = []
for i, n in enumerate([2050, 5000, 10000]):
    print(f"Inference size n = {n}:")

    # get a random subset of true images
    true_imgs = get_images(test_set, n)
    true_dataset = from_images_to_dataset(true_imgs)

    # get fake images of same size
    diffusion = diffusion.cuda()

    # split into batches to avoid CUDA out-of-memory issue
    nc = n // 2500
    nr = n % 2500
    n_split = [2500] * nc if nc else []
    n_split = n_split + [nr] if nr else n_split

    fake_imgs = []
    for m in n_split:
        temp_imgs = diffusion.sample(m)  # on cuda device by default
        temp_imgs = temp_imgs.cpu()
        fake_imgs.append(temp_imgs)
    diffusion = diffusion.cpu()  # a workaround for CUDA out-of-memory issue

    fake_imgs = torch.cat(fake_imgs, 0)
    fake_imgs = fake_imgs * 2 - 1  # range [-1, 1]
    fake_dataset = from_images_to_dataset(fake_imgs)

    score = calculate_fretchet(
        true_dataset,
        fake_dataset,
        batch_size=50,
        device=cuda_device,
        model=model,
        dims=latent_dim,
    )

    res_ddpm.append(pvalue(fid_null_dist_dict[str(n)], score))
    score_ddpm.append(score)


print("PAI with DDPM finished, results are (n = 2050, 5000, 10000):")
print(f"(FID scores) {score_ddpm}")
print(f"(P-values) {res_ddpm}")


print("Saving to ./results/ folder ...")

score_dict = {
    "dcgan": score_dcgan,
    "glow": score_glow,
    "ddpm": score_ddpm,
}

pvalue_dict = {
    "dcgan": res_dcgan,
    "glow": res_glow,
    "ddpm": res_ddpm,
}

score_save_path = "./results/competitor_score_dict.pkl"
pickle.dump(score_dict, open(score_save_path, "wb"))
print(f"Competitor score dict saved in {score_save_path}")

pvalue_save_path = "./results/competitor_pvalue_dict.pkl"
pickle.dump(pvalue_dict, open(pvalue_save_path, "wb"))
print(f"Competitor p-value dict saved in {pvalue_save_path}")
