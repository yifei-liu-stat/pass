"""
Generate imgs from given prompts, with CUDA parallelizatoin
"""

import torch
from diffusers import StableDiffusionPipeline

import torch.multiprocessing as mp
import os  # identify the process information
import argparse
from PIL import Image
from tqdm import tqdm

from utils import suppress_stderr


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    default="CompVis/stable-diffusion-v1-4",
    help="Version of stable diffusion model.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="The sun sets behind the mountains.",
    help="Prompt for image generation.",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7,
    help="Guidance scale for the stable diffusion model.",
)
parser.add_argument(
    "--total_num_imgs",
    type=int,
    default=200,
    help="Total number of images to generate.",
)
parser.add_argument(
    "--image_size", type=int, default=299, help="Image size to be outputed."
)
parser.add_argument(
    "--device_ids",
    type=int,
    nargs="+",
    default=[0, 1, 2, 3],
    help="List of device IDs to use for multiprocessing.",
)
parser.add_argument(
    "--plot_folder",
    type=str,
    default="./imgs/",
    help="Path to the folder where images will be saved.",
)


args = parser.parse_args()

MODEL_ID = args.model_id
PROMPT = args.prompt
GUIDANCE_SCALE = args.guidance_scale
TOTAL_NUM_IMGS = args.total_num_imgs
IMAGE_SIZE = args.image_size
DEVICE_IDS = args.device_ids
PLOT_FOLDER = args.plot_folder

if not os.path.isdir(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

num_imgs_per_device = TOTAL_NUM_IMGS // len(DEVICE_IDS)
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.safety_checker = lambda images, clip_input: (
    images,
    False,
)  # disable NSFW checker for false positives


def count_imgs(img_folder, cuda_number=0):
    """Count number of images generated from `cuda:cuda_number` under `img_folder`

    Args:
        img_folder (str): Path to image folder
        cuda_number (int, optional): 0-index cuda device nubmer. Defaults to 0.

    Returns:
        int: number of images generated from `cuda:cuda_number` under `img_folder`
    """
    keyword = "-".join(["img", str(cuda_number)])
    img_names = os.listdir(img_folder)
    return sum([img_name.startswith(keyword) for img_name in img_names])


def stablediffusion(cuda_number=0):
    counts = count_imgs(PLOT_FOLDER, cuda_number)
    device = "cuda:" + str(cuda_number)
    pipe_cuda = pipe.to(device)

    print(
        f'Generating images from CUDA {cuda_number} based one the prompt "{PROMPT}" ...'
    )

    loop = tqdm(
        enumerate(range(num_imgs_per_device)),
        total=num_imgs_per_device,
        dynamic_ncols=True,
    )
    loop.set_description(f"CUDA {cuda_number}:")
    for i, _ in loop:
        with suppress_stderr():
            img = pipe_cuda(PROMPT, guidance_scale=GUIDANCE_SCALE).images[0]

            img_name = "-".join(["img", str(cuda_number), str(counts + i)]) + ".png"
            plot_path = os.path.join(PLOT_FOLDER, img_name)
            img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img.save(plot_path)


if __name__ == "__main__":
    # start multiprocessing

    ctx = mp.get_context("spawn")

    print("Start multiprocessing")

    processes = []
    for device_id in DEVICE_IDS:
        process = ctx.Process(target=stablediffusion, args=(device_id,))

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Multiprocessing complete")


