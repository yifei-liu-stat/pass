"""
PAI for estimating the null distribution of FID score between visual representations of two prompts.
"""

import torch
import torch.multiprocessing as mp

from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance

from tqdm import tqdm
import pickle
import os
import argparse
from utils import suppress_stderr


parser = argparse.ArgumentParser(
    description="Calculate FID between two prompts using StableDiffusionPipeline, with MC replications, possibly distributed with specfied CUDA devices."
)
parser.add_argument(
    "--inception_dimension", type=int, default=192, help="Inception Dimension"
)
parser.add_argument(
    "--sd_model_id",
    type=str,
    default="CompVis/stable-diffusion-v1-4",
    help="Stable Diffusion model ID",
)
parser.add_argument("--guidance_scale", type=int, default=2, help="Guidance Scale")
parser.add_argument(
    "--prompt1",
    type=str,
    default="The sun sets behind the mountains.",
    help="First prompt",
)
parser.add_argument(
    "--prompt2",
    type=str,
    default="The sun sets behind the mountains.",
    help="Second prompt",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="./fid_by_prompts/sunset1",
    help="Save folder",
)
parser.add_argument(
    "--devices", nargs="+", type=int, default=[0, 1, 2, 3], help="List of device IDs"
)
parser.add_argument(
    "--monte_carlo_size", type=int, default=200, help="Monte Carlo size"
)
parser.add_argument("--sample_size", type=int, default=200, help="Sample size")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument(
    "--return_flag",
    type=bool,
    default=False,
    help="Either returns fids or save them in pkl format (exclusive).",
)


args = parser.parse_args()

INCEPTION_DIMENSION = args.inception_dimension
SD_MODEL_ID = args.sd_model_id
GUIDANCE_SCALE = args.guidance_scale
PROMPT1 = args.prompt1
PROMPT2 = args.prompt2
SAVE_FOLDER = args.save_folder
DEVICES = args.devices
MONTE_CARLO_SIZE = args.monte_carlo_size
SAMPLE_SIZE = args.sample_size
BATCH_SIZE = args.batch_size
RETURN_FLAG = args.return_flag


keyword = SAVE_FOLDER.split("/")[-1]
if not os.path.isdir(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

num_batches, remaining = SAMPLE_SIZE // BATCH_SIZE, SAMPLE_SIZE % BATCH_SIZE
batch_size_list = [BATCH_SIZE] * num_batches + ([remaining] if remaining else [])


mc_size, remaining = MONTE_CARLO_SIZE // len(DEVICES), MONTE_CARLO_SIZE % len(DEVICES)
if remaining:
    mc_size += 1
    remaining = MONTE_CARLO_SIZE - mc_size * (len(DEVICES) - 1)
    mc_sizes = [mc_size] * (len(DEVICES) - 1) + [remaining]
else:
    mc_sizes = [mc_size] * len(DEVICES)


def simulate(device_id, mc_size):
    pklpath = os.path.join(SAVE_FOLDER, "-".join([keyword, str(device_id)]) + ".pkl")
    device = "cuda:" + str(device_id)

    print(
        f'Simulating distribution of FID scores between visual representations of prompts "{PROMPT1}" and "{PROMPT2}" on CUDA {device_id} ...'
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = lambda images, clip_input: (
        images,
        False,
    )  # disable NSFW checker
    fid = FrechetInceptionDistance(feature=INCEPTION_DIMENSION, normalize=True).to(
        device
    )

    if not RETURN_FLAG:
        if os.path.exists(pklpath):
            # change mc_size if keyword-device_id.pkl already exists
            fids = pickle.load(open(pklpath, "rb"))

            if len(fids) >= mc_size:
                # stop generating if the specified mc_size has been satisfied
                print(
                    f"CUDA {device_id}: already has {len(fids)} replications, exceeding allocated Monte Carlo size {mc_size}), stop generating."
                )
                return
            else:
                # otherwise, update the mc_size to generate fewer fids scores
                mc_size = mc_size - len(fids)  # strictly positive.
                print(
                    f"CUDA {device_id}: already has {len(fids)} replications, sill have {mc_size} replications left, continue generating:"
                )
        else:
            # initialize fids if the path does not exist
            fids = []

    for d in range(mc_size):
        loop = tqdm(
            enumerate(batch_size_list), total=len(batch_size_list), dynamic_ncols=True
        )
        loop.set_description(f"CUDA {device_id}, Replication [{d} / {mc_size}]")
        for _, batch_size in loop:
            with suppress_stderr():
                imgs1 = pipe(
                    [PROMPT1] * batch_size,
                    output_type="nd",
                    guidance_scale=GUIDANCE_SCALE,
                ).images
                imgs1 = torch.tensor(imgs1).permute(0, 3, 1, 2).to(device)
                fid.update(imgs1, real=True)

                imgs2 = pipe(
                    [PROMPT2] * batch_size,
                    output_type="nd",
                    guidance_scale=GUIDANCE_SCALE,
                ).images
                imgs2 = torch.tensor(imgs2).permute(0, 3, 1, 2).to(device)
                fid.update(imgs2, real=False)

        fidvalue = fid.compute()

        fids.append(fidvalue.item())
        if not RETURN_FLAG:
            pickle.dump(fids, open(pklpath, "wb"))

        fid.reset()

    if RETURN_FLAG:
        return fids


if __name__ == "__main__":
    ctx = mp.get_context("spawn")

    print("Start multiprocessing")

    processes = []
    for device_id, mc_size in zip(DEVICES, mc_sizes):
        process = ctx.Process(target=simulate, args=(device_id, mc_size))

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Multiprocessing complete")
