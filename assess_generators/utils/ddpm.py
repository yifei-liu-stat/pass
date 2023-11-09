"""
DDPM for CIFAR-10. 
"""

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


# reload the trained diffusion model for cifar10
def load_diffusion_cifar10(milestone, ckpt_path=None):
    """
    Example:
    >>> from utils.ddpm import load_diffusion_cifar10
    >>> _, diffusion = load_diffusion_cifar10(0, "./ckpt/ddpm_pass.pt")
    >>> diffusion = diffusion.cuda()
    >>> fake_imgs_ddpm = diffusion.sample(25).cpu()
    """
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        p2_loss_weight_gamma=1,  # a big difference with p2 weighting - the original paper used a reweighted loss too
        loss_type="l1",  # L1 or L2
    ).cuda()

    # initialize the model
    trainer = Trainer(
        diffusion,
        folder=None,
        train_idx=None,
        train_batch_size=32,
        train_lr=8e-5,
        save_and_sample_every=2000,
        train_num_steps=400000,  # 700000; total training steps.
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn off mixed precision
        results_folder=None,
    )

    trainer.load(milestone=milestone, ckpt_path=ckpt_path)

    return trainer, diffusion



