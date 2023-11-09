"""
DCGAN for CIFAR-10. Code adapted from https://github.com/Amaranth819/DCGAN-CIFAR10-Pytorch
"""


import torch
import torch.nn as nn

# Use CIFAR dataset. The size of images is 3x32x32.

# Image size
n_img = [3, 32, 32]
# Latent vector size
nz = 1024


class Generator(nn.Module):
    """
    Noise [bs, nz, 1, 1] -> Fake images [bs, 3, 32, 32]
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # [bs, nz, 1, 1] -> [bs, 128, 4, 4]
            nn.ConvTranspose2d(nz, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # [bs, 128, 4, 4] -> [bs, 64, 8, 8]
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # [bs, 64, 8, 8] -> [bs, 32, 16, 16]
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # [bs, 32, 16, 16] -> [bs, 3, 32, 32]
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    Input images [bs, 3, 32, 32] -> [bs]
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # [bs, 3, 32, 32] -> [bs, 32, 16, 16]
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # [bs, 32, 16, 16] -> [bs, 64, 8, 8]
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # [bs, 64, 8, 8] -> [bs, 128, 4, 4]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # [bs, 128, 4, 4] -> [bs, 1, 1, 1]
            nn.Conv2d(128, 1, 4, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x).view(-1)
        return x


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.uniform_(m.weight, -0.1, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    netg = Generator().to(device)
    netd = Discriminator().to(device)

    x1 = torch.zeros((16, 3, 32, 32), dtype=torch.float32, device=device)
    x2 = torch.zeros((16, nz), dtype=torch.float32, device=device)
    y1 = netd(x1)
    y2 = netg(x2)
    print(y1.size())
    print(y2.size())


# load the generator of DCGAN for training cifar10 (30 epochs)
def load_generator_dcgan(trained_dcgan_path):
    # intialize the generator
    generator = Generator()
    generator.load_state_dict(torch.load(trained_dcgan_path))
    generator.eval()
    return generator
