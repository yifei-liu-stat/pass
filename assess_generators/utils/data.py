import numpy as np

import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

import random
from PIL import Image as im
import matplotlib.pyplot as plt


# save a bunch of images
def save_the_image(batch_array, folderpath):
    batch_array = np.array(batch_array, dtype=np.uint8)
    for index, image in enumerate(batch_array):
        ret_tensor = im.fromarray(image)
        ret_tensor.save(folderpath + "" + "image-" + str(index) + ".png")


def show(image, path):
    plt.rcParams["savefig.bbox"] = "tight"
    plt.figure()
    plt.imshow(np.transpose(image, [1, 2, 0]))
    plt.axis("off")
    plt.savefig(path)
    plt.close()


# get a bunch of random images from dataset
def get_images(dataset, num):
    idxs = random.sample(range(len(dataset)), num)
    return torch.stack([dataset[i][0] for i in idxs], dim=0)


# get a bunch of random images from dataset, in the form of DATASET
def get_images_dataset(dataset, num):
    idxs = random.sample(range(len(dataset)), num)
    sub_dataset = Subset(dataset, idxs)
    return sub_dataset


# from images tensors (e.g. of shape (1000, 32, 32, 3)) to a pseudo dataset object
def from_images_to_dataset(images, labels=None):
    n = len(images)
    if not labels:
        labels = [0] * n
    dataset = [(images[i], labels[i]) for i in range(n)]
    return dataset


# get cifar-10 datasets, scaled between -1 and 1
def get_cifar10(path_to_data):
    # Prepare cifar-10 datasets
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # range [-1, 1]
    )

    train_set = CIFAR10(
        root=path_to_data, train=True, transform=transform, download=True
    )
    test_set = CIFAR10(
        root=path_to_data, train=False, transform=transform, download=True
    )
    return train_set, test_set
