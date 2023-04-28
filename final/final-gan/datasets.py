import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import sys

from networks import (
    MLP_CGAN_RELU,
    SMALL_MLP_CGAN_LEAKY,
    MLP_CGAN_LEAKY,
    BIG_MLP_CGAN_LEAKY,
    DCCGAN,
    Discriminator,
    CNN,
    LATENT_DIM
)

'''
Versions of the MNIST dataset that this file should load
    1. original fashion mnist dataset
    2. imbalanced fashion mnist dataset (reference dataset that should be evaluated on)
        - varying levels of imbalance
        - sampling uniform image
        - sampling from uniform class, then uniform image from class (equivalent to upsampling)
    3. re-balanced fashion mnist dataset
        - sub-sampling
        - super-sampling
        - SMOTE
        - GAN-sampling
'''

SEED = 10707
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((28, 28), antialias=True),
    ])
    return all_transforms


class FashionImbalanced(data.Dataset):
    '''
        Imbalances the FashionMNIST dataset. Keeps the number of samples
        per class as the class value increases
    '''

    def __init__(self, scale=0.5, train=True):

        # load the original dataset
        all_transforms = get_transforms()
        self.ds = datasets.FashionMNIST('./', train=train, download=True, transform=all_transforms)

        # the following variables will be re-balanced
        self.images = []
        self.labels = []
        self.class2idx = {}
        self.class2count = {}

        for i in range(len(self.ds)):
            img, label = self.ds[i]
            self.images.append(img)
            self.labels.append(label)
            if label not in self.class2idx:
                self.class2idx[label] = []
            self.class2idx[label].append(i)

        for label, idxs in self.class2idx.items():
            self.class2count[label] = len(idxs)

        # imbalance it
        self.__imbalance(scale)

    def __imbalance(self, scale):
        self.class2newcount = {}
        for label, count in self.class2count.items():
            self.class2newcount[label] = int(np.ceil(count * (scale ** label)))

        # set seed to ensure that all models are trained with the same dataset
        random.seed(10707)

        self.new_indices = []
        self.class2newidx = {}
        for label, idxs in self.class2idx.items():
            newcount = self.class2newcount[label]
            self.class2newidx[label] = random.sample(idxs, k=newcount)

        # unset seed to ensure randomness outside of this portion
        random.seed()

        # now sub-sample the dataset
        self.newimages = []
        self.newlabels = []
        self.class2idx = {}
        i = 0
        for label, idxs in self.class2newidx.items():
            self.class2idx[label] = []
            for old_idx in idxs:
                self.newimages.append(self.images[old_idx])
                self.newlabels.append(self.labels[old_idx])
                self.class2idx[label].append(i)
                i += 1

        # set the final values (self.class2idx already set)
        self.images = self.newimages
        self.labels = self.newlabels
        self.class2count = self.class2newcount

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


class FashionDownsampled(data.Dataset):

    def __init__(self, scale=0.5, train=True):
        super().__init__()
        self.ds = FashionImbalanced(scale=scale, train=train)
        self.smallest_class_count = min(list(self.ds.class2count.values()))
        self.num_classes = len(list(self.ds.class2count.keys()))
        self.length = self.num_classes * self.smallest_class_count

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # get class and then get image
        cls = idx // self.smallest_class_count
        idx = idx % self.smallest_class_count

        sample_idx = self.ds.class2idx[cls][idx]
        return self.ds.images[sample_idx], self.ds.labels[sample_idx]


class FashionUpsampled(data.Dataset):

    def __init__(self, scale=0.5, train=True):
        super().__init__()
        self.ds = FashionImbalanced(scale=scale, train=train)
        self.largest_class_count = max(list(self.ds.class2count.values()))
        self.num_classes = len(list(self.ds.class2count.keys()))
        self.length = self.num_classes * self.largest_class_count

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # get class and then get image
        cls = idx // self.largest_class_count
        idx = idx % self.largest_class_count

        class_idxs = self.ds.class2idx[cls]
        sample_idx = class_idxs[idx % len(class_idxs)]
        return self.ds.images[sample_idx], self.ds.labels[sample_idx]


class FashionModelRebalanced(data.Dataset):

    def __init__(self, genpath, gencls, scale=0.5, train=True):
        super().__init__()

        # load original dataset
        self.ds = FashionImbalanced(scale=scale, train=train)
        self.largest_class_count = max(list(self.ds.class2count.values()))
        self.num_classes = len(list(self.ds.class2count.keys()))
        self.length = self.num_classes * self.largest_class_count

        G = gencls()
        G.load_state_dict(torch.load(genpath, map_location=DEVICE))
        G.eval()

        # for each class up-sample by generating images from a GAN
        self.images = []
        self.labels = []
        for cls, idxs in self.ds.class2idx.items():
            # add existing images
            num_existing = len(idxs)
            for idx in idxs:
                self.images.append(self.ds.images[idx])
                self.labels.append(self.ds.labels[idx])

            # supplement with generated images
            num_leftover = self.largest_class_count - num_existing

            zs = torch.randn(num_leftover, LATENT_DIM)
            labels = (torch.ones((num_leftover,)) * cls).to(torch.int64)
            gen_images = G(zs, labels).detach()

            for i in range(num_leftover):
                generated_image = gen_images[i, :, :, :]
                self.images.append(generated_image)
                self.labels.append(cls)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_fashion_original(bs=128):
    all_transforms = get_transforms()
    ds_train = datasets.FashionMNIST('./', train=True, download=True, transform=all_transforms)
    ds_valid = datasets.FashionMNIST('./', train=False, download=True, transform=all_transforms)
    dl_train = data.DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
    dl_valid = data.DataLoader(ds_valid, batch_size=bs, shuffle=True, num_workers=4)
    return ds_train, dl_train, ds_valid, dl_valid


def load_fashion_imbalanced(bs=128, scale=0.5):
    ds_train = FashionImbalanced(train=True, scale=scale)
    ds_valid = FashionImbalanced(train=False, scale=scale)
    dl_train = data.DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
    dl_valid = data.DataLoader(ds_valid, batch_size=bs, shuffle=True, num_workers=4)
    return ds_train, dl_train, ds_valid, dl_valid


def load_fashion_rebalanced(cls, bs=128, scale=0.5):
    '''helper function for returning datasets/loaders that are rebalanced'''
    ds_train = cls(scale=scale, train=True)
    ds_valid = cls(scale=scale, train=False)
    dl_train = data.DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
    dl_valid = data.DataLoader(ds_valid, batch_size=bs, shuffle=True, num_workers=4)
    return ds_train, dl_train, ds_valid, dl_valid


def load_fashion_downsampled(bs=128, scale=0.5):
    return load_fashion_rebalanced(FashionDownsampled, bs=bs, scale=scale)


def load_fashion_upsampled(bs=128, scale=0.5):
    return load_fashion_rebalanced(FashionUpsampled, bs=bs, scale=scale)


def load_fashion_gan(genpath, gencls, bs=128, scale=0.5):
    '''helper function for returning datasets/loaders that are generated via gans'''
    def load_model_ds(scale=scale, train=True):
        return FashionModelRebalanced(genpath, gencls, scale=scale, train=train)
    return load_fashion_rebalanced(load_model_ds, bs=bs, scale=scale)


def get_class_counts(ds):
    counts = {}
    for i in range(len(ds)):
        _, label = ds[i]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    return counts


if __name__ == "__main__":
    ds = FashionModelRebalanced("results_cgan/high_unbalanced_mlp_cgan_leaky_generator.pt", MLP_CGAN_LEAKY)

    for i in [6000-1, 2*6000-1, 3*6000-1, 4*6000-1, 5*6000-1, 6*6000-1, 7*6000-1, 8*6000-1, 9*6000-1]:
        image, label = ds[i]
        image = (image + 1) / 2
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
