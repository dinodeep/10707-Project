import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from networks import (
    MLP_CGAN_RELU,
    SMALL_MLP_CGAN_LEAKY,
    MLP_CGAN_LEAKY,
    BIG_MLP_CGAN_LEAKY,
    DCCGAN,
    Discriminator,
    LATENT_DIM
)
from train_gan import DEVICE, save_predictions

NUM_PREDICTIONS = 100

ALL_DATASETS = ["original", "low_unbalanced", "high_unbalanced"]

ALL_GANS = {
    "mlp_cgan_relu": MLP_CGAN_RELU,
    "small_mlp_cgan_leaky": SMALL_MLP_CGAN_LEAKY,
    "mlp_cgan_leaky": MLP_CGAN_LEAKY,
    "big_mlp_cgan_leaky": BIG_MLP_CGAN_LEAKY,
    # "mlp_cgan_diff_loss": None,
    "dccgan_leaky": DCCGAN,
}

'''
    - class distribution for low imbalance and high imbalance
    - 10 x 10 image of generated images for all gans (10 per class)
    - final results table for all gans
    - accuracy and loss plot for every baseline and mlp_cgan_leaky reuslts
    - accuracy and loss plot for everything
'''


def create_imgs():
    dataset = ALL_DATASETS[2]
    gan_str = "mlp_cgan_leaky"

    # load the generator
    G = ALL_GANS[gan_str]().to(DEVICE)
    G.load_state_dict(torch.load(f"results_cgan/{dataset}_{gan_str}_generator.pt", map_location=DEVICE))
    G.eval()

    # perform predictions for every class and save results
    os.makedirs(f"imgs/{dataset}_{gan_str}", exist_ok=True)
    for cls in range(10):
        noise = torch.randn(NUM_PREDICTIONS, LATENT_DIM).to(DEVICE)
        labels = torch.ones(NUM_PREDICTIONS).to(torch.int64).to(DEVICE) * cls
        save_predictions(G, noise, labels, f"imgs/{dataset}_{gan_str}/class-{cls}.png", nrow=10)


def main():
    create_imgs()


if __name__ == "__main__":
    main()
