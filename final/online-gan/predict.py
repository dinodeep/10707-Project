import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
from tqdm import tqdm

from networks import Generator, Discriminator
from train import GEN_SAVE_PATH, DISC_SAVE_PATH, LATENT_DIM, DEVICE, save_predictions

NUM_PREDICTIONS = 100


def main():

    # load hte generator
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(GEN_SAVE_PATH, map_location=DEVICE))
    G.eval()

    # perform predictions for every class and save results
    for cls in range(10):
        noise = torch.randn(NUM_PREDICTIONS, LATENT_DIM).to(DEVICE)
        labels = torch.ones(NUM_PREDICTIONS).to(torch.int64).to(DEVICE) * cls
        save_predictions(G, noise, labels, f"results/class-{cls}.png", nrow=10)


if __name__ == "__main__":
    main()
