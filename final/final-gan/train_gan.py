import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
from tqdm import tqdm

from networks import (
    MLP_CGAN_RELU,
    SMALL_MLP_CGAN_LEAKY,
    MLP_CGAN_LEAKY,
    BIG_MLP_CGAN_LEAKY,
    DCCGAN,
    Discriminator,
    LATENT_DIM
)
from datasets import load_fashion_original, load_fashion_imbalanced, load_fashion_upsampled

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 256
SAVE_EPOCH_INTERVAL = 5

LOW_IMBALANCE_SCALE = 7/9
HIGH_IMBALANCE_SCALE = 1/2

# NOTE: we are training gans with upsampled datasets so that they learn the representation
#       of unknown classes properly, as opposed to training them with some classes with few samples
ALL_DATASETS = {
    "original": load_fashion_original,
    "low_unbalanced": lambda bs: load_fashion_upsampled(bs=bs, scale=LOW_IMBALANCE_SCALE),
    "high_unbalanced": lambda bs: load_fashion_upsampled(bs=bs, scale=HIGH_IMBALANCE_SCALE),
}

ALL_GANS = {
    "mlp_cgan_relu": MLP_CGAN_RELU,
    "small_mlp_cgan_leaky": SMALL_MLP_CGAN_LEAKY,
    "mlp_cgan_leaky": MLP_CGAN_LEAKY,
    "big_mlp_cgan_leaky": BIG_MLP_CGAN_LEAKY,
    # "mlp_cgan_diff_loss": None,
    "dccgan_leaky": DCCGAN,
}


def save_predictions(G, noise, labels, save_path, nrow=10):
    B = noise.shape[0]
    gen_images = G(noise, labels)
    images = (gen_images + 1) / 2
    torchvision.utils.save_image(images, save_path, nrow=nrow)


def train(DATASET, GAN):

    GEN_SAVE_PATH = f"results_cgan/{DATASET}_{GAN}_generator.pt"
    DISC_SAVE_PATH = f"results_cgan/{DATASET}_{GAN}_discriminator.pt"

    _, dl, _, _ = ALL_DATASETS[DATASET](bs=BATCH_SIZE)

    # initialize everything for training
    G = ALL_GANS[GAN]().to(DEVICE)
    D = Discriminator().to(DEVICE)

    loss = torch.nn.BCELoss()
    g_opt = optim.Adam(G.parameters(), lr=0.0002)
    d_opt = optim.Adam(D.parameters(), lr=0.0002)

    num_steps = EPOCHS * len(dl)

    # initialize models
    with tqdm(total=num_steps) as pbar:
        for i in range(EPOCHS):
            print(f"Starting Epoch {i}")
            for j, batch in enumerate(dl):
                bs = batch[0].shape[0]
                real_imgs = batch[0].to(DEVICE)
                real_labels = batch[1].to(DEVICE)

                # create false predictions
                noise = torch.randn(bs, LATENT_DIM).to(DEVICE)
                fake_labels = torch.randint(0, 10, (bs,)).to(DEVICE)
                gen_imgs = G(noise, fake_labels)

                # discriminator update
                disc_labels = torch.ones(bs).to(DEVICE)
                gen_labels = torch.zeros(bs).to(DEVICE)

                d_opt.zero_grad()
                d_real = D(real_imgs, real_labels).view(bs)
                d_fake = D(gen_imgs.detach(), fake_labels).view(bs)

                loss_real = loss(d_real, disc_labels)
                loss_fake = loss(d_fake, gen_labels)
                d_loss = (loss_real + loss_fake) / 2

                d_loss.backward()
                d_opt.step()

                # generator update
                g_opt.zero_grad()
                gen_imgs = G(noise, fake_labels)
                d_fake = D(gen_imgs, fake_labels).view(bs)
                g_loss = loss(d_fake, disc_labels)
                g_loss.backward()
                g_opt.step()

                pbar.update()

            if i % SAVE_EPOCH_INTERVAL == 0:
                print(f"Done Epoch {i}. Saving models")
                torch.save(G.state_dict(), GEN_SAVE_PATH)
                torch.save(D.state_dict(), DISC_SAVE_PATH)


def main():

    for d in ALL_DATASETS.keys():
        for g in ALL_GANS.keys():
            train(d, g)


if __name__ == "__main__":
    main()
