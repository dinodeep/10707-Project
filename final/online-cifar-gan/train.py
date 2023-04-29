import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
from tqdm import tqdm

from networks import Generator, Discriminator
from data import load_fashion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
LATENT_DIM = 100
BATCH_SIZE = 100

GEN_SAVE_PATH = "results/generator.pt"
DISC_SAVE_PATH = "results/discriminator.pt"
SAVE_EPOCH_INTERVAL = 5


def save_predictions(G, noise, labels, save_path, nrow=10):
    B = noise.shape[0]
    gen_images = G(noise, labels).reshape(B, 1, 28, 28)
    images = (gen_images + 1) / 2
    torchvision.utils.save_image(images, save_path, nrow=nrow)


def train():
    _, dl = load_fashion(bs=BATCH_SIZE)

    # initialize everything for training
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    loss = torch.nn.BCELoss()
    g_opt = optim.Adam(G.parameters(), lr=0.0002)
    d_opt = optim.Adam(D.parameters(), lr=0.0002)

    step = 0
    num_steps = EPOCHS * len(dl)

    # initialize models
    with tqdm(total=num_steps) as pbar:
        for i in range(EPOCHS):
            print(f"Starting Epoch {i}")
            for j, batch in enumerate(dl):
                bs = batch[0].shape[0]
                real_imgs = batch[0].view(bs, 784).to(DEVICE)
                real_labels = batch[1].to(DEVICE)

                # create false predictions
                noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
                fake_labels = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)
                gen_imgs = G(noise, fake_labels)

                # discriminator update
                disc_labels = torch.ones(bs).to(DEVICE)
                gen_labels = torch.zeros(bs).to(DEVICE)

                d_opt.zero_grad()
                d_real = D(real_imgs, real_labels).view(bs)
                d_fake = D(gen_imgs.detach(), fake_labels).view(bs) #! why detach here?

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


if __name__ == "__main__":
    train()
