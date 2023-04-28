import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

from networks import Generator, Discriminator, CNN
from datasets import load_fashion_original

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5
LATENT_DIM = 100
BATCH_SIZE = 100

LOW_IMBALANCE_SCALE = 7/9
HIGH_IMBALANCE_SCALE = 1/2

ALL_METHODS = {
    "original": load_fashion_original,
    "low-unbalanced": None,
    "high-unbalanced": None,
    # the below ones are meant to be built off of purely the unbalanced datasets
    "downsampled": None,
    "upsampled": None,
    "mlp-gan-relu": None,
    "mlp-gan-leaky": None,
    "big-mlp-gan-leaky": None,
    "mlp-gan-diff-loss": None,
    "dcgan-leaky": None,
}

METHOD = "original"
CNN_SAVE_PATH = f"results_cnn/{METHOD}_cnn.pt"
PLOT_SAVE_PATH = f"results_cnn/{METHOD}_plot.png"
FINAL_SAVE_PATH = f"results_cnn/{METHOD}_results.pkl"
SAVE_EPOCH_INTERVAL = 1


def evaluate(model, dl, lossfn):
    # compute the loss and accuracy of the model
    model = model.to(DEVICE)

    with torch.no_grad():
        acc = 0
        loss = 0
        for batch in dl:
            # unpack data
            bs = batch[0].shape[0]
            imgs = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)

            # perform predictions and compute loss
            yhat = model(imgs)
            loss += bs * lossfn(yhat, F.one_hot(labels, num_classes=10).to(torch.float32))

            # compute the accuracy
            labelshat = torch.argmax(yhat, dim=-1)
            acc += torch.sum(labelshat == labels)

        n = len(dl.dataset)
        acc = acc.item() / n
        loss = loss.item() / n

    return acc, loss


def train():

    load_method_ds = ALL_METHODS[METHOD]
    _, train_dl, _, test_dl = load_method_ds(bs=BATCH_SIZE)

    # initialize everything for training
    model = CNN().to(DEVICE)

    lossfn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0005)

    step = 0
    num_steps = len(train_dl) * EPOCHS

    # for final plotting
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for i in range(EPOCHS):
        for j, batch in enumerate(train_dl):
            # unpack data
            imgs = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)

            # perform predictions and compute loss
            yhat = model(imgs)
            loss = lossfn(yhat, F.one_hot(labels, num_classes=10).to(torch.float32))

            # update
            opt.zero_grad()
            loss.backward()
            opt.step()

        # evaluate the model
        if i % SAVE_EPOCH_INTERVAL == 0:
            train_acc, train_loss = evaluate(model, train_dl, lossfn)
            test_acc, test_loss = evaluate(model, test_dl, lossfn)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_losses)
            test_accs.append(test_acc)

            print(f"Epoch {i}: train-acc={train_acc:.3f} train-loss={train_loss:.3f} test-acc={test_acc:.3f} test-loss={test_loss:.3f} -- saving model")
            torch.save(model.state_dict(), CNN_SAVE_PATH)

    # save all of the results to a pickle file
    with open(FINAL_SAVE_PATH, "wb") as f:
        d = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
        }
        pickle.dump(d, f)


if __name__ == "__main__":
    train()
