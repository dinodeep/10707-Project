import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

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
from datasets import (
    load_fashion_original,
    load_fashion_imbalanced,
    load_fashion_downsampled,
    load_fashion_upsampled,
    load_fashion_gan,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100 #! adjust to until convergence
BATCH_SIZE = 100

LOW_IMBALANCE_SCALE = 7/9
HIGH_IMBALANCE_SCALE = 1/2
SAVE_EPOCH_INTERVAL = 1

CHOSEN_SCALE = LOW_IMBALANCE_SCALE
SCALE_STR = "low_unbalanced" if CHOSEN_SCALE == LOW_IMBALANCE_SCALE else "high_unbalanced"

ALL_METHODS = {
    "original": load_fashion_original,
    "low_unbalanced": lambda bs: load_fashion_imbalanced(bs=bs, scale=LOW_IMBALANCE_SCALE),
    "high_unbalanced": lambda bs: load_fashion_imbalanced(bs=bs, scale=HIGH_IMBALANCE_SCALE),
    # the below ones are meant to be built off of purely the unbalanced datasets
    "downsampled": lambda bs: load_fashion_downsampled(bs=bs, scale=CHOSEN_SCALE),
    "upsampled": lambda bs: load_fashion_upsampled(bs=bs, scale=CHOSEN_SCALE),
    "mlp_cgan_relu": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_mlp_cgan_relu_generator.pt", MLP_CGAN_RELU, bs=bs, scale=CHOSEN_SCALE),
    "small_mlp_cgan_leaky": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_small_mlp_cgan_leaky_generator.pt", SMALL_MLP_CGAN_LEAKY, bs=bs, scale=CHOSEN_SCALE),
    "mlp_cgan_leaky": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_mlp_cgan_leaky_generator.pt", MLP_CGAN_LEAKY, bs=bs, scale=CHOSEN_SCALE),
    "big_mlp_cgan_leaky": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_big_mlp_cgan_leaky_generator.pt", BIG_MLP_CGAN_LEAKY, bs=bs, scale=CHOSEN_SCALE),
    # "mlp_cgan_diff_loss": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_mlp_cgan_diff_loss_generator.pt", None, bs=bs, scale=CHOSEN_SCALE),
    # "dccgan_leaky": lambda bs: load_fashion_gan(f"results_cgan/{SCALE_STR}_dccgan_leaky_generator.pt", DCCGAN, bs=bs, scale=CHOSEN_SCALE),
}


METHOD = "original"


def evaluate(model, dl, lossfn):
    # compute the loss and accuracy of the model
    model = model.to(DEVICE)

    with torch.no_grad():
        all_true = []
        all_pred = []

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

            # add to final results
            all_true.append(batch[1].cpu().reshape(-1))
            all_pred.append(labelshat.reshape(-1))

        # concatenate and compute other metrics
        all_true = torch.concat(all_true).numpy(force=True).astype(np.int64)
        all_pred = torch.concat(all_pred).numpy(force=True).astype(np.int64)

        n = len(dl.dataset)
        acc = acc.item() / n
        loss = loss.item() / n
        f1 = f1_score(all_true, all_pred, average='weighted')

    return acc, loss, f1


def train(METHOD):

    CNN_SAVE_PATH = f"results_cnn/{SCALE_STR}_{METHOD}_cnn.pt"
    FINAL_SAVE_PATH = f"results_cnn/{SCALE_STR}_{METHOD}_results.pkl"

    load_method_ds = ALL_METHODS[METHOD]
    _, train_dl, _, _ = load_method_ds(bs=BATCH_SIZE)

    # NOTE: despite training with different datasets, we would like to evaluate
    #       with the same data so that we are making fair comparisons on
    #       performance
    _, _, _, test_dl = load_fashion_original(bs=BATCH_SIZE)

    # initialize everything for training
    model = CNN().to(DEVICE)

    lossfn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0005)

    step = 0
    num_steps = len(train_dl) * EPOCHS

    # for final plotting
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    train_f1s, test_f1s = [], []

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
            train_acc, train_loss, train_f1 = evaluate(model, train_dl, lossfn)
            test_acc, test_loss, test_f1 = evaluate(model, test_dl, lossfn)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_f1s.append(train_f1)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)

            print(f"Epoch {i}: train_acc={train_acc:.3f} train_loss={train_loss:.3f} train_f1={train_f1:.3f} test_acc={test_acc:.3f} test_loss={test_loss:.3f} test_f1={test_f1:.3f} -- saving model")
            torch.save(model.state_dict(), CNN_SAVE_PATH)

    # save all of the results to a pickle file
    with open(FINAL_SAVE_PATH, "wb") as f:
        d = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "train_f1s": train_f1s,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "test_f1s": test_f1s,
        }
        pickle.dump(d, f)


def main():
    for method in ALL_METHODS.keys():
        print(f"PERFORMING METHOD={method}")
        train(method)


if __name__ == "__main__":
    main()
