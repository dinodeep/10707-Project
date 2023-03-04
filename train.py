import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from data import *

SPLITS = ["train", "val", "test"]
DATA_DIR = "nerf_synthetic"
CLASSES2IDX = {cls: i for i, cls in enumerate(
    ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"])}
IDX2CLASSES = {i: cls for cls, i in CLASSES2IDX.items()}
SIZE = 224
EPOCHS = 10


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        numChannels = 3
        numClasses = len(CLASSES2IDX)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(11, 11)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(11, 11)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(4, 4))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(6, 6), stride=(4, 4)),
            nn.Flatten(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=50, out_features=500),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=500, out_features=numClasses),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        yhat = self.layer1(x)
        yhat = self.layer2(yhat)
        yhat = self.layer3(yhat)
        yhat = self.layer4(yhat)
        yhat = self.layer5(yhat)
        return yhat


def createTrainingUtils(model):
    trainDS = NERFDatasetSupersampled(DATA_DIR, SPLITS[0])
    validDS = NERFDataset(DATA_DIR, SPLITS[1])
    trainDL = DataLoader(trainDS, batch_size=32, shuffle=True)
    validDL = DataLoader(validDS, batch_size=32, shuffle=True)
    lossfn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, trainDS, validDS, trainDL, validDL, lossfn, opt


def evaluateDL(model, dl, lossfn):

    loss = 0
    ncorrect = 0
    for x, y in dl:
        yhat = model(x)

        # compute the loss
        loss += lossfn(yhat, y) * x.shape[0]

        # compute the number of correct samples
        labelHat = torch.argmax(yhat, dim=1)
        label = torch.argmax(y, dim=1)
        ncorrect += torch.sum(label == labelHat)

    nsamples = len(dl.dataset)
    avgLoss = loss / nsamples
    acc = ncorrect / nsamples

    return avgLoss, acc


def evaluate(model, trainDL, validDL, lossfn):

    with torch.no_grad():
        trainLoss, trainAcc = evaluateDL(model, trainDL, lossfn)
        validLoss, validAcc = evaluateDL(model, validDL, lossfn)

    return trainLoss, trainAcc, validLoss, validAcc


def train(model, trainDL, validDL, lossfn, opt, epochs=EPOCHS):

    trainLosses, trainAccs, validLosses, validAccs = [], [], [], []

    for e in range(EPOCHS):
        print(len(trainDL))
        for i, (x, y) in enumerate(trainDL):

            # forward pass
            yhat = model(x)
            loss = lossfn(yhat, y)
            # backward pass
            opt.zero_grad()
            
            loss.backward()
            opt.step()

            # print(f"\t{i} done")

        # evaluate
        trainLoss, trainAcc, validLoss, validAcc = evaluate(model, trainDL, validDL, lossfn)
        trainLosses.append(trainLoss.item())
        trainAccs.append(trainAcc.item())
        validLosses.append(validLoss.item())
        validAccs.append(validAcc.item())

        print(trainLoss, trainAcc, validLoss, validAcc)

    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, trainLosses, "r", label="Train")
    plt.plot(epochs, validLosses, "b", label="Validation")
    plt.title("Train Loss on Supersampled Data/Valid Loss on Balanced Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("raw_data_loss.png")
    plt.show()
    return model, trainLoss, trainAcc, validLoss, validAcc


def main():
    model = CNN()
    model, trainDS, validDS, trainDL, validDL, lossfn, opt = createTrainingUtils(model)

    train(model, trainDL, validDL, lossfn, opt)


if __name__ == "__main__":
    main()
