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
            nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=28800, out_features=500),
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
    trainDS = NERFDataset(DATA_DIR, SPLITS[0])
    validDS = NERFDataset(DATA_DIR, SPLITS[1])
    trainDL = DataLoader(trainDS, batch_size=32, shuffle=False)
    validDL = DataLoader(validDS, batch_size=32, shuffle=False)
    lossfn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, trainDS, validDS, trainDL, validDL, lossfn, opt


def evaluateDL(model, dl, lossfn):

    avgLoss = 0
    acc = 0
    for x, y in dl:
        yhat = model(x)

        # compute the loss
        loss += lossfn(yhat, y)

        # compute the number of correct samples
        labelHat = torch.argmax(yhat, dim=1)
        label = torch.argmax(y, dim=1)
        acc += torch.sum(label == labelHat)

    nsamples = len(dl.dataset)
    avgLoss /= nsamples
    acc /= nsamples

    return acc, avgLoss


def evaluate(model, trainDL, validDL, lossfn):

    with torch.no_grad():
        trainLoss, trainAcc = evaluateDL(model, trainDL, lossfn)
        validLoss, validAcc = evaluateDL(model, validDL, lossfn)

    return trainLoss, trainAcc, validLoss, validAcc


def train(model, trainDL, validDL, lossfn, opt, epochs=EPOCHS):

    trainLosses, trainAccs, validLosses, validAccs = [], [], [], []

    for e in range(EPOCHS):
        for x, y in trainDL:

            # forward pass
            yhat = model(x)

            # backward pass
            opt.zero_grad()
            loss = lossfn(yhat, y)
            loss.backward()
            opt.step()

        # evaluate
        trainLoss, trainAcc, validLoss, validAcc = evaluate(model, trainDL, validDL)
        trainLosses.append(trainLosses)
        trainAccs.append(trainAcc)
        validLosses.append(validLoss)
        validAccs.append(validAcc)

        print(trainLoss, trainAcc, validLoss, validAcc)

    return model, trainLoss, trainAcc, validLoss, validAcc


def main():
    model = CNN()
    model, trainDS, validDS, trainDL, validDL, lossfn, opt = createTrainingUtils(model)

    train(model, trainDL, validDL, lossfn, opt)


if __name__ == "__main__":
    main()
