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

def createTrainingUtils(model):
  trainDS = NERFDataset(SPLITS[0])
  validDS = NERFDataset(SPLITS[1])
  trainDL = DataLoader(trainDS, batch_size=32, shuffle=False)
  validDL = DataLoader(validDS, batch_size=32, shuffle=False)
  lossfn  = CrossEntropyLoss()
  opt     = SGD(model.parameters(), lr=0.01, momentum=0.9) 
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
      loss = lossfn(y, yhat)
      loss.backward()
      opt.step()

    # evaluate
    trainLoss, trainAcc, validLoss, validAcc = evaluate(model, trainDL, validDL)
    trainLosses.append(trainLosses)
    trainAccs.append(trainAcc)
    validLosses.append(validLoss)
    validAccs.append(validAccs)

  return model, trainLoss, trainAcc, validLoss, validAcc

def main():
    train()

if __name__ == "__main__":
    main()