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

from torchvision import transforms

from data import *

SPLITS = ["train", "val", "test"]
DATA_DIR = "nerf_synthetic"
CLASSES2IDX = {cls: i for i, cls in enumerate(
    ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"])}
IDX2CLASSES = {i: cls for cls, i in CLASSES2IDX.items()}
SIZE = 224
EPOCHS = 10

def train():
    pass

def main():
    train()

if __name__ == "__main__":
    main()