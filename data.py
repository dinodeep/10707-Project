import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from torchvision import transforms

SPLITS = ["train", "val", "test"]
DATA_DIR = "nerf_synthetic"
CLASSES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
CLASSES2IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX2CLASSES = {i: cls for cls, i in CLASSES2IDX.items()}
SIZE = 224


class NERFDataset(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.size = SIZE

        self.imgPaths = []
        self.labels = []
        self.classPaths = {}
        self.classCounts = {}

        for cls in CLASSES:
            classPath = os.path.join(self.path, cls)
            if os.path.isdir(classPath):
                splitPath = os.path.join(classPath, self.split)
                imgPaths = [os.path.join(splitPath, filename) for filename in os.listdir(splitPath)]

                self.classPaths[cls] = imgPaths
                self.labels += [cls for _ in imgPaths]
                self.imgPaths += imgPaths
                self.classCounts[cls] = len(imgPaths)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        imgPath = self.imgPaths[idx]
        img = Image.open(imgPath)
        img = img.convert('RGB')

        trans = transforms.Compose([
            transforms.Resize((self.size, self.size)), # replace None with Resize Transformation to make shapes uniform,
            transforms.ToTensor(), # convert the PIL Image object into a Tensor
            transforms.ConvertImageDtype(torch.float32),
            # replace None with a Normalize transform (mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5])
            transforms.Normalize(mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5]),
        ])
        img = trans(img)
        img = torch.FloatTensor(img)

        numClasses = len(self.classCounts)
        labelIdx = CLASSES2IDX[self.labels[idx]]
        onehot = F.one_hot(torch.Tensor([labelIdx]).to(torch.int64),
                           num_classes=numClasses).reshape((-1)).to(torch.float32)

        return img, onehot


if __name__ == "__main__":
    # print("collecting data")
    NERFDataset(DATA_DIR, SPLITS[0])
