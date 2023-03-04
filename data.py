import os
from PIL import Image
import numpy as np
import random

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
CLASSES2COUNT = {"chair": 100, "drums": 50, "ficus": 25,
                 "hotdog": 10, "lego": 10, "materials": 10, "mic": 10, "ship": 10}
CLASSES2IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX2CLASSES = {i: cls for cls, i in CLASSES2IDX.items()}
SIZE = 224


class NERFDataset(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.size = SIZE

        self.loadData()

    def loadData(self):
        self.imgPaths = []
        self.labels = []
        self.classPaths = {}
        self.classCounts = {}

        for cls in CLASSES:
            classPath = os.path.join(self.path, cls)
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
            # replace None with Resize Transformation to make shapes uniform,
            transforms.Resize((self.size, self.size)),
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


class NERFDatasetLongTail(NERFDataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.size = SIZE

        self.loadData()

    def loadData(self):
        self.imgPaths = []
        self.labels = []
        self.classPaths = {}
        self.classCounts = {}

        for cls in CLASSES:
            classPath = os.path.join(self.path, cls)
            splitPath = os.path.join(classPath, self.split)
            imgPaths = [os.path.join(splitPath, filename) for filename in os.listdir(splitPath)]

            random.shuffle(imgPaths)
            imgPaths = imgPaths[:CLASSES2COUNT[cls]]

            self.classPaths[cls] = imgPaths
            self.labels += [cls for _ in imgPaths]
            self.imgPaths += imgPaths
            self.classCounts[cls] = len(imgPaths)


class NERFDatasetSubsampled(NERFDataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.size = SIZE

        self.loadData()

    def loadData(self):
        self.imgPaths = []
        self.labels = []
        self.classPaths = {}
        self.classCounts = {}

        minCount = min([count for count in CLASSES2COUNT.values()])

        for cls in CLASSES:
            classPath = os.path.join(self.path, cls)
            splitPath = os.path.join(classPath, self.split)
            imgPaths = [os.path.join(splitPath, filename) for filename in os.listdir(splitPath)]

            random.shuffle(imgPaths)
            imgPaths = imgPaths[:minCount]

            self.classPaths[cls] = imgPaths
            self.labels += [cls for _ in imgPaths]
            self.imgPaths += imgPaths
            self.classCounts[cls] = len(imgPaths)


class NERFDatasetSupersampled(NERFDataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.size = SIZE

        self.loadData()

    def loadData(self):
        self.imgPaths = []
        self.labels = []
        self.classPaths = {}
        self.classCounts = {}

        maxCount = max([count for count in CLASSES2COUNT.values()])

        for cls in CLASSES:
            classPath = os.path.join(self.path, cls)
            splitPath = os.path.join(classPath, self.split)
            imgPaths = [os.path.join(splitPath, filename) for filename in os.listdir(splitPath)]

            superSampled = imgPaths
            while len(superSampled) < maxCount:
                superSampled += imgPaths[:len(imgPaths) - maxCount]
            imgPaths = superSampled
            random.shuffle(imgPaths)

            self.classPaths[cls] = imgPaths
            self.labels += [cls for _ in imgPaths]
            self.imgPaths += imgPaths
            self.classCounts[cls] = len(imgPaths)


if __name__ == "__main__":
    # print("collecting data")
    dss = [
        NERFDataset(DATA_DIR, SPLITS[0]),
        NERFDatasetLongTail(DATA_DIR, SPLITS[0]),
        NERFDatasetSubsampled(DATA_DIR, SPLITS[0]),
        NERFDatasetSupersampled(DATA_DIR, SPLITS[0])
    ]

    for ds in dss:
        print(type(ds))
        for cls, paths in ds.classPaths.items():
            print("\t", cls, len(paths))
