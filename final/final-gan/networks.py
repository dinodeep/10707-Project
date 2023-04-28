import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 100


class MLP_CGAN_RELU(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.layers(x)
        return output.reshape((-1, 1, 28, 28))


class SMALL_MLP_CGAN_LEAKY(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.layers(x)
        return output.reshape((-1, 1, 28, 28))


class MLP_CGAN_LEAKY(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.layers(x)
        return output.reshape((-1, 1, 28, 28))


class BIG_MLP_CGAN_LEAKY(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 384),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(384),
            nn.Linear(384, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 768),
            nn.LeakyReLU(0.2),
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.layers(x)
        return output.reshape((-1, 1, 28, 28))


class DCCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + 10
        self.in_channels = 1
        self.img_start_size = 7
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.layers1 = nn.Sequential(
            nn.Linear(input_dim, self.in_channels * self.img_start_size ** 2),
            nn.LeakyReLU(0.2),
        )

        # reshape, then continue
        self.layers2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.layers1(x)
        output = output.reshape((-1, 1, 7, 7))
        output = self.layers2(output)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        input_dim = 784 + 10
        output_dim = 1
        self.label_embedding = nn.Embedding(10, 10)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.reshape((-1, 1*28*28))
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)

        return output.to(device)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layers(x)
