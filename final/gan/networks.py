import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DEVICE


class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.upscale_factor = upscale_factor
        self.padding = padding

        self.pixelShuffle = nn.PixelShuffle(self.upscale_factor)
        self.conv = nn.Conv2d(self.input_channels, self.n_filters,
                              kernel_size=self.kernel_size, stride=1, padding=self.padding)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return
        #! might be doing something incorrectly here

        # NOTE: are we using repeat or repeat_interleave?
        # (x is batch x channel x height x width)
        x = torch.repeat_interleave(x, int(self.upscale_factor ** 2), dim=1)
        x = self.pixelShuffle(x)
        x = self.conv(x)
        return x


class DownSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.downscale_ratio = downscale_ratio
        self.padding = padding

        self.pixelUnshuffle = nn.PixelUnshuffle(self.downscale_ratio)
        self.conv = nn.Conv2d(self.input_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2 x batch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output
        #! might be doing something incorrectly here
        x = self.pixelUnshuffle(x)
        b, _, h, w = x.shape
        x = torch.split(x, self.input_channels, dim=1)
        x = torch.stack(x)
        x = torch.mean(x, dim=0)
        x = self.conv(x)
        return x


class ResBlockUp(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.input_channels, self.n_filters,
                      kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # NOTE: what would other parameters be here?
            UpSampleConv2D(self.n_filters, kernel_size=self.kernel_size, n_filters=self.n_filters, padding=1)
        )

        self.upsample_residual = UpSampleConv2D(self.input_channels, kernel_size=1, n_filters=self.n_filters)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        return self.layers(x) + self.upsample_residual(x)


class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.input_channels, self.n_filters, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            DownSampleConv2D(self.n_filters, kernel_size=self.kernel_size, n_filters=self.n_filters, padding=1)
        )
        self.downsample_residual = DownSampleConv2D(self.input_channels, kernel_size=1, n_filters=self.n_filters)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        return self.layers(x) + self.downsample_residual(x)


class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.input_channels, self.n_filters, self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=1)
        )

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        return self.layers(x) + x


class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.size = starting_image_size
        self.emb_dim = 50

        # create 10 embeddings (1 for each class), each of shape self.emb_dim
        self.embedding = nn.Embedding(10, self.emb_dim)
        self.dense_emb = nn.Linear(self.emb_dim, self.size * self.size)

        self.dense = nn.Linear(128, 2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(129, kernel_size=3, n_filters=128),
            ResBlockUp(128, kernel_size=3, n_filters=128),
            ResBlockUp(128, kernel_size=3, n_filters=128),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    @torch.jit.script_method
    def forward_given_samples(self, z, labels):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        bs = z.shape[0]
        label_embs = self.embedding(labels)
        label_embs = self.dense_emb(label_embs).reshape((bs, 1, self.size, self.size))

        z = self.dense(z)
        bs, _ = z.shape
        z = z.reshape((bs, 128, self.size, self.size))

        # combine the embedding
        emb = torch.concat([label_embs, z], dim=1)
        y = self.layers(emb)
        return y

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        z = torch.normal(0.0, 1.0, (n_samples, 128)).cpu()
        labels = torch.randint(0, 10, (n_samples,)).cpu()
        return self.forward_given_samples(z, labels), labels


class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers

        self.emb_dim = 50
        self.embedding = nn.Embedding(10, self.emb_dim)
        self.dense_emb = nn.Linear(self.emb_dim, 32*32)

        self.layers = nn.Sequential(
            ResBlockDown(4, kernel_size=3, n_filters=128),
            ResBlockDown(128, kernel_size=3, n_filters=128),
            ResBlock(128, kernel_size=3, n_filters=128),
            ResBlock(128, kernel_size=3, n_filters=128),
            nn.ReLU(),
        )
        # include the plus ten for the one-hot vector
        self.dense_layers = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.Linear(64, 1, bias=True)
        )

    @torch.jit.script_method
    def forward(self, x, label):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        # labels: [B,]
        # embeddings: [B, 10]
        embeddings = self.embedding(label)
        emb_enc = self.dense_emb(embeddings).reshape((-1, 1, 32, 32))

        # x: [B, 3, 32, 32]
        # enc: [B, 4, 32, 32]
        enc = torch.concat([x, emb_enc], dim=1)
        z = self.layers(enc)
        z = z.reshape((z.shape[0], 128, -1))
        z = torch.sum(z, dim=2)

        # make conditional
        # z: [B, 256]
        return self.dense_layers(z)


if __name__ == "__main__":
    # some tests to make sure that shapes work as expected
    d = Discriminator()
    g = Generator()
    results = g.forward(n_samples=10)
    print(results.shape)
    results = d(results)
    print(results.shape)
