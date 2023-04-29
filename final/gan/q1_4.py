import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def mseLoss(x, c):
    t = torch.full(x.shape, c).to(torch.float32).cuda()
    return F.mse_loss(x, t)


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.4.1: Implement LSGAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    dloss = mseLoss(discrim_real, 1)
    gloss = mseLoss(discrim_fake, 0)
    loss = (dloss / 2) + (gloss / 2)

    # print(f"discriminator loss: {loss:.3f} = {dloss:.3f} + {gloss:.3f}")
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.4.1: Implement LSGAN loss for generator.
    """
    loss = mseLoss(discrim_fake, 1)
    # print(f"generator loss: {loss:.3f}")
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
