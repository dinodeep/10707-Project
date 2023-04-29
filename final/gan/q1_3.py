import argparse
import os
from utils import get_args, DEVICE

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """

    # compute the loss using pytorch functional
    ldisc = F.binary_cross_entropy_with_logits(discrim_real, torch.ones(discrim_real.shape).to(DEVICE))
    lgen = F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros(discrim_fake.shape).to(DEVICE))
    loss = ldisc + lgen
    # print(f"discriminator loss: {loss:.3f} = {ldisc:.3f} + {lgen:.3f}")

    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """

    # compute loss using pytorch functional
    loss = -F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros(discrim_fake.shape).to(DEVICE))
    print(f"generator loss: {loss:.3f}")

    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    prefix = "data_gan/"
    # os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=250,
        amp_enabled=not args.disable_amp,
    )
