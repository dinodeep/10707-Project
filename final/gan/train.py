from glob import glob
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import interpolate_latent_space, save_plot, DEVICE #, get_fid
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset

from networks import Discriminator, Generator


def build_transforms(mu=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    # NOTE: don't do anything fancy for 2, hint: the input image is between 0 and 1.
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K iterations.
    # The learning rate for the generator should be decayed to 0 over 100K iterations.

    # NOTE: did I do the schedulers correctly?
    # NOTE: my version of PyTorch doesn't seem to have the PolynomialLR scheduler, so I will assume that a sufficiently small enough LR is practically 0
    # NOTE: does the learning rate have to be exactly 0?
    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0, 0.9))
    optim_generator = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0, 0.9))

    step_size = 50000
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size, gamma=0.1)
    scheduler_generator = torch.optim.lr_scheduler.StepLR(optim_generator, step_size, gamma=0.1)

    # scheduler_discriminator = torch.optim.lr_scheduler.PolynomialLR(optim_discriminator, total_iters=500000, power=1.0)
    # scheduler_generator = torch.optim.lr_scheduler.PolynomialLR(optim_generator, total_iters=100000, power=1.0)

    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


class CIFARDataset(Dataset):
    def __init__(self):
        transforms = build_transforms()

        # load the CIFAR dataset
        self.ds = CIFAR10("cifar10-data", train=True, transform=transforms, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):

    torch.backends.cudnn.benchmark = True # speed up training

    use_cifar = True
    if use_cifar:
        ds = CIFARDataset()
    else:
        ds = Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms)

    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    use_same_labels = True

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total=num_iterations)
    while iters < num_iterations:
        for batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = batch[0].to(DEVICE)
                target_batch = batch[1].to(DEVICE)
                bs = train_batch.shape[0]
                ############################ UPDATE DISCRIMINATOR ######################################
                # TODO 1.2: compute generator, discriminator and interpolated outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                if not use_same_labels:
                    gen_images, gen_labels = gen(n_samples=train_batch.shape[0])
                    discrim_real = disc(train_batch, target_batch)
                    discrim_fake = disc(gen_images, gen_labels)
                    discrim_interp = None
                    interp = None
                else:
                    z = torch.normal(0.0, 1.0, (bs, 128)).cpu()
                    gen_images = gen.forward_given_samples(z, target_batch)
                    discrim_real = disc(train_batch, target_batch)
                    discrim_fake = disc(gen_images, target_batch)
                    discrim_interp = None
                    interp = None
                # # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # eps = torch.rand((1), dtype=torch.float32).to(DEVICE)
                # interp = eps * train_batch + (1 - eps) * gen_images
                # discrim_interp = disc(interp)

                # NOTE: do I have to compute the discriminator loss here?
                discriminator_loss = disc_loss_fn(discrim_real, discrim_fake, discrim_interp, interp, lamb)

            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # TODO 1.2: compute generator and discriminator output on generated data.
                    if not use_same_labels:
                        gen_images, gen_labels = gen(n_samples=batch_size)
                        discrim_fake = disc(gen_images, gen_labels)
                    else:
                        z = torch.normal(0.0, 1.0, (bs, 128)).cpu()
                        gen_images = gen.forward_given_samples(z, target_batch)
                        discrim_fake = disc(gen_images, target_batch)

                    # NOTE: do I have to compute the generator loss here?
                    generator_loss = gen_loss_fn(discrim_fake)

                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        gen_images, gen_labels = gen(n_samples=batch_size)
                        generated_samples = (gen_images / 2) + 0.5 # move to range [0, 1]

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if os.environ.get('PYTORCH_JIT', 1):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    # fid = get_fid(
                    #     gen,
                    #     dataset_name="cub",
                    #     dataset_resolution=32,
                    #     z_dimension=128,
                    #     batch_size=256,
                    #     num_gen=10_000,
                    # )
                    print(f"Iteration {iters} FID: lmao")
                    #fids_list.append(fid)
                    iters_list.append(iters)
                    '''
                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    '''
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    # fid = get_fid(
    #     gen,
    #     dataset_name="cub",
    #     dataset_resolution=32,
    #     z_dimension=128,
    #     batch_size=256,
    #     num_gen=50_000,
    # )
    print(f"Final FID (Full 50K): hi")


def main():
    ''' print outputs from the model '''

    # load the models
    gen = torch.jit.load("data_gan/generator.pt")

    # display some things
    for cls in range(10):
        n_images = 100
        zs = torch.normal(0.0, 1.0, (n_images, 128))
        labels = torch.ones((n_images,)).to(torch.int64) * cls

        # generate images
        images = gen.forward_given_samples(zs, labels)
        images = (images + 1) / 2

        save_image(images, f"class_generation/image-{cls}.png", nrow=10)


if __name__ == "__main__":
    main()
